import os
import json
import secrets
import logging

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    send_file,
    session,
    flash,
)

from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from joblib import dump, load


# =========================================================
# APPLICATION CONFIGURATION
# =========================================================

app = Flask(__name__)

app.secret_key = os.environ.get("SECRET_KEY") or secrets.token_hex(32)

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
GRAPH_DIR = os.path.join(BASE_DIR, "static", "graphs")
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")

CUSTOMERS_FILE = os.path.join(DATASET_DIR, "customers.csv")
CLUSTERED_FILE = os.path.join(DATASET_DIR, "clustered_customers.csv")
MODEL_FILE = os.path.join(DATASET_DIR, "kmeans_model.joblib")
METADATA_FILE = os.path.join(DATASET_DIR, "model_metadata.json")
USERS_FILE = os.path.join(BASE_DIR, "users.json")

RESULTS_GRAPH = os.path.join(GRAPH_DIR, "results.png")

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(GRAPH_DIR, exist_ok=True)

app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=bool(os.environ.get("RENDER")),
)

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


# =========================================================
# REQUIRED DATASET COLUMNS
# =========================================================

INCOME_COLUMN = "Annual Income (k$)"
SPENDING_COLUMN = "Spending Score (1-100)"

FEATURE_COLUMNS = [
    INCOME_COLUMN,
    SPENDING_COLUMN,
]

REQUIRED_COLUMNS = [
    "CustomerID",
    "Gender",
    "Age",
    INCOME_COLUMN,
    SPENDING_COLUMN,
]


# =========================================================
# AUTHENTICATION
# =========================================================

def login_required(function):
    @wraps(function)
    def decorated_function(*args, **kwargs):
        if "user" not in session:
            flash("Please log in to continue.", "warning")
            return redirect(url_for("login"))

        return function(*args, **kwargs)

    return decorated_function


def read_users():
    """Read registered users from the JSON file."""

    if not os.path.exists(USERS_FILE):
        return {}

    try:
        with open(USERS_FILE, "r", encoding="utf-8") as file:
            users = json.load(file)

        return users if isinstance(users, dict) else {}

    except (json.JSONDecodeError, OSError):
        logger.exception("Unable to read users.json")
        return {}


def write_users(users):
    """Save registered users to the JSON file."""

    temporary_file = USERS_FILE + ".tmp"

    with open(temporary_file, "w", encoding="utf-8") as file:
        json.dump(users, file, indent=4)

    os.replace(temporary_file, USERS_FILE)


# =========================================================
# DATASET HELPERS
# =========================================================

def read_customer_csv(file_path):
    """Read a CSV file and normalize its column names."""

    df = pd.read_csv(file_path)

    df.columns = df.columns.astype(str).str.strip()

    return df


def validate_dataset(df):
    """Validate required customer columns and numeric values."""

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    if df.empty:
        raise ValueError("The uploaded CSV file contains no customer records.")

    for column in [
        "Age",
        INCOME_COLUMN,
        SPENDING_COLUMN,
    ]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    df.dropna(
        subset=[
            INCOME_COLUMN,
            SPENDING_COLUMN,
        ],
        inplace=True,
    )

    if df.empty:
        raise ValueError(
            "No valid income and spending values were found."
        )

    if len(df) < 2:
        raise ValueError(
            "At least two valid customers are required for clustering."
        )

    if (df[INCOME_COLUMN] < 0).any():
        raise ValueError("Annual income cannot contain negative values.")

    if not df[SPENDING_COLUMN].between(1, 100).all():
        raise ValueError(
            "Spending Score values must be between 1 and 100."
        )

    return df.reset_index(drop=True)


def get_segment_mapping(df):
    """
    Map cluster numbers to readable segment names.

    For three clusters:
        Lowest average spending  -> Low Spending
        Middle average spending  -> Medium Spending
        Highest average spending -> High Spending

    For other cluster counts, use Group 1, Group 2, etc.
    """

    if "Cluster" not in df.columns:
        raise ValueError("The dataset does not contain a Cluster column.")

    if SPENDING_COLUMN not in df.columns:
        raise ValueError(
            f"The dataset does not contain {SPENDING_COLUMN}."
        )

    cluster_means = (
        df.groupby("Cluster")[SPENDING_COLUMN]
        .mean()
        .sort_values()
    )

    cluster_ids = cluster_means.index.tolist()

    if len(cluster_ids) == 3:
        names = [
            "Low Spending",
            "Medium Spending",
            "High Spending",
        ]

    else:
        names = [
            f"Group {number}"
            for number in range(1, len(cluster_ids) + 1)
        ]

    return dict(zip(cluster_ids, names))


def get_valid_clustered_data():
    """Read and validate the previously clustered customer dataset."""

    if not os.path.exists(CLUSTERED_FILE):
        raise FileNotFoundError(
            "Clustered data was not found. Run K-Means clustering first."
        )

    df = read_customer_csv(CLUSTERED_FILE)

    missing_columns = [
        column
        for column in FEATURE_COLUMNS + ["Cluster"]
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "The clustered dataset is missing: "
            + ", ".join(missing_columns)
        )

    df[INCOME_COLUMN] = pd.to_numeric(
        df[INCOME_COLUMN],
        errors="coerce",
    )

    df[SPENDING_COLUMN] = pd.to_numeric(
        df[SPENDING_COLUMN],
        errors="coerce",
    )

    df["Cluster"] = pd.to_numeric(
        df["Cluster"],
        errors="coerce",
    )

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    df.dropna(
        subset=FEATURE_COLUMNS + ["Cluster"],
        inplace=True,
    )

    if df.empty:
        raise ValueError("No valid clustered customer records were found.")

    df["Cluster"] = df["Cluster"].astype(int)

    mapping = get_segment_mapping(df)

    df["Segment"] = df["Cluster"].map(mapping)

    return df.reset_index(drop=True)


def create_results_graph(df):
    """Generate a scatter plot of customer clusters."""

    plt.figure(figsize=(10, 6))

    for cluster_id in sorted(df["Cluster"].unique()):
        cluster_df = df[df["Cluster"] == cluster_id]

        plt.scatter(
            cluster_df[INCOME_COLUMN],
            cluster_df[SPENDING_COLUMN],
            label=f"Cluster {cluster_id}",
            alpha=0.75,
        )

    plt.xlabel("Annual Income (k$)")
    plt.ylabel("Spending Score (1-100)")
    plt.title("Customer Segmentation Using K-Means")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()

    plt.savefig(
        RESULTS_GRAPH,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()


def load_model_metadata():
    """Read model metadata if it exists."""

    if not os.path.exists(METADATA_FILE):
        return {}

    try:
        with open(METADATA_FILE, "r", encoding="utf-8") as file:
            return json.load(file)

    except (json.JSONDecodeError, OSError):
        logger.exception("Unable to read model metadata")
        return {}


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
def home():
    return render_template("home.html")


# =========================================================
# DASHBOARD
# =========================================================

@app.route("/index")
@login_required
def index():
    try:
        if os.path.exists(CLUSTERED_FILE):
            df = read_customer_csv(CLUSTERED_FILE)

        elif os.path.exists(CUSTOMERS_FILE):
            df = read_customer_csv(CUSTOMERS_FILE)

        else:
            df = pd.DataFrame()

        total_customers = len(df)

        total_columns = len(df.columns)

        average_age = 0
        average_income = 0
        average_spending = 0

        if "Age" in df.columns:
            average_age = round(
                pd.to_numeric(df["Age"], errors="coerce").mean(),
                2,
            )

        if INCOME_COLUMN in df.columns:
            average_income = round(
                pd.to_numeric(
                    df[INCOME_COLUMN],
                    errors="coerce",
                ).mean(),
                2,
            )

        if SPENDING_COLUMN in df.columns:
            average_spending = round(
                pd.to_numeric(
                    df[SPENDING_COLUMN],
                    errors="coerce",
                ).mean(),
                2,
            )

        return render_template(
            "index.html",
            total_customers=total_customers,
            total_columns=total_columns,
            average_age=average_age,
            average_income=average_income,
            average_spending=average_spending,
        )

    except Exception:
        logger.exception("Dashboard failed to load")

        flash("Unable to load the dashboard.", "danger")

        return render_template(
            "index.html",
            total_customers=0,
            total_columns=0,
            average_age=0,
            average_income=0,
            average_spending=0,
        )


# =========================================================
# UPLOAD DATASET
# =========================================================

@app.route("/upload", methods=["GET", "POST"])
@login_required
def upload():
    if request.method == "POST":

        uploaded_file = request.files.get("file")

        if not uploaded_file or not uploaded_file.filename:
            flash("Please select a CSV file.", "warning")
            return redirect(url_for("upload"))

        filename = secure_filename(uploaded_file.filename)

        if not filename.lower().endswith(".csv"):
            flash("Only CSV files are supported.", "danger")
            return redirect(url_for("upload"))

        try:
            df = pd.read_csv(uploaded_file)

            df.columns = df.columns.astype(str).str.strip()

            df = validate_dataset(df)

            df.to_csv(CUSTOMERS_FILE, index=False)

            # A new dataset invalidates previous clustering results.
            for old_file in [
                CLUSTERED_FILE,
                MODEL_FILE,
                METADATA_FILE,
                RESULTS_GRAPH,
            ]:
                if os.path.exists(old_file):
                    os.remove(old_file)

            flash(
                f"Dataset uploaded successfully: {len(df)} valid customers.",
                "success",
            )

            return redirect(url_for("analysis"))

        except Exception as error:
            logger.exception("Dataset upload failed")

            flash(str(error), "danger")

            return redirect(url_for("upload"))

    return render_template("upload.html")


# =========================================================
# DATA ANALYSIS
# =========================================================

@app.route("/analysis")
@login_required
def analysis():
    try:
        if not os.path.exists(CUSTOMERS_FILE):
            flash("Please upload a customer CSV file first.", "warning")
            return redirect(url_for("upload"))

        df = read_customer_csv(CUSTOMERS_FILE)

        df = validate_dataset(df)

        total_customers = len(df)
        total_columns = len(df.columns)

        mean_age = round(df["Age"].mean(), 2)
        mean_income = round(df[INCOME_COLUMN].mean(), 2)
        mean_spending = round(df[SPENDING_COLUMN].mean(), 2)

        missing_values = int(df.isna().sum().sum())

        customer_table = df.head(20).to_html(
            index=False,
            classes="table table-hover table-striped",
            border=0,
            na_rep="-",
            escape=True,
        )

        return render_template(
            "analysis.html",
            total_customers=total_customers,
            total_columns=total_columns,
            mean_age=mean_age,
            mean_income=mean_income,
            mean_spending=mean_spending,
            missing_values=missing_values,
            customer_table=customer_table,
        )

    except Exception:
        logger.exception("Analysis page failed")

        flash(
            "Unable to analyze the dataset. Check the CSV columns.",
            "danger",
        )

        return redirect(url_for("index"))


# =========================================================
# K-MEANS CLUSTERING
# =========================================================

@app.route("/kmeans", methods=["GET", "POST"])
@login_required
def kmeans_page():
    if request.method == "POST":

        if not os.path.exists(CUSTOMERS_FILE):
            flash("Please upload a dataset first.", "warning")
            return redirect(url_for("upload"))

        try:
            k = int(request.form.get("clusters", 3))

            if k < 2:
                flash("The number of clusters must be at least 2.", "danger")
                return redirect(url_for("kmeans_page"))

            df = read_customer_csv(CUSTOMERS_FILE)

            df = validate_dataset(df)

            if k >= len(df):
                flash(
                    "The number of clusters must be smaller than "
                    "the number of customers.",
                    "danger",
                )
                return redirect(url_for("kmeans_page"))

            features = df[FEATURE_COLUMNS].astype(float)

            model = KMeans(
                n_clusters=k,
                random_state=42,
                n_init=10,
            )

            df["Cluster"] = model.fit_predict(features)

            mapping = get_segment_mapping(df)

            df["Segment"] = df["Cluster"].map(mapping)

            # Save clustered customer data.
            df.to_csv(CLUSTERED_FILE, index=False)

            # Save the trained model.
            dump(model, MODEL_FILE)

            metadata = {
                "feature_columns": FEATURE_COLUMNS,
                "n_clusters": k,
                "random_state": 42,
                "n_init": 10,
                "segment_mapping": {
                    str(key): value
                    for key, value in mapping.items()
                },
            }

            with open(
                METADATA_FILE,
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(metadata, file, indent=4)

            create_results_graph(df)

            flash(
                f"K-Means clustering completed successfully "
                f"with {k} clusters.",
                "success",
            )

            return redirect(url_for("results"))

        except Exception:
            logger.exception("K-Means clustering failed")

            flash(
                "Clustering failed. Check your dataset and application logs.",
                "danger",
            )

            return redirect(url_for("kmeans_page"))

    return render_template("kmeans.html")


# =========================================================
# CLUSTERING RESULTS
# =========================================================

@app.route("/results")
@login_required
def results():
    try:
        df = get_valid_clustered_data()

        customer_table = df.to_html(
            index=False,
            classes="table table-hover table-striped",
            border=0,
            na_rep="-",
            escape=True,
        )

        cluster_counts = (
            df["Segment"]
            .value_counts()
            .to_dict()
        )

        return render_template(
            "results.html",
            customer_table=customer_table,
            total_customers=len(df),
            cluster_counts=cluster_counts,
            results_graph=url_for(
                "static",
                filename="graphs/results.png",
            ),
        )

    except FileNotFoundError:
        flash("Please run K-Means clustering first.", "warning")
        return redirect(url_for("kmeans_page"))

    except Exception:
        logger.exception("Results page failed")

        flash("Unable to load clustering results.", "danger")

        return redirect(url_for("index"))


# =========================================================
# CUSTOMER SEGMENTS
# =========================================================

@app.route("/segments")
@login_required
def segments():
    try:
        df = get_valid_clustered_data()

        segment_counts = df["Segment"].value_counts().to_dict()

        low_count = int(segment_counts.get("Low Spending", 0))
        medium_count = int(segment_counts.get("Medium Spending", 0))
        high_count = int(segment_counts.get("High Spending", 0))

        total_customers = len(df)

        # Convert to JSON for safe use by Chart.js.
        chart_labels = list(segment_counts.keys())

        chart_values = [
            int(segment_counts[label])
            for label in chart_labels
        ]

        customer_table = df.to_html(
            index=False,
            classes="table table-hover table-striped align-middle",
            border=0,
            na_rep="-",
            escape=True,
        )

        return render_template(
            "segments.html",
            low_count=low_count,
            medium_count=medium_count,
            high_count=high_count,
            total_customers=total_customers,
            segment_counts=segment_counts,
            chart_labels=chart_labels,
            chart_values=chart_values,
            customer_table=customer_table,
        )

    except FileNotFoundError:
        flash("Please run K-Means clustering first.", "warning")
        return redirect(url_for("kmeans_page"))

    except Exception:
        logger.exception("Customer segments page failed")

        flash(
            "Unable to load customer segments. Check the application logs.",
            "danger",
        )

        return redirect(url_for("index"))


# =========================================================
# DOWNLOAD CLUSTERED DATASET
# =========================================================

@app.route("/download")
@login_required
def download():
    if not os.path.exists(CLUSTERED_FILE):
        flash("Please run K-Means clustering before downloading.", "warning")
        return redirect(url_for("kmeans_page"))

    return send_file(
        CLUSTERED_FILE,
        as_attachment=True,
        download_name="clustered_customers.csv",
        mimetype="text/csv",
    )


# =========================================================
# CUSTOMER PREDICTION
# =========================================================

@app.route("/prediction", methods=["GET", "POST"])
@login_required
def prediction():
    prediction_result = None
    predicted_cluster = None

    form_values = {
        "gender": "",
        "age": "",
        "income": "",
        "spending": "",
    }

    if request.method == "POST":

        form_values = {
            "gender": request.form.get("gender", "").strip(),
            "age": request.form.get("age", "").strip(),
            "income": request.form.get("income", "").strip(),
            "spending": request.form.get("spending", "").strip(),
        }

        try:
            age = float(form_values["age"])
            income = float(form_values["income"])
            spending = float(form_values["spending"])

            if not 0 <= age <= 120:
                raise ValueError("Please enter an age between 0 and 120.")

            if income < 0:
                raise ValueError("Annual income cannot be negative.")

            if not 1 <= spending <= 100:
                raise ValueError(
                    "Spending score must be between 1 and 100."
                )

            if not os.path.exists(CLUSTERED_FILE):
                flash(
                    "Run K-Means clustering before making predictions.",
                    "warning",
                )
                return redirect(url_for("kmeans_page"))

            if not os.path.exists(MODEL_FILE):
                flash(
                    "The saved model was not found. Run K-Means again.",
                    "warning",
                )
                return redirect(url_for("kmeans_page"))

            model = load(MODEL_FILE)

            input_df = pd.DataFrame(
                [[income, spending]],
                columns=FEATURE_COLUMNS,
            )

            predicted_cluster = int(model.predict(input_df)[0])

            clustered_df = get_valid_clustered_data()

            mapping = get_segment_mapping(clustered_df)

            prediction_result = mapping.get(
                predicted_cluster,
                f"Group {predicted_cluster + 1}",
            )

            flash(
                f"Customer segment predicted: {prediction_result}",
                "success",
            )

        except ValueError as error:
            flash(str(error), "danger")

        except Exception:
            logger.exception("Customer prediction failed")

            flash(
                "Prediction failed. Run K-Means again and try once more.",
                "danger",
            )

    return render_template(
        "prediction.html",
        prediction=prediction_result,
        cluster=predicted_cluster,
        form_values=form_values,
    )


# =========================================================
# REGISTER
# =========================================================

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":

        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        if not name or not email or not password:
            flash("Please complete all required fields.", "warning")
            return redirect(url_for("register"))

        if len(password) < 8:
            flash(
                "Password must contain at least 8 characters.",
                "warning",
            )
            return redirect(url_for("register"))

        try:
            users = read_users()

            if email in users:
                flash("An account with this email already exists.", "danger")
                return redirect(url_for("register"))

            users[email] = {
                "name": name,
                "password": generate_password_hash(password),
            }

            write_users(users)

            flash(
                "Registration successful. Please log in.",
                "success",
            )

            return redirect(url_for("login"))

        except Exception:
            logger.exception("Registration failed")

            flash(
                "Unable to register your account. Please try again.",
                "danger",
            )

    return render_template("register.html")


# =========================================================
# LOGIN
# =========================================================

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":

        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        users = read_users()

        user = users.get(email)

        if user and check_password_hash(
            user.get("password", ""),
            password,
        ):
            session.clear()

            session["user"] = email

            session["user_name"] = user.get("name", email)

            flash("Login successful.", "success")

            return redirect(url_for("index"))

        flash("Invalid email or password.", "danger")

    return render_template("login.html")


# =========================================================
# LOGOUT
# =========================================================

@app.route("/logout")
@login_required
def logout():
    session.clear()

    flash("You have been logged out.", "success")

    return redirect(url_for("home"))


# =========================================================
# ERROR HANDLERS
# =========================================================

@app.errorhandler(413)
def file_too_large(error):
    flash("The uploaded file is too large. Maximum size is 10 MB.", "danger")
    return redirect(url_for("upload"))


@app.errorhandler(404)
def page_not_found(error):
    return (
        render_template(
            "404.html",
        ),
        404,
    ) if os.path.exists(
        os.path.join(TEMPLATE_DIR, "404.html")
    ) else (
        "Page not found.",
        404,
    )


@app.errorhandler(500)
def internal_server_error(error):
    logger.exception("Internal server error")

    return (
        "An internal server error occurred. Check the application logs.",
        500,
    )


# =========================================================
# RUN APPLICATION LOCALLY
# =========================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
    )