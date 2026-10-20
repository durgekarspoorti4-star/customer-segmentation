from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    send_file,
    session,
    flash
)

import os
import io
import json
import secrets
import logging

import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from werkzeug.security import (
    generate_password_hash,
    check_password_hash
)
from werkzeug.utils import secure_filename
from functools import wraps


# ============================================================
# APPLICATION CONFIGURATION
# ============================================================

app = Flask(__name__)

app.config["SECRET_KEY"] = os.environ.get(
    "SECRET_KEY",
    "local-development-key-change-before-deployment"
)

app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = bool(os.environ.get("RENDER"))
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

logging.basicConfig(level=logging.INFO)


# ============================================================
# DIRECTORIES
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
GRAPH_DIR = os.path.join(BASE_DIR, "static", "graphs")

USERS_FILE = os.path.join(BASE_DIR, "users.json")

CUSTOMERS_FILE = os.path.join(DATASET_DIR, "customers.csv")
CLUSTERED_FILE = os.path.join(
    DATASET_DIR,
    "clustered_customers.csv"
)

MODEL_FILE = os.path.join(DATASET_DIR, "kmeans_model.joblib")
METADATA_FILE = os.path.join(DATASET_DIR, "model_metadata.json")

RESULT_GRAPH = os.path.join(GRAPH_DIR, "results.png")

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(GRAPH_DIR, exist_ok=True)


# ============================================================
# DATASET CONFIGURATION
# ============================================================

INCOME_COLUMN = "Annual Income (k$)"
SPENDING_COLUMN = "Spending Score (1-100)"

FEATURE_COLUMNS = [
    INCOME_COLUMN,
    SPENDING_COLUMN
]

REQUIRED_COLUMNS = [
    "Age",
    INCOME_COLUMN,
    SPENDING_COLUMN
]

SEGMENT_NAMES = [
    "Low Spending",
    "Medium Spending",
    "High Spending"
]


# ============================================================
# AUTHENTICATION DECORATOR
# ============================================================

def login_required(view):
    @wraps(view)
    def wrapped_view(*args, **kwargs):
        if "user" not in session:
            return redirect(url_for("login"))

        return view(*args, **kwargs)

    return wrapped_view


# ============================================================
# GENERAL HELPER FUNCTIONS
# ============================================================

def read_csv_file(path):
    """
    Read a CSV file and normalize column names.
    """
    df = pd.read_csv(path)
    df.columns = df.columns.astype(str).str.strip()
    return df


def clean_numeric_columns(df, columns):
    """
    Convert selected columns to numeric values.
    Invalid values become NaN.
    """
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    return df


def get_segment_mapping(df):
    """
    Map cluster IDs to segment names based on average spending.

    For three clusters:
        Lowest average spending -> Low Spending
        Middle average spending -> Medium Spending
        Highest average spending -> High Spending

    For other cluster counts, use Group 1, Group 2, etc.
    """
    if "Cluster" not in df.columns:
        return {}

    if SPENDING_COLUMN not in df.columns:
        return {}

    working = df[["Cluster", SPENDING_COLUMN]].copy()

    working = clean_numeric_columns(
        working,
        ["Cluster", SPENDING_COLUMN]
    )

    working = working.dropna(
        subset=["Cluster", SPENDING_COLUMN]
    )

    if working.empty:
        return {}

    working["Cluster"] = working["Cluster"].astype(int)

    averages = (
        working.groupby("Cluster")[SPENDING_COLUMN]
        .mean()
        .sort_values()
    )

    clusters = list(averages.index)

    if len(clusters) == 3:
        return {
            clusters[0]: "Low Spending",
            clusters[1]: "Medium Spending",
            clusters[2]: "High Spending"
        }

    return {
        cluster: f"Group {index + 1}"
        for index, cluster in enumerate(clusters)
    }


def add_segment_column(df):
    """
    Add a Segment column based on cluster spending averages.
    """
    df = df.copy()

    if "Cluster" not in df.columns:
        return df

    mapping = get_segment_mapping(df)

    df["Segment"] = pd.to_numeric(
        df["Cluster"],
        errors="coerce"
    ).map(mapping)

    return df


def get_valid_clustered_data():
    """
    Read and validate clustered customer data.
    """
    if not os.path.exists(CLUSTERED_FILE):
        return None

    df = read_csv_file(CLUSTERED_FILE)

    if "Cluster" not in df.columns and "cluster" in df.columns:
        df.rename(
            columns={"cluster": "Cluster"},
            inplace=True
        )

    required = FEATURE_COLUMNS + ["Cluster"]

    if not all(column in df.columns for column in required):
        return None

    df = clean_numeric_columns(df, required)

    df = df.dropna(
        subset=required
    ).copy()

    if df.empty:
        return None

    df["Cluster"] = df["Cluster"].astype(int)

    df = add_segment_column(df)

    return df


def save_plot():
    """
    Create the customer segmentation scatter plot.
    """
    df = get_valid_clustered_data()

    if df is None:
        return False

    plt.figure(figsize=(10, 6))

    sns.scatterplot(
        data=df,
        x=INCOME_COLUMN,
        y=SPENDING_COLUMN,
        hue="Segment",
        palette="viridis",
        s=90
    )

    plt.title("Customer Segmentation Using K-Means")
    plt.xlabel("Annual Income (k$)")
    plt.ylabel("Spending Score (1-100)")
    plt.grid(alpha=0.2)
    plt.tight_layout()

    plt.savefig(
        RESULT_GRAPH,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    return True


def create_html_table(df, max_rows=None):
    """
    Convert a DataFrame to an HTML table for the templates.
    """
    display_df = df.copy()

    if max_rows is not None:
        display_df = display_df.head(max_rows)

    display_df = display_df.astype(object).where(
        pd.notna(display_df),
        ""
    )

    return display_df.to_html(
        classes="table table-striped table-hover table-bordered",
        index=False,
        border=0,
        escape=True
    )


def load_users():
    """
    Load registered users from users.json.
    """
    if not os.path.exists(USERS_FILE):
        return []

    try:
        with open(USERS_FILE, "r", encoding="utf-8") as file:
            users = json.load(file)

        return users if isinstance(users, list) else []

    except (json.JSONDecodeError, OSError):
        app.logger.exception("Unable to read users file")
        return []


def save_users(users):
    """
    Save users to users.json.
    """
    with open(USERS_FILE, "w", encoding="utf-8") as file:
        json.dump(users, file, indent=4)


def get_filtered_customers():
    """
    Apply the customer search filters and return the filtered DataFrame.

    This helper is shared by the search page and CSV export.
    """
    df = get_valid_clustered_data()

    if df is None:
        return None

    customer_id = request.args.get(
        "customer_id",
        ""
    ).strip()

    selected_segment = request.args.get(
        "segment",
        "All"
    ).strip()

    min_age = request.args.get(
        "min_age",
        ""
    ).strip()

    max_age = request.args.get(
        "max_age",
        ""
    ).strip()

    min_income = request.args.get(
        "min_income",
        ""
    ).strip()

    max_income = request.args.get(
        "max_income",
        ""
    ).strip()

    numeric_columns = [
        "CustomerID",
        "Age",
        INCOME_COLUMN,
        SPENDING_COLUMN
    ]

    df = clean_numeric_columns(
        df,
        numeric_columns
    )

    if customer_id and "CustomerID" in df.columns:
        df = df[
            df["CustomerID"].astype("Int64").astype(str)
            .str.contains(customer_id, na=False)
        ]

    if selected_segment and selected_segment != "All":
        df = df[df["Segment"] == selected_segment]

    try:
        if min_age and "Age" in df.columns:
            df = df[df["Age"] >= float(min_age)]

        if max_age and "Age" in df.columns:
            df = df[df["Age"] <= float(max_age)]

        if min_income:
            df = df[df[INCOME_COLUMN] >= float(min_income)]

        if max_income:
            df = df[df[INCOME_COLUMN] <= float(max_income)]

    except ValueError:
        raise ValueError(
            "Please enter valid numeric filter values."
        )

    filters = {
        "customer_id": customer_id,
        "selected_segment": selected_segment or "All",
        "min_age": min_age,
        "max_age": max_age,
        "min_income": min_income,
        "max_income": max_income
    }

    return df, filters


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():
    return render_template("home.html")


# ============================================================
# DASHBOARD
# ============================================================

@app.route("/index")
@login_required
def index():

    total_customers = 0
    total_clusters = 0

    average_income = 0
    average_spending = 0

    customer_labels = []
    customer_counts = []
    segment_spending = []

    table_columns = []
    customer_rows = []

    graph_exists = os.path.exists(RESULT_GRAPH)

    df = get_valid_clustered_data()

    if df is not None:

        total_customers = len(df)

        average_income = round(
            float(df[INCOME_COLUMN].mean()),
            2
        )

        average_spending = round(
            float(df[SPENDING_COLUMN].mean()),
            2
        )

        summary = (
            df.groupby("Segment")
            .agg(
                Customer_Count=("Cluster", "size"),
                Average_Spending=(SPENDING_COLUMN, "mean")
            )
        )

        summary = summary.sort_values(
            "Average_Spending"
        )

        total_clusters = len(summary)

        customer_labels = summary.index.tolist()

        customer_counts = (
            summary["Customer_Count"]
            .astype(int)
            .tolist()
        )

        segment_spending = (
            summary["Average_Spending"]
            .round(2)
            .tolist()
        )

        table_df = df.head(10).copy()

        table_df = table_df.astype(object).where(
            pd.notna(table_df),
            ""
        )

        table_columns = table_df.columns.tolist()
        customer_rows = table_df.values.tolist()

    return render_template(
        "index.html",
        total_customers=total_customers,
        total_clusters=total_clusters,
        average_income=average_income,
        average_spending=average_spending,
        graph_exists=graph_exists,
        customer_labels=customer_labels,
        customer_counts=customer_counts,
        segment_spending=segment_spending,
        table_columns=table_columns,
        customer_rows=customer_rows
    )


# ============================================================
# DATASET UPLOAD
# ============================================================

@app.route("/upload", methods=["GET", "POST"])
@login_required
def upload():

    if request.method == "POST":

        uploaded_file = request.files.get("file")

        if not uploaded_file or not uploaded_file.filename:
            flash("Please select a CSV file.", "warning")
            return redirect(url_for("upload"))

        safe_filename = secure_filename(
            uploaded_file.filename
        )

        if not safe_filename.lower().endswith(".csv"):
            flash("Only CSV files are supported.", "danger")
            return redirect(url_for("upload"))

        try:
            df = pd.read_csv(uploaded_file)

            df.columns = df.columns.astype(str).str.strip()

            missing_columns = [
                column
                for column in REQUIRED_COLUMNS
                if column not in df.columns
            ]

            if missing_columns:
                flash(
                    "Missing required columns: "
                    + ", ".join(missing_columns),
                    "danger"
                )
                return redirect(url_for("upload"))

            if df.empty:
                flash("The uploaded dataset is empty.", "danger")
                return redirect(url_for("upload"))

            df = clean_numeric_columns(
                df,
                REQUIRED_COLUMNS
            )

            df = df.dropna(
                subset=REQUIRED_COLUMNS
            )

            if df.empty:
                flash(
                    "No valid customer records were found.",
                    "danger"
                )
                return redirect(url_for("upload"))

            if (df[SPENDING_COLUMN] < 1).any() or (
                df[SPENDING_COLUMN] > 100
            ).any():
                flash(
                    "Spending Score values must be between 1 and 100.",
                    "danger"
                )
                return redirect(url_for("upload"))

            if (df["Age"] < 1).any() or (df["Age"] > 100).any():
                flash(
                    "Age values must be between 1 and 100.",
                    "danger"
                )
                return redirect(url_for("upload"))

            if (df[INCOME_COLUMN] < 0).any():
                flash(
                    "Annual income cannot be negative.",
                    "danger"
                )
                return redirect(url_for("upload"))

            df.to_csv(
                CUSTOMERS_FILE,
                index=False
            )

            # New data invalidates old clustering results and model.
            for path in [
                CLUSTERED_FILE,
                MODEL_FILE,
                METADATA_FILE,
                RESULT_GRAPH
            ]:
                if os.path.exists(path):
                    os.remove(path)

            flash(
                "Dataset uploaded successfully.",
                "success"
            )

            return redirect(url_for("analysis"))

        except (pd.errors.ParserError, UnicodeDecodeError, ValueError):
            flash(
                "Unable to read this CSV file. Please check its format.",
                "danger"
            )
            return redirect(url_for("upload"))

        except Exception:
            app.logger.exception("Dataset upload failed")
            flash(
                "An unexpected error occurred while uploading the dataset.",
                "danger"
            )
            return redirect(url_for("upload"))

    return render_template("upload.html")


# ============================================================
# DATA ANALYSIS
# ============================================================

@app.route("/analysis")
@login_required
def analysis():

    if not os.path.exists(CUSTOMERS_FILE):
        flash("Upload a dataset before viewing the analysis.", "warning")
        return redirect(url_for("upload"))

    try:
        df = read_csv_file(CUSTOMERS_FILE)

        missing_columns = [
            column
            for column in REQUIRED_COLUMNS
            if column not in df.columns
        ]

        if missing_columns:
            return (
                "Dataset is missing required columns: "
                + ", ".join(missing_columns),
                400
            )

        df = clean_numeric_columns(
            df,
            REQUIRED_COLUMNS
        )

        total_customers = len(df)
        total_columns = len(df.columns)

        average_age = round(
            float(df["Age"].mean()),
            2
        ) if df["Age"].notna().any() else 0

        average_income = round(
            float(df[INCOME_COLUMN].mean()),
            2
        ) if df[INCOME_COLUMN].notna().any() else 0

        average_spending = round(
            float(df[SPENDING_COLUMN].mean()),
            2
        ) if df[SPENDING_COLUMN].notna().any() else 0

        missing_values = int(
            df.isnull().sum().sum()
        )

        table = create_html_table(
            df,
            max_rows=20
        )

        return render_template(
            "analysis.html",
            total_customers=total_customers,
            total_columns=total_columns,
            average_age=average_age,
            average_income=average_income,
            average_spending=average_spending,
            missing_values=missing_values,
            table=table
        )

    except Exception:
        app.logger.exception("Data analysis failed")
        return "Unable to analyze the dataset. Check the application logs.", 500


# ============================================================
# K-MEANS CLUSTERING
# ============================================================

@app.route("/kmeans", methods=["GET", "POST"])
@login_required
def kmeans_page():

    if request.method == "POST":

        try:
            k = int(request.form.get("k", 3))
        except (ValueError, TypeError):
            return "Invalid number of clusters.", 400

        if not 2 <= k <= 10:
            return "Choose between 2 and 10 clusters.", 400

        if not os.path.exists(CUSTOMERS_FILE):
            flash("Please upload a dataset first.", "warning")
            return redirect(url_for("upload"))

        try:
            df = read_csv_file(CUSTOMERS_FILE)

            if not all(
                column in df.columns
                for column in FEATURE_COLUMNS
            ):
                return "The dataset is missing required clustering columns.", 400

            df = clean_numeric_columns(
                df,
                FEATURE_COLUMNS
            )

            df = df.dropna(
                subset=FEATURE_COLUMNS
            ).copy()

            if len(df) < k:
                return (
                    "The number of clusters cannot exceed "
                    "the number of valid customers.",
                    400
                )

            model = KMeans(
                n_clusters=k,
                random_state=42,
                n_init=10
            )

            df["Cluster"] = model.fit_predict(
                df[FEATURE_COLUMNS]
            )

            # Save the exact fitted model for future predictions.
            import joblib

            joblib.dump(
                model,
                MODEL_FILE
            )

            with open(
                METADATA_FILE,
                "w",
                encoding="utf-8"
            ) as file:

                json.dump(
                    {
                        "features": FEATURE_COLUMNS,
                        "n_clusters": k,
                        "random_state": 42,
                        "n_init": 10
                    },
                    file,
                    indent=4
                )

            df.to_csv(
                CLUSTERED_FILE,
                index=False
            )

            save_plot()

            flash(
                f"K-Means clustering completed with {k} clusters.",
                "success"
            )

            return redirect(url_for("results"))

        except Exception:
            app.logger.exception("K-Means clustering failed")
            return "Unable to perform K-Means clustering.", 500

    return render_template(
        "kmeans.html",
        result=False
    )


# ============================================================
# RESULTS PAGE
# ============================================================

@app.route("/results")
@login_required
def results():

    df = get_valid_clustered_data()

    if df is None:
        flash("Run K-Means clustering to view results.", "warning")
        return redirect(url_for("kmeans_page"))

    try:
        summary = (
            df.groupby(["Cluster", "Segment"])
            .agg(
                Customer_Count=("Cluster", "size"),
                Average_Income=(INCOME_COLUMN, "mean"),
                Average_Spending=(SPENDING_COLUMN, "mean")
            )
            .reset_index()
        )

        summary["Average_Income"] = (
            summary["Average_Income"].round(2)
        )

        summary["Average_Spending"] = (
            summary["Average_Spending"].round(2)
        )

        save_plot()

        summary_table = create_html_table(summary)
        customer_table = create_html_table(df)

        return render_template(
            "results.html",
            summary_table=summary_table,
            customer_table=customer_table
        )

    except Exception:
        app.logger.exception("Unable to display clustering results")
        return "Unable to display the clustering results.", 500


# ============================================================
# DOWNLOAD CLUSTERED DATASET
# ============================================================

@app.route("/download")
@login_required
def download():

    if not os.path.exists(CLUSTERED_FILE):
        flash("Run K-Means before downloading results.", "warning")
        return redirect(url_for("kmeans_page"))

    return send_file(
        CLUSTERED_FILE,
        as_attachment=True,
        download_name="customer_segmentation_results.csv"
    )


# ============================================================
# CUSTOMER PREDICTION
# ============================================================

@app.route("/prediction", methods=["GET", "POST"])
@login_required
def prediction():

    prediction_result = None
    cluster_result = None

    if request.method == "POST":

        try:
            gender = request.form.get(
                "gender",
                ""
            ).strip().lower()

            age = float(request.form.get("age", ""))
            income = float(request.form.get("income", ""))
            spending = float(request.form.get("spending", ""))

            if gender not in ("male", "female", "m", "f"):
                return "Please select Male or Female.", 400

            if not 1 <= age <= 100:
                return "Age must be between 1 and 100.", 400

            if income < 0:
                return "Income cannot be negative.", 400

            if not 1 <= spending <= 100:
                return "Spending Score must be between 1 and 100.", 400

        except (ValueError, TypeError):
            return "Please enter valid customer details.", 400

        if not all(
            os.path.exists(path)
            for path in [
                MODEL_FILE,
                METADATA_FILE,
                CLUSTERED_FILE
            ]
        ):
            flash(
                "Run K-Means clustering before making predictions.",
                "warning"
            )
            return redirect(url_for("kmeans_page"))

        try:
            import joblib

            model = joblib.load(MODEL_FILE)

            with open(
                METADATA_FILE,
                "r",
                encoding="utf-8"
            ) as file:
                metadata = json.load(file)

            df = get_valid_clustered_data()

            if df is None:
                return "The saved clustering dataset is invalid.", 400

            feature_columns = metadata.get(
                "features",
                FEATURE_COLUMNS
            )

            if feature_columns != FEATURE_COLUMNS:
                return "The saved model features are incompatible.", 400

            new_customer = pd.DataFrame(
                [{
                    INCOME_COLUMN: income,
                    SPENDING_COLUMN: spending
                }],
                columns=feature_columns
            )

            # Use the exact model fitted in the K-Means route.
            cluster_result = int(
                model.predict(new_customer)[0]
            )

            segment_map = get_segment_mapping(df)

            prediction_result = segment_map.get(
                cluster_result,
                "Unknown Segment"
            )

        except Exception:
            app.logger.exception("Prediction failed")
            return "Unable to predict the customer segment.", 500

    return render_template(
        "prediction.html",
        prediction=prediction_result,
        cluster=cluster_result
    )


# ============================================================
# CUSTOMER SEGMENTS
# ============================================================

@app.route("/segments")
@login_required
def segments():

    df = get_valid_clustered_data()

    if df is None:
        flash("Run K-Means before viewing customer segments.", "warning")
        return redirect(url_for("kmeans_page"))

    low_count = int(
        (df["Segment"] == "Low Spending").sum()
    )

    medium_count = int(
        (df["Segment"] == "Medium Spending").sum()
    )

    high_count = int(
        (df["Segment"] == "High Spending").sum()
    )

    customer_table = create_html_table(df)

    return render_template(
        "segments.html",
        low_count=low_count,
        medium_count=medium_count,
        high_count=high_count,
        customer_table=customer_table
    )


# ============================================================
# SEARCH AND FILTER CUSTOMERS
# ============================================================

@app.route("/search", methods=["GET", "POST"])
@login_required
def search_customers():

    if not os.path.exists(CLUSTERED_FILE):
        flash("Run K-Means before searching customers.", "warning")
        return redirect(url_for("kmeans_page"))

    try:
        result = get_filtered_customers()

        if result is None:
            return "The clustered dataset is invalid.", 400

        df, filters = result

        customer_table = create_html_table(df)

        return render_template(
            "search.html",
            customer_table=customer_table,
            result_count=len(df),
            customer_id=filters["customer_id"],
            selected_segment=filters["selected_segment"],
            min_age=filters["min_age"],
            max_age=filters["max_age"],
            min_income=filters["min_income"],
            max_income=filters["max_income"]
        )

    except ValueError as error:
        return str(error), 400

    except Exception:
        app.logger.exception("Customer search failed")
        return "Unable to search customers.", 500


# ============================================================
# DOWNLOAD FILTERED CUSTOMERS
# ============================================================

@app.route("/download-filtered")
@login_required
def download_filtered():

    if not os.path.exists(CLUSTERED_FILE):
        flash("Run K-Means before downloading filtered customers.", "warning")
        return redirect(url_for("kmeans_page"))

    try:
        result = get_filtered_customers()

        if result is None:
            return "The clustered dataset is invalid.", 400

        df, filters = result

        buffer = io.BytesIO()

        df.to_csv(
            buffer,
            index=False
        )

        buffer.seek(0)

        return send_file(
            buffer,
            mimetype="text/csv",
            as_attachment=True,
            download_name="filtered_customer_segments.csv"
        )

    except ValueError as error:
        return str(error), 400

    except Exception:
        app.logger.exception("Filtered download failed")
        return "Unable to download filtered customers.", 500


# ============================================================
# CUSTOMER INSIGHTS
# ============================================================

@app.route("/insights")
@login_required
def insights():

    df = get_valid_clustered_data()

    if df is None:
        flash("Run K-Means before viewing insights.", "warning")
        return redirect(url_for("kmeans_page"))

    summary = (
        df.groupby("Segment", as_index=False)
        .agg(
            Customers=("Cluster", "size"),
            Average_Income=(INCOME_COLUMN, "mean"),
            Average_Spending=(SPENDING_COLUMN, "mean")
        )
    )

    summary["Average_Income"] = (
        summary["Average_Income"].round(2)
    )

    summary["Average_Spending"] = (
        summary["Average_Spending"].round(2)
    )

    insights_list = []

    for _, row in summary.iterrows():

        segment = row["Segment"]

        if segment == "Low Spending":
            recommendation = (
                "Consider targeted offers, loyalty rewards, "
                "and customer feedback to improve engagement."
            )

        elif segment == "Medium Spending":
            recommendation = (
                "Consider personalized product suggestions, "
                "bundles, and loyalty programs."
            )

        elif segment == "High Spending":
            recommendation = (
                "Consider VIP rewards, exclusive offers, "
                "and personalized recommendations."
            )

        else:
            recommendation = (
                "Study this group's purchasing behavior "
                "before choosing a marketing strategy."
            )

        insights_list.append({
            "segment": segment,
            "customers": int(row["Customers"]),
            "income": float(row["Average_Income"]),
            "spending": float(row["Average_Spending"]),
            "recommendation": recommendation
        })

    summary_table = create_html_table(summary)

    return render_template(
        "insights.html",
        total_customers=len(df),
        total_segments=df["Segment"].nunique(),
        summary_table=summary_table,
        insights_list=insights_list
    )


# ============================================================
# REGISTRATION
# ============================================================

@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        name = request.form.get(
            "name",
            ""
        ).strip()

        email = request.form.get(
            "email",
            ""
        ).strip().lower()

        password = request.form.get(
            "password",
            ""
        )

        if not name or not email or not password:
            flash("Please complete all registration fields.", "warning")
            return redirect(url_for("register"))

        if len(password) < 8:
            flash(
                "Password must contain at least 8 characters.",
                "warning"
            )
            return redirect(url_for("register"))

        users = load_users()

        if any(
            user.get("email", "").lower() == email
            for user in users
        ):
            flash("This email is already registered.", "warning")
            return redirect(url_for("register"))

        users.append({
            "name": name,
            "email": email,
            "password": generate_password_hash(password)
        })

        try:
            save_users(users)

        except OSError:
            app.logger.exception("Registration failed")
            return "Unable to save the new account.", 500

        flash(
            "Registration successful. Please log in.",
            "success"
        )

        return redirect(url_for("login"))

    return render_template("register.html")


# ============================================================
# LOGIN
# ============================================================

@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        email = request.form.get(
            "email",
            ""
        ).strip().lower()

        password = request.form.get(
            "password",
            ""
        )

        users = load_users()

        for user in users:

            if user.get("email", "").lower() != email:
                continue

            stored_password = user.get("password", "")

            try:
                valid_password = check_password_hash(
                    stored_password,
                    password
                )
            except (ValueError, TypeError):
                valid_password = False

            # Migrate legacy plaintext passwords after successful login.
            if (
                not valid_password
                and stored_password
                and stored_password == password
            ):
                user["password"] = generate_password_hash(password)

                try:
                    save_users(users)
                    valid_password = True
                except OSError:
                    app.logger.exception(
                        "Unable to upgrade legacy password"
                    )

            if valid_password:

                session.clear()

                session["user"] = user.get(
                    "name",
                    email
                )

                return redirect(url_for("index"))

        flash(
            "Invalid email or password.",
            "danger"
        )

        return redirect(url_for("login"))

    return render_template("login.html")


# ============================================================
# LOGOUT
# ============================================================

@app.route("/logout")
def logout():

    session.clear()

    return redirect(url_for("home"))


# ============================================================
# ERROR HANDLERS
# ============================================================

@app.errorhandler(413)
def file_too_large(error):
    flash(
        "The uploaded file is too large. Maximum size is 10 MB.",
        "danger"
    )
    return redirect(url_for("upload"))


@app.errorhandler(404)
def page_not_found(error):
    return (
        render_template(
            "404.html"
        ) if os.path.exists(
            os.path.join(BASE_DIR, "templates", "404.html")
        ) else "Page not found.",
        404
    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )