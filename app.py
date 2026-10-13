from flask import Flask, render_template, request, redirect, url_for, send_file, session
import pandas as pd
import matplotlib
matplotlib.use('Agg') # FIX for Render - no display
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import os
import json

app = Flask(__name__)

app.config["SECRET_KEY"] = os.environ.get(
    "SECRET_KEY",
    "customer-segmentation-local-dev-key"
)

app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=bool(os.environ.get("RENDER"))
)

# === RENDER PATH FIX ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
GRAPH_DIR = os.path.join(BASE_DIR, "static", "graphs")
USERS_FILE = os.path.join(BASE_DIR, "users.json")

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(GRAPH_DIR, exist_ok=True)
# =======================

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/index")
def index():
    if "user" not in session:
        return redirect(url_for("login"))
    file_path = os.path.join(DATASET_DIR, "clustered_customers.csv")
    total_customers = 0
    total_clusters = 0
    average_income = 0
    average_spending = 0
    graph_exists = os.path.exists(os.path.join(GRAPH_DIR, "results.png"))

    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        total_customers = len(df)
        if "Cluster" in df.columns:
            total_clusters = df["Cluster"].nunique()
        if "Annual Income (k$)" in df.columns:
            df["Annual Income (k$)"] = pd.to_numeric(df["Annual Income (k$)"], errors="coerce")
            average_income = round(df["Annual Income (k$)"].mean(), 2)
        if "Spending Score (1-100)" in df.columns:
            df["Spending Score (1-100)"] = pd.to_numeric(df["Spending Score (1-100)"], errors="coerce")
            average_spending = round(df["Spending Score (1-100)"].mean(), 2)

    return render_template("index.html", total_customers=total_customers, total_clusters=total_clusters, average_income=average_income, average_spending=average_spending, graph_exists=graph_exists)

#... keep all your other routes same but replace every "dataset/" with os.path.join(DATASET_DIR,...)
# and "static/graphs" with GRAPH_DIR
# and "users.json" with USERS_FILE

# I will give you full corrected file below - copy everything:

@app.route("/upload", methods=["GET", "POST"])
def upload():
    if "user" not in session:
        return redirect(url_for("login"))
    if request.method == "POST":
        file = request.files.get("file")
        if file and file.filename.endswith(".csv"):
            file.save(os.path.join(DATASET_DIR, "customers.csv"))
            return redirect(url_for("index"))
    return render_template("upload.html")

@app.route("/analysis")
def analysis():
    print("DEBUG: Analysis route was called")

    if "user" not in session:
        return redirect(url_for("login"))

    # Keep the rest of your existing function here
    file_path = os.path.join(DATASET_DIR, "customers.csv")

    if not os.path.exists(file_path):
        return redirect(url_for("upload"))

    try:
        df = pd.read_csv(file_path)

        # Clean column names
        df.columns = df.columns.str.strip()

        required_columns = [
            "Age",
            "Annual Income (k$)",
            "Spending Score (1-100)"
        ]

        # Check for required columns
        missing_columns = [
            column for column in required_columns
            if column not in df.columns
        ]

        if missing_columns:
            return (
                "Dataset is missing required columns: "
                + ", ".join(missing_columns),
                400
            )

        # Convert values to numbers
        for column in required_columns:
            df[column] = pd.to_numeric(
                df[column], errors="coerce"
            )

        total_customers = len(df)
        total_columns = len(df.columns)

        average_age = round(df["Age"].mean(), 2)
        average_income = round(
            df["Annual Income (k$)"].mean(), 2
        )
        average_spending = round(
            df["Spending Score (1-100)"].mean(), 2
        )

        missing_values = int(df.isnull().sum().sum())

        table = df.head(20).to_html(
            classes="table table-striped table-hover",
            index=False,
            border=0
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

    except Exception as e:
        app.logger.exception("Error in Data Analysis page")
        return f"Error analyzing dataset: {str(e)}", 500
@app.route("/kmeans", methods=["GET", "POST"])
def kmeans_page():
    if "user" not in session:
        return redirect(url_for("login"))
    if request.method == "POST":
        try:
            k = int(request.form["k"])
        except:
            return "Invalid number of clusters"
        if k < 2:
            return "Number of clusters must be at least 2"
        file_path = os.path.join(DATASET_DIR, "customers.csv")
        if not os.path.exists(file_path):
            return redirect(url_for("upload"))
        df = pd.read_csv(file_path)
        df["Annual Income (k$)"] = pd.to_numeric(df["Annual Income (k$)"], errors="coerce")
        df["Spending Score (1-100)"] = pd.to_numeric(df["Spending Score (1-100)"], errors="coerce")
        df = df.dropna(subset=["Annual Income (k$)", "Spending Score (1-100)"])
        if len(df) < k:
            return "Number of clusters is greater than available customers."
        features = df[["Annual Income (k$)", "Spending Score (1-100)"]]
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        df["Cluster"] = model.fit_predict(features)
        df.to_csv(os.path.join(DATASET_DIR, "clustered_customers.csv"), index=False)
        return render_template("kmeans.html", result=True, k=k)
    return render_template("kmeans.html", result=False)

@app.route("/results")
def results():
    if "user" not in session:
        return redirect(url_for("login"))
    file_path = os.path.join(DATASET_DIR, "clustered_customers.csv")
    if not os.path.exists(file_path):
        return redirect(url_for("kmeans_page"))
    df = pd.read_csv(file_path)
    df["Annual Income (k$)"] = pd.to_numeric(df["Annual Income (k$)"], errors="coerce")
    df["Spending Score (1-100)"] = pd.to_numeric(df["Spending Score (1-100)"], errors="coerce")
    df["Cluster"] = pd.to_numeric(df["Cluster"], errors="coerce")
    df = df.dropna(subset=["Annual Income (k$)", "Spending Score (1-100)", "Cluster"])
    df["Cluster"] = df["Cluster"].astype(int)
    cluster_average = df.groupby("Cluster")["Spending Score (1-100)"].mean().sort_values()
    clusters = list(cluster_average.index)
    segment_names = {}
    if len(clusters) == 3:
        segment_names[clusters[0]] = "Low Spending"
        segment_names[clusters[1]] = "Medium Spending"
        segment_names[clusters[2]] = "High Spending"
    else:
        for i, cluster in enumerate(clusters):
            segment_names[cluster] = f"Group {i + 1}"
    df["Segment"] = df["Cluster"].map(segment_names)
    summary = df.groupby(["Cluster", "Segment"]).agg(Customer_Count=("Cluster", "size"), Average_Income=("Annual Income (k$)", "mean"), Average_Spending=("Spending Score (1-100)", "mean")).reset_index()
    summary["Average_Income"] = summary["Average_Income"].round(2)
    summary["Average_Spending"] = summary["Average_Spending"].round(2)
    plt.figure(figsize=(9, 6))
    sns.scatterplot(data=df, x="Annual Income (k$)", y="Spending Score (1-100)", hue="Segment", s=100)
    plt.title("Customer Segmentation using K-Means")
    plt.savefig(os.path.join(GRAPH_DIR, "results.png"))
    plt.close()
    df.to_csv(os.path.join(DATASET_DIR, "clustered_customers.csv"), index=False)
    summary_table = summary.to_html(classes="table table-bordered table-striped", index=False)
    customer_table = df.to_html(classes="table table-bordered table-striped", index=False)
    return render_template("results.html", summary_table=summary_table, customer_table=customer_table)

@app.route("/download")
def download():
    file_path = os.path.join(DATASET_DIR, "clustered_customers.csv")
    if not os.path.exists(file_path):
        return redirect(url_for("kmeans_page"))
    return send_file(file_path, as_attachment=True, download_name="customer_segmentation_results.csv")

@app.route("/elbow")
def elbow():
    if "user" not in session:
        return redirect(url_for("login"))
    file_path = os.path.join(DATASET_DIR, "customers.csv")
    if not os.path.exists(file_path):
        return redirect(url_for("upload"))
    df = pd.read_csv(file_path)
    df["Annual Income (k$)"] = pd.to_numeric(df["Annual Income (k$)"], errors="coerce")
    df["Spending Score (1-100)"] = pd.to_numeric(df["Spending Score (1-100)"], errors="coerce")
    df = df.dropna(subset=["Annual Income (k$)", "Spending Score (1-100)"])
    features = df[["Annual Income (k$)", "Spending Score (1-100)"]]
    inertia = []
    k_values = range(2, 11)
    for k in k_values:
        if k > len(features):
            break
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        model.fit(features)
        inertia.append(model.inertia_)
    plt.figure(figsize=(8, 5))
    plt.plot(list(k_values)[:len(inertia)], inertia, marker="o")
    plt.title("Elbow Method for Optimal K")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Inertia")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_DIR, "elbow.png"))
    plt.close()
    return render_template("elbow.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    if "user" not in session:
        return redirect(url_for("login"))

    prediction_result = None
    cluster_result = None

    file_path = os.path.join(DATASET_DIR, "clustered_customers.csv")

    if request.method == "POST":
        try:
            gender = request.form["gender"].strip().lower()
            age = float(request.form["age"])
            income = float(request.form["income"])
            spending = float(request.form["spending"])

            if gender not in ["male", "female", "m", "f"]:
                return "Please select Male or Female."

            if not 1 <= age <= 100:
                return "Please enter an age between 1 and 100."

            if income < 0:
                return "Income cannot be negative."

            if not 1 <= spending <= 100:
                return "Spending Score must be between 1 and 100."

        except (ValueError, KeyError):
            return "Please enter valid customer details."

        # Require the existing K-Means results
        if not os.path.exists(file_path):
            return redirect(url_for("kmeans_page"))

        df = pd.read_csv(file_path)

        required_columns = [
            "Annual Income (k$)",
            "Spending Score (1-100)",
            "Cluster"
        ]

        if not all(col in df.columns for col in required_columns):
            return "Please run K-Means again before making predictions."

        # Clean dataset
        for col in [
            "Annual Income (k$)",
            "Spending Score (1-100)"
        ]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df["Cluster"] = pd.to_numeric(
            df["Cluster"], errors="coerce"
        )

        df = df.dropna(subset=required_columns)

        if len(df) < 3:
            return "At least 3 valid customers are required."

        df["Cluster"] = df["Cluster"].astype(int)

        # Use the same features as the K-Means page
        feature_columns = [
            "Annual Income (k$)",
            "Spending Score (1-100)"
        ]

        features = df[feature_columns]

        # Train the model using the same settings as K-Means
        k = df["Cluster"].nunique()

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        model.fit(features)

        # Match the newly trained cluster labels to saved labels
        from scipy.optimize import linear_sum_assignment
        import numpy as np

        saved_labels = df["Cluster"].to_numpy()
        fitted_labels = model.labels_

        saved_clusters = sorted(df["Cluster"].unique())
        fitted_clusters = sorted(np.unique(fitted_labels))

        cost_matrix = np.zeros(
            (len(fitted_clusters), len(saved_clusters))
        )

        for i, fitted_cluster in enumerate(fitted_clusters):
            for j, saved_cluster in enumerate(saved_clusters):
                cost_matrix[i, j] = -np.sum(
                    (fitted_labels == fitted_cluster)
                    & (saved_labels == saved_cluster)
                )

        rows, cols = linear_sum_assignment(cost_matrix)

        cluster_mapping = {
            fitted_clusters[i]: saved_clusters[j]
            for i, j in zip(rows, cols)
        }

        # Predict the new customer's cluster
        new_customer = pd.DataFrame([{
            "Annual Income (k$)": income,
            "Spending Score (1-100)": spending
        }], columns=feature_columns)

        fitted_cluster = int(model.predict(new_customer)[0])
        cluster_result = int(cluster_mapping[fitted_cluster])

        # Determine the segment using the saved cluster results
        cluster_average = (
            df.groupby("Cluster")["Spending Score (1-100)"]
            .mean()
            .sort_values()
        )

        clusters = list(cluster_average.index)

        if len(clusters) == 3:
            segment_names = {
                clusters[0]: "Low Spending",
                clusters[1]: "Medium Spending",
                clusters[2]: "High Spending"
            }
        else:
            segment_names = {
                cluster: f"Group {i + 1}"
                for i, cluster in enumerate(clusters)
            }

        prediction_result = segment_names.get(
            cluster_result, "Unknown Segment"
        )

    return render_template(
        "prediction.html",
        prediction=prediction_result,
        cluster=cluster_result
    )
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form["name"]
        email = request.form["email"]
        password = request.form["password"]
        users = []
        if os.path.exists(USERS_FILE):
            with open(USERS_FILE, "r") as file:
                try:
                    users = json.load(file)
                except:
                    users = []
        users.append({"name": name, "email": email, "password": password})
        with open(USERS_FILE, "w") as file:
            json.dump(users, file, indent=4)
        return redirect(url_for("login"))
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        if os.path.exists(USERS_FILE):
            with open(USERS_FILE, "r") as file:
                try:
                    users = json.load(file)
                except:
                    users = []
            for user in users:
                if user["email"] == email and user["password"] == password:
                    session["user"] = user["name"]
                    return redirect(url_for("index"))
        return "Invalid email or password"
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))

@app.route("/segments")
def segments():
    if "user" not in session:
        return redirect(url_for("login"))
    file_path = os.path.join(DATASET_DIR, "clustered_customers.csv")
    if not os.path.exists(file_path):
        return redirect(url_for("kmeans_page"))
    df = pd.read_csv(file_path)
    if "Segment" not in df.columns:
        cluster_average = df.groupby("Cluster")["Spending Score (1-100)"].mean().sort_values()
        clusters = list(cluster_average.index)
        segment_names = {}
        if len(clusters) == 3:
            segment_names[clusters[0]] = "Low Spending"
            segment_names[clusters[1]] = "Medium Spending"
            segment_names[clusters[2]] = "High Spending"
        else:
            for i, cluster in enumerate(clusters):
                segment_names[cluster] = f"Group {i + 1}"
        df["Segment"] = df["Cluster"].map(segment_names)
    low_count = len(df[df["Segment"] == "Low Spending"])
    medium_count = len(df[df["Segment"] == "Medium Spending"])
    high_count = len(df[df["Segment"] == "High Spending"])
    customer_table = df.to_html(classes="table table-bordered table-striped", index=False)
    return render_template("segments.html", low_count=low_count, medium_count=medium_count, high_count=high_count, customer_table=customer_table)
@app.route("/search", methods=["GET", "POST"])
def search_customers():
    if "user" not in session:
        return redirect(url_for("login"))

    file_path = os.path.join(
        DATASET_DIR,
        "clustered_customers.csv"
    )

    if not os.path.exists(file_path):
        return redirect(url_for("kmeans_page"))

    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()

    # Create segment labels if they do not exist
    if "Segment" not in df.columns:
        if "Cluster" not in df.columns:
            return "Please run K-Means before searching.", 400

        df["Cluster"] = pd.to_numeric(
            df["Cluster"], errors="coerce"
        )

        df["Spending Score (1-100)"] = pd.to_numeric(
            df["Spending Score (1-100)"],
            errors="coerce"
        )

        averages = (
            df.groupby("Cluster")["Spending Score (1-100)"]
            .mean()
            .sort_values()
        )

        clusters = list(averages.index)

        if len(clusters) == 3:
            segment_map = {
                clusters[0]: "Low Spending",
                clusters[1]: "Medium Spending",
                clusters[2]: "High Spending"
            }
        else:
            segment_map = {
                cluster: f"Group {i + 1}"
                for i, cluster in enumerate(clusters)
            }

        df["Segment"] = df["Cluster"].map(segment_map)

    # Read filters from the form
    search_id = request.values.get("customer_id", "").strip()
    selected_segment = request.values.get("segment", "All")
    min_age = request.values.get("min_age", "").strip()
    max_age = request.values.get("max_age", "").strip()
    min_income = request.values.get("min_income", "").strip()
    max_income = request.values.get("max_income", "").strip()

    # Convert numerical columns safely
    numeric_columns = [
        "CustomerID",
        "Age",
        "Annual Income (k$)",
        "Spending Score (1-100)"
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column], errors="coerce"
            )

    # Apply filters
    if search_id and "CustomerID" in df.columns:
        df = df[
            df["CustomerID"].astype("Int64").astype(str)
            .str.contains(search_id, na=False)
        ]

    if selected_segment != "All":
        df = df[df["Segment"] == selected_segment]

    try:
        if min_age and "Age" in df.columns:
            df = df[df["Age"] >= float(min_age)]

        if max_age and "Age" in df.columns:
            df = df[df["Age"] <= float(max_age)]

        if min_income and "Annual Income (k$)" in df.columns:
            df = df[
                df["Annual Income (k$)"] >= float(min_income)
            ]

        if max_income and "Annual Income (k$)" in df.columns:
            df = df[
                df["Annual Income (k$)"] <= float(max_income)
            ]

    except ValueError:
        return "Please enter valid numeric filter values.", 400

    customer_table = df.to_html(
        classes="table table-striped table-hover",
        index=False,
        border=0
    )

    return render_template(
        "search.html",
        customer_table=customer_table,
        result_count=len(df),
        customer_id=search_id,
        selected_segment=selected_segment,
        min_age=min_age,
        max_age=max_age,
        min_income=min_income,
        max_income=max_income
    )


@app.route("/download-filtered")
def download_filtered():
    if "user" not in session:
        return redirect(url_for("login"))

    # Apply the same filters through the search route's logic
    # in the next iteration; this endpoint exports the full
    # clustered dataset for now.
    file_path = os.path.join(
        DATASET_DIR,
        "clustered_customers.csv"
    )

    if not os.path.exists(file_path):
        return redirect(url_for("kmeans_page"))

    return send_file(
        file_path,
        as_attachment=True,
        download_name="customer_segmentation_results.csv"
    )
@app.route("/insights")
def insights():
    if "user" not in session:
        return redirect(url_for("login"))

    file_path = os.path.join(
        DATASET_DIR,
        "clustered_customers.csv"
    )

    if not os.path.exists(file_path):
        return redirect(url_for("kmeans_page"))

    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()

    required = [
        "Cluster",
        "Annual Income (k$)",
        "Spending Score (1-100)"
    ]

    if not all(col in df.columns for col in required):
        return "Please run K-Means before viewing customer insights.", 400

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=required)

    if df.empty:
        return "No valid customer records found.", 400

    # Determine segment names from average spending
    averages = (
        df.groupby("Cluster")["Spending Score (1-100)"]
        .mean()
        .sort_values()
    )

    clusters = list(averages.index)

    if len(clusters) == 3:
        segment_map = {
            clusters[0]: "Low Spending",
            clusters[1]: "Medium Spending",
            clusters[2]: "High Spending"
        }
    else:
        segment_map = {
            cluster: f"Group {i + 1}"
            for i, cluster in enumerate(clusters)
        }

    df["Segment"] = df["Cluster"].map(segment_map)

    # Create summary statistics
    summary = (
        df.groupby("Segment", as_index=False)
        .agg(
            Customers=("Cluster", "size"),
            Average_Income=("Annual Income (k$)", "mean"),
            Average_Spending=("Spending Score (1-100)", "mean")
        )
    )

    summary["Average_Income"] = summary["Average_Income"].round(2)
    summary["Average_Spending"] = summary["Average_Spending"].round(2)

    insights_list = []

    for _, row in summary.iterrows():
        segment = row["Segment"]
        count = int(row["Customers"])
        income = row["Average_Income"]
        spending = row["Average_Spending"]

        if segment == "Low Spending":
            recommendation = (
                "Consider targeted offers, loyalty rewards, "
                "and customer feedback to encourage engagement."
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
                "before selecting a marketing strategy."
            )

        insights_list.append({
            "segment": segment,
            "customers": count,
            "income": income,
            "spending": spending,
            "recommendation": recommendation
        })

    summary_table = summary.to_html(
        classes="table table-striped table-hover",
        index=False,
        border=0
    )

    return render_template(
        "insights.html",
        total_customers=len(df),
        total_segments=df["Segment"].nunique(),
        summary_table=summary_table,
        insights_list=insights_list
    )
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)