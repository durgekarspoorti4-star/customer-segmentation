from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    send_file,
    session
)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import os
import json


# =========================================================
# FLASK APP
# =========================================================

app = Flask(__name__)
app.secret_key = "customer-segmentation-secret-key"


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
def home():
    return render_template("home.html")


# =========================================================
# DASHBOARD / INDEX
# =========================================================

@app.route("/index")
def index():

    if "user" not in session:
        return redirect(url_for("login"))

    file_path = "dataset/clustered_customers.csv"

    total_customers = 0
    total_clusters = 0
    average_income = 0
    average_spending = 0
    graph_exists = False

    if os.path.exists(file_path):

        df = pd.read_csv(file_path)

        total_customers = len(df)

        # Number of clusters
        if "Cluster" in df.columns:
            total_clusters = df["Cluster"].nunique()

        # Average income
        if "Annual Income (k$)" in df.columns:

            df["Annual Income (k$)"] = pd.to_numeric(
                df["Annual Income (k$)"],
                errors="coerce"
            )

            average_income = round(
                df["Annual Income (k$)"].mean(),
                2
            )

        # Average spending
        if "Spending Score (1-100)" in df.columns:

            df["Spending Score (1-100)"] = pd.to_numeric(
                df["Spending Score (1-100)"],
                errors="coerce"
            )

            average_spending = round(
                df["Spending Score (1-100)"].mean(),
                2
            )

        graph_exists = os.path.exists(
            "static/graphs/results.png"
        )

    return render_template(
        "index.html",
        total_customers=total_customers,
        total_clusters=total_clusters,
        average_income=average_income,
        average_spending=average_spending,
        graph_exists=graph_exists
    )


# =========================================================
# UPLOAD DATASET
# =========================================================

@app.route("/upload", methods=["GET", "POST"])
def upload():

    if "user" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":

        file = request.files.get("file")

        if file and file.filename.endswith(".csv"):

            os.makedirs("dataset", exist_ok=True)

            file.save("dataset/customers.csv")

            return redirect(url_for("index"))

    return render_template("upload.html")


# =========================================================
# DATA ANALYSIS
# =========================================================

@app.route("/analysis")
def analysis():

    if "user" not in session:
        return redirect(url_for("login"))

    file_path = "dataset/customers.csv"

    if not os.path.exists(file_path):
        return redirect(url_for("upload"))

    df = pd.read_csv(file_path)

    # Convert numeric columns
    df["Age"] = pd.to_numeric(
        df["Age"],
        errors="coerce"
    )

    df["Annual Income (k$)"] = pd.to_numeric(
        df["Annual Income (k$)"],
        errors="coerce"
    )

    df["Spending Score (1-100)"] = pd.to_numeric(
        df["Spending Score (1-100)"],
        errors="coerce"
    )

    # Statistics
    total_customers = len(df)
    total_columns = len(df.columns)

    average_age = round(
        df["Age"].mean(),
        2
    )

    average_income = round(
        df["Annual Income (k$)"].mean(),
        2
    )

    average_spending = round(
        df["Spending Score (1-100)"].mean(),
        2
    )

    missing_values = int(
        df.isnull().sum().sum()
    )

    # Table
    table = df.head(20).to_html(
        classes="table table-striped table-hover",
        index=False
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


# =========================================================
# K-MEANS
# =========================================================

@app.route("/kmeans", methods=["GET", "POST"])
def kmeans_page():

    if "user" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":

        try:
            k = int(request.form["k"])
        except (ValueError, TypeError):
            return "Invalid number of clusters"

        if k < 2:
            return "Number of clusters must be at least 2"

        # Load dataset
        file_path = "dataset/customers.csv"

        if not os.path.exists(file_path):
            return redirect(url_for("upload"))

        df = pd.read_csv(file_path)

        # Convert required columns
        df["Annual Income (k$)"] = pd.to_numeric(
            df["Annual Income (k$)"],
            errors="coerce"
        )

        df["Spending Score (1-100)"] = pd.to_numeric(
            df["Spending Score (1-100)"],
            errors="coerce"
        )

        # Remove missing values
        df = df.dropna(
            subset=[
                "Annual Income (k$)",
                "Spending Score (1-100)"
            ]
        )

        if len(df) < k:
            return "Number of clusters is greater than available customers."

        # Features
        features = df[
            [
                "Annual Income (k$)",
                "Spending Score (1-100)"
            ]
        ]

        # K-Means
        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        df["Cluster"] = model.fit_predict(features)

        # Save result
        os.makedirs("dataset", exist_ok=True)

        df.to_csv(
            "dataset/clustered_customers.csv",
            index=False
        )

        return render_template(
            "kmeans.html",
            result=True,
            k=k
        )

    return render_template(
        "kmeans.html",
        result=False
    )


# =========================================================
# RESULTS
# =========================================================

@app.route("/results")
def results():

    if "user" not in session:
        return redirect(url_for("login"))

    file_path = "dataset/clustered_customers.csv"

    if not os.path.exists(file_path):
        return redirect(url_for("kmeans_page"))

    df = pd.read_csv(file_path)

    # Convert columns
    df["Annual Income (k$)"] = pd.to_numeric(
        df["Annual Income (k$)"],
        errors="coerce"
    )

    df["Spending Score (1-100)"] = pd.to_numeric(
        df["Spending Score (1-100)"],
        errors="coerce"
    )

    df["Cluster"] = pd.to_numeric(
        df["Cluster"],
        errors="coerce"
    )

    df = df.dropna(
        subset=[
            "Annual Income (k$)",
            "Spending Score (1-100)",
            "Cluster"
        ]
    )

    df["Cluster"] = df["Cluster"].astype(int)

    # Average spending by cluster
    cluster_average = (
        df.groupby("Cluster")["Spending Score (1-100)"]
        .mean()
        .sort_values()
    )

    clusters = list(cluster_average.index)

    # Segment names
    segment_names = {}

    if len(clusters) == 3:

        segment_names[clusters[0]] = "Low Spending"
        segment_names[clusters[1]] = "Medium Spending"
        segment_names[clusters[2]] = "High Spending"

    else:

        for i, cluster in enumerate(clusters):
            segment_names[cluster] = f"Group {i + 1}"

    df["Segment"] = df["Cluster"].map(segment_names)

    # Summary
    summary = (
        df.groupby(["Cluster", "Segment"])
        .agg(
            Customer_Count=("Cluster", "size"),
            Average_Income=("Annual Income (k$)", "mean"),
            Average_Spending=("Spending Score (1-100)", "mean")
        )
        .reset_index()
    )

    summary["Average_Income"] = summary[
        "Average_Income"
    ].round(2)

    summary["Average_Spending"] = summary[
        "Average_Spending"
    ].round(2)

    # Create graph folder
    os.makedirs(
        "static/graphs",
        exist_ok=True
    )

    # Create graph
    plt.figure(figsize=(9, 6))

    sns.scatterplot(
        data=df,
        x="Annual Income (k$)",
        y="Spending Score (1-100)",
        hue="Segment",
        s=100
    )

    plt.title("Customer Segmentation using K-Means")
    plt.xlabel("Annual Income (k$)")
    plt.ylabel("Spending Score (1-100)")
    plt.tight_layout()

    plt.savefig(
        "static/graphs/results.png"
    )

    plt.close()

    # Save updated dataset
    df.to_csv(
        "dataset/clustered_customers.csv",
        index=False
    )

    # Convert tables
    summary_table = summary.to_html(
        classes="table table-bordered table-striped",
        index=False
    )

    customer_table = df.to_html(
        classes="table table-bordered table-striped",
        index=False
    )

    return render_template(
        "results.html",
        summary_table=summary_table,
        customer_table=customer_table
    )


# =========================================================
# SEGMENTS
# =========================================================

@app.route("/segments")
def segments():

    if "user" not in session:
        return redirect(url_for("login"))

    file_path = "dataset/clustered_customers.csv"

    if not os.path.exists(file_path):
        return redirect(url_for("kmeans_page"))

    df = pd.read_csv(file_path)

    # Create Segment if missing
    if "Segment" not in df.columns:

        cluster_average = (
            df.groupby("Cluster")["Spending Score (1-100)"]
            .mean()
            .sort_values()
        )

        clusters = list(cluster_average.index)

        segment_names = {}

        if len(clusters) == 3:

            segment_names[clusters[0]] = "Low Spending"
            segment_names[clusters[1]] = "Medium Spending"
            segment_names[clusters[2]] = "High Spending"

        else:

            for i, cluster in enumerate(clusters):
                segment_names[cluster] = f"Group {i + 1}"

        df["Segment"] = df["Cluster"].map(
            segment_names
        )

    # Count customers
    low_count = len(
        df[df["Segment"] == "Low Spending"]
    )

    medium_count = len(
        df[df["Segment"] == "Medium Spending"]
    )

    high_count = len(
        df[df["Segment"] == "High Spending"]
    )

    # Customer table
    customer_table = df.to_html(
        classes="table table-bordered table-striped",
        index=False
    )

    return render_template(
        "segments.html",
        low_count=low_count,
        medium_count=medium_count,
        high_count=high_count,
        customer_table=customer_table
    )


# =========================================================
# DOWNLOAD
# =========================================================

@app.route("/download")
def download():

    file_path = "dataset/clustered_customers.csv"

    if not os.path.exists(file_path):
        return redirect(url_for("kmeans_page"))

    return send_file(
        file_path,
        as_attachment=True,
        download_name="customer_segmentation_results.csv"
    )


# =========================================================
# ELBOW METHOD
# =========================================================

@app.route("/elbow")
def elbow():

    if "user" not in session:
        return redirect(url_for("login"))

    file_path = "dataset/customers.csv"

    if not os.path.exists(file_path):
        return redirect(url_for("upload"))

    df = pd.read_csv(file_path)

    # Convert values
    df["Annual Income (k$)"] = pd.to_numeric(
        df["Annual Income (k$)"],
        errors="coerce"
    )

    df["Spending Score (1-100)"] = pd.to_numeric(
        df["Spending Score (1-100)"],
        errors="coerce"
    )

    # Remove missing values
    df = df.dropna(
        subset=[
            "Annual Income (k$)",
            "Spending Score (1-100)"
        ]
    )

    features = df[
        [
            "Annual Income (k$)",
            "Spending Score (1-100)"
        ]
    ]

    # Calculate inertia
    inertia = []
    k_values = range(2, 11)

    for k in k_values:

        if k > len(features):
            break

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        model.fit(features)

        inertia.append(model.inertia_)

    # Create graph folder
    os.makedirs(
        "static/graphs",
        exist_ok=True
    )

    # Create graph
    plt.figure(figsize=(8, 5))

    plt.plot(
        list(k_values)[:len(inertia)],
        inertia,
        marker="o"
    )

    plt.title("Elbow Method for Optimal K")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Inertia")

    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        "static/graphs/elbow.png"
    )

    plt.close()

    return render_template("elbow.html")


# =========================================================
# ABOUT
# =========================================================

@app.route("/about")
def about():
    return render_template("about.html")


# =========================================================
# PREDICTION
# =========================================================

@app.route("/prediction", methods=["GET", "POST"])
def prediction():

    if "user" not in session:
        return redirect(url_for("login"))

    prediction_result = None
    cluster_result = None

    if request.method == "POST":

        try:
            income = float(
                request.form["income"]
            )

            spending = float(
                request.form["spending"]
            )

        except (ValueError, TypeError):
            return "Please enter valid numeric values."

        file_path = "dataset/clustered_customers.csv"

        if not os.path.exists(file_path):
            return redirect(url_for("kmeans_page"))

        df = pd.read_csv(file_path)

        # Convert values
        df["Annual Income (k$)"] = pd.to_numeric(
            df["Annual Income (k$)"],
            errors="coerce"
        )

        df["Spending Score (1-100)"] = pd.to_numeric(
            df["Spending Score (1-100)"],
            errors="coerce"
        )

        # Remove missing values
        df = df.dropna(
            subset=[
                "Annual Income (k$)",
                "Spending Score (1-100)"
            ]
        )

        features = df[
            [
                "Annual Income (k$)",
                "Spending Score (1-100)"
            ]
        ]

        # Use the same K-Means settings
        model = KMeans(
            n_clusters=3,
            random_state=42,
            n_init=10
        )

        model.fit(features)

        # Predict customer
        new_customer = [[
            income,
            spending
        ]]

        cluster_result = int(
            model.predict(new_customer)[0]
        )

        # Determine segment using cluster average
        df["Cluster"] = model.labels_

        cluster_average = (
            df.groupby("Cluster")[
                "Spending Score (1-100)"
            ]
            .mean()
            .sort_values()
        )

        clusters = list(
            cluster_average.index
        )

        if cluster_result == clusters[0]:

            prediction_result = "Low Spending"

        elif cluster_result == clusters[1]:

            prediction_result = "Medium Spending"

        else:

            prediction_result = "High Spending"

    return render_template(
        "prediction.html",
        prediction=prediction_result,
        cluster=cluster_result
    )


# =========================================================
# REGISTER
# =========================================================

@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        name = request.form["name"]
        email = request.form["email"]
        password = request.form["password"]

        users = []

        if os.path.exists("users.json"):

            with open(
                "users.json",
                "r"
            ) as file:

                try:
                    users = json.load(file)

                except json.JSONDecodeError:
                    users = []

        users.append({
            "name": name,
            "email": email,
            "password": password
        })

        with open(
            "users.json",
            "w"
        ) as file:

            json.dump(
                users,
                file,
                indent=4
            )

        return redirect(
            url_for("login")
        )

    return render_template(
        "register.html"
    )


# =========================================================
# LOGIN
# =========================================================

@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        email = request.form["email"]
        password = request.form["password"]

        if os.path.exists("users.json"):

            with open(
                "users.json",
                "r"
            ) as file:

                try:
                    users = json.load(file)

                except json.JSONDecodeError:
                    users = []

            for user in users:

                if (
                    user["email"] == email
                    and user["password"] == password
                ):

                    session["user"] = user["name"]

                    return redirect(
                        url_for("index")
                    )

        return "Invalid email or password"

    return render_template(
        "login.html"
    )


# =========================================================
# LOGOUT
# =========================================================

@app.route("/logout")
def logout():

    session.clear()

    return redirect(
        url_for("home")
    )


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":
    app.run(
        debug=True
    )