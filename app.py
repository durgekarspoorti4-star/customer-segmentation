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
app.secret_key = "customer-segmentation-secret-key"

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
    if "user" not in session:
        return redirect(url_for("login"))
    file_path = os.path.join(DATASET_DIR, "customers.csv")
    if not os.path.exists(file_path):
        return redirect(url_for("upload"))
    df = pd.read_csv(file_path)
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df["Annual Income (k$)"] = pd.to_numeric(df["Annual Income (k$)"], errors="coerce")
    df["Spending Score (1-100)"] = pd.to_numeric(df["Spending Score (1-100)"], errors="coerce")
    total_customers = len(df)
    total_columns = len(df.columns)
    average_age = round(df["Age"].mean(), 2)
    average_income = round(df["Annual Income (k$)"].mean(), 2)
    average_spending = round(df["Spending Score (1-100)"].mean(), 2)
    missing_values = int(df.isnull().sum().sum())
    table = df.head(20).to_html(classes="table table-striped table-hover", index=False)
    return render_template("analysis.html", total_customers=total_customers, total_columns=total_columns, average_age=average_age, average_income=average_income, average_spending=average_spending, missing_values=missing_values, table=table)

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
    if request.method == "POST":
        try:
            income = float(request.form["income"])
            spending = float(request.form["spending"])
        except:
            return "Please enter valid numeric values."
        file_path = os.path.join(DATASET_DIR, "clustered_customers.csv")
        if not os.path.exists(file_path):
            file_path = os.path.join(DATASET_DIR, "customers.csv")
        if not os.path.exists(file_path):
            return redirect(url_for("upload"))
        df = pd.read_csv(file_path)
        df["Annual Income (k$)"] = pd.to_numeric(df["Annual Income (k$)"], errors="coerce")
        df["Spending Score (1-100)"] = pd.to_numeric(df["Spending Score (1-100)"], errors="coerce")
        df = df.dropna(subset=["Annual Income (k$)", "Spending Score (1-100)"])
        features = df[["Annual Income (k$)", "Spending Score (1-100)"]]
        model = KMeans(n_clusters=3, random_state=42, n_init=10)
        model.fit(features)
        new_customer = [[income, spending]]
        cluster_result = int(model.predict(new_customer)[0])
        df["Cluster"] = model.labels_
        cluster_average = df.groupby("Cluster")["Spending Score (1-100)"].mean().sort_values()
        clusters = list(cluster_average.index)
        if cluster_result == clusters[0]:
            prediction_result = "Low Spending"
        elif cluster_result == clusters[1]:
            prediction_result = "Medium Spending"
        else:
            prediction_result = "High Spending"
    return render_template("prediction.html", prediction=prediction_result, cluster=cluster_result)

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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)