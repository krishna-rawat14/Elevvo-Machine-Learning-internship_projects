from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

DATA_PATH = Path("data/Mall_Customers.csv")
INCOME_COLUMN = "Annual Income (k$)"
SPENDING_COLUMN = "Spending Score (1-100)"

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Download the Mall Customers CSV and put it at that path."
        )

    df = pd.read_csv(DATA_PATH)
    print("Dataset shape:", df.shape)
    print("Columns:", list(df.columns))
    print(df.head())

    missing = [c for c in [INCOME_COLUMN, SPENDING_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Edit the column constants in main.py.")

    # Convert selected columns to numeric and remove rows with missing values.
    features = df[[INCOME_COLUMN, SPENDING_COLUMN]].apply(pd.to_numeric, errors="coerce")
    valid = features.dropna()
    if len(valid) < 10:
        raise ValueError("Too few usable rows after cleaning.")

    # Scale features because K-Means uses distances.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(valid)

    # Elbow curve: smaller inertia is expected as K grows, so look for a bend.
    ks = range(1, 11)
    inertias = []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)

    plt.figure(figsize=(7, 5))
    plt.plot(list(ks), inertias, marker="o")
    plt.xlabel("Number of clusters (K)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method")
    plt.tight_layout()
    plt.savefig("elbow_method.png", dpi=150)
    plt.close()

    # K=5 is a common demonstration choice for this dataset; inspect the elbow plot
    # and compare alternatives rather than treating 5 as universally optimal.
    chosen_k = 5
    model = KMeans(n_clusters=chosen_k, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)

    result = valid.copy()
    result["Cluster"] = labels
    result.to_csv("customers_with_clusters.csv", index=False)

    score = silhouette_score(X_scaled, labels) if len(set(labels)) > 1 else float("nan")
    print(f"\nChosen K: {chosen_k}")
    print(f"Silhouette score: {score:.3f}")
    print("\nAverage income and spending score by cluster:")
    print(result.groupby("Cluster")[[INCOME_COLUMN, SPENDING_COLUMN]].mean().round(2))

    # Plot original values so the axes remain interpretable to a reader.
    plt.figure(figsize=(8, 6))
    for cluster_id in sorted(result["Cluster"].unique()):
        group = result[result["Cluster"] == cluster_id]
        plt.scatter(group[INCOME_COLUMN], group[SPENDING_COLUMN], label=f"Cluster {cluster_id}", alpha=0.75)
    plt.xlabel(INCOME_COLUMN)
    plt.ylabel(SPENDING_COLUMN)
    plt.title("Customer Segments from K-Means")
    plt.legend()
    plt.tight_layout()
    plt.savefig("customer_clusters.png", dpi=150)
    plt.close()

    print("\nSaved: elbow_method.png, customer_clusters.png, customers_with_clusters.csv")

if __name__ == "__main__":
    main()
