# Project 2: Customer Segmentation

## Goal
Group customers using Annual Income and Spending Score with K-Means clustering.

## Dataset
The PDF recommends the Mall Customer dataset. Put the CSV at `data/Mall_Customers.csv`. Common expected columns are `Annual Income (k$)` and `Spending Score (1-100)`. If yours differs, change `INCOME_COLUMN` and `SPENDING_COLUMN` in `main.py`.

## Run
```bash
python main.py
```

## What to explain
- Scaling: places income and spending score on comparable scales before distance-based clustering.
- K-Means: assigns each customer to the nearest cluster center, then updates centers iteratively.
- Elbow method: compares within-cluster sum of squares across values of K. The elbow is a heuristic, not a guaranteed correct answer.
- Silhouette score: higher values generally indicate better-separated clusters; it is not defined for K=1 or when every row is its own cluster.

The cluster labels are groups for exploration, not fixed customer identities or proof of customer behaviour.
