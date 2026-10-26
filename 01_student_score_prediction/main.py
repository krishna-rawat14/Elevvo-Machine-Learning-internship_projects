from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = Path("data/StudentPerformanceFactors.csv")

# Common column-name variants found in student-performance datasets.
STUDY_HOURS_CANDIDATES = ["Hours_Studied", "StudyHours", "Study_Hours", "study_hours", "hours_studied"]
TARGET_CANDIDATES = ["Exam_Score", "Final_Score", "Score", "exam_score", "final_score"]

def find_column(columns, candidates):
    """Return the first matching column, ignoring case and surrounding spaces."""
    normalized = {str(c).strip().lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]
    return None

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Download the recommended Student Performance Factors CSV and put it at that path."
        )

    df = pd.read_csv(DATA_PATH)
    print("Dataset shape:", df.shape)
    print("Columns:", list(df.columns))
    print(df.head())

    study_col = find_column(df.columns, STUDY_HOURS_CANDIDATES)
    target_col = find_column(df.columns, TARGET_CANDIDATES)
    if study_col is None or target_col is None:
        raise ValueError(
            "Could not identify study-hours and exam-score columns. "
            "Edit STUDY_HOURS_CANDIDATES / TARGET_CANDIDATES to match your CSV."
        )

    # Keep only the selected columns, coerce non-numeric values, and remove missing rows.
    data = df[[study_col, target_col]].copy()
    data[study_col] = pd.to_numeric(data[study_col], errors="coerce")
    data[target_col] = pd.to_numeric(data[target_col], errors="coerce")
    data = data.dropna()
    if len(data) < 10:
        raise ValueError("Too few usable rows after cleaning; check the dataset and column names.")

    # A scatter plot helps us visually inspect the relationship.
    plt.figure(figsize=(7, 5))
    plt.scatter(data[study_col], data[target_col], alpha=0.6)
    plt.xlabel(study_col)
    plt.ylabel(target_col)
    plt.title("Study Hours vs Exam Score")
    plt.tight_layout()
    plt.savefig("study_hours_vs_score.png", dpi=150)
    plt.close()

    X = data[[study_col]]  # Features must be a 2D table.
    y = data[target_col]   # Target is the value we want to predict.

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)  # Learn the line from training examples.
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    print(f"\nFeature: {study_col} | Target: {target_col}")
    print(f"Intercept: {model.intercept_:.3f}")
    print(f"Coefficient: {model.coef_[0]:.3f}")
    print(f"MAE:  {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R²:   {r2:.3f}")

    # Plot predictions against actual values. A perfect model would lie on y=x.
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, predictions, alpha=0.7)
    low = min(float(y_test.min()), float(predictions.min()))
    high = max(float(y_test.max()), float(predictions.max()))
    plt.plot([low, high], [low, high], linestyle="--")
    plt.xlabel("Actual score")
    plt.ylabel("Predicted score")
    plt.title("Actual vs Predicted Scores")
    plt.tight_layout()
    plt.savefig("actual_vs_predicted.png", dpi=150)
    plt.close()

    # Example prediction; replace 5 with a study-hour value appropriate to the dataset.
    example_hours = 5
    example = pd.DataFrame({study_col: [example_hours]})
    print(f"\nExample: {example_hours} study hours -> predicted score {model.predict(example)[0]:.2f}")
    print("\nSaved plots: study_hours_vs_score.png, actual_vs_predicted.png")

if __name__ == "__main__":
    main()
