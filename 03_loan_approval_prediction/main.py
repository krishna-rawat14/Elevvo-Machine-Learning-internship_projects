from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    precision_score, recall_score, f1_score, accuracy_score
)

DATA_PATH = Path("data/loan_approval.csv")
TARGET_CANDIDATES = ["Loan_Status", "loan_status", "LoanApproved", "approved", "loan_approved"]

def find_target(columns):
    normalized = {str(c).strip().lower(): c for c in columns}
    for candidate in TARGET_CANDIDATES:
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]
    return None

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Download a Loan Approval Prediction CSV and put it at that path."
        )

    df = pd.read_csv(DATA_PATH)
    print("Dataset shape:", df.shape)
    print("Columns:", list(df.columns))
    print(df.head())

    target_col = find_target(df.columns)
    if target_col is None:
        raise ValueError(
            "Could not find the target column. Edit TARGET_CANDIDATES to match your dataset."
        )

    df = df.dropna(subset=[target_col]).copy()
    y_raw = df[target_col].astype(str).str.strip()

    # Remove likely ID columns because they usually do not describe applicant eligibility.
    id_like = [c for c in df.columns if c.strip().lower() in {"loan_id", "id", "applicant_id"}]
    X = df.drop(columns=[target_col] + id_like)

    # Drop columns that are entirely empty.
    X = X.dropna(axis=1, how="all")
    if X.empty:
        raise ValueError("No usable feature columns found.")

    # Encode target labels into 0/1 (or integer classes) for scikit-learn.
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    if len(label_encoder.classes_) < 2:
        raise ValueError("Target must contain at least two classes.")
    print("Target classes:", list(label_encoder.classes_))

    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("numeric", numeric_pipeline, numeric_cols),
        ("categorical", categorical_pipeline, categorical_cols)
    ])

    # Stratification preserves approximate class proportions in train and test sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced"))
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Macro averages give each class equal weight, useful when classes are imbalanced.
    print("\nAccuracy:", round(accuracy_score(y_test, predictions), 3))
    print("Macro precision:", round(precision_score(y_test, predictions, average="macro", zero_division=0), 3))
    print("Macro recall:", round(recall_score(y_test, predictions, average="macro", zero_division=0), 3))
    print("Macro F1:", round(f1_score(y_test, predictions, average="macro", zero_division=0), 3))
    print("\nClassification report:")
    print(classification_report(
        y_test, predictions,
        labels=list(range(len(label_encoder.classes_))),
        target_names=[str(c) for c in label_encoder.classes_],
        zero_division=0
    ))

    cm = confusion_matrix(y_test, predictions, labels=list(range(len(label_encoder.classes_))))
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
    display.plot(xticks_rotation=45)
    plt.title("Loan Approval — Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.close()

    print("Saved: confusion_matrix.png")
    print("\nNote: this model is for learning only, not real lending decisions.")

if __name__ == "__main__":
    main()
