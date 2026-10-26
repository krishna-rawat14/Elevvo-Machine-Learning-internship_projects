# Project 3: Loan Approval Prediction

## Goal
Predict loan approval from applicant information and evaluate precision, recall, and F1-score.

## Dataset
The PDF recommends a Loan Approval Prediction dataset from Kaggle. Put its CSV at `data/loan_approval.csv`.

Different versions use different column names and target labels. This script attempts common target names (`Loan_Status`, `loan_status`, `LoanApproved`, `approved`) and treats the target as a classification label. If your file differs, update `TARGET_CANDIDATES` and check the label meanings. It excludes obvious identifier columns if present.

## Run
```bash
python main.py
```

## What to explain
- Missing values: median for numeric columns and most-frequent value for categorical columns.
- One-hot encoding: converts categories into numeric indicator columns.
- Logistic regression: estimates class probabilities and is a useful baseline classifier.
- Precision: among predicted approvals, the fraction that were actually approvals.
- Recall: among actual approvals, the fraction correctly identified.
- F1-score: harmonic mean of precision and recall.
- Confusion matrix: shows true vs predicted class counts.

This is an educational model only. Do not use it to make real lending decisions; financial approval requires fairness, compliance, and much stronger validation.
