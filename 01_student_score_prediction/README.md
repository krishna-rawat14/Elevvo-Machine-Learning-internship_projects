# Project 1: Student Score Prediction

## Goal
Predict exam scores using study hours (and optionally other numeric factors). This is a regression task.

## Dataset
The PDF recommends the Student Performance Factors dataset from Kaggle. Download its CSV and place it here as `data/StudentPerformanceFactors.csv`. Create the `data` folder first.

The code tries to identify a study-hours column and an exam-score column from common names. If your CSV uses different names, edit `STUDY_HOURS_CANDIDATES` or `TARGET_CANDIDATES` near the top of `main.py`.

## Run
From this folder:
```bash
python main.py
```

## What to explain
- Cleaning: remove rows missing the selected input or target.
- Train/test split: hold out data that the model did not learn from.
- Linear regression: fits a straight-line relationship between input and score.
- MAE: average absolute prediction error, in score units.
- RMSE: penalizes larger errors more strongly than MAE.
- R²: compares the model against predicting the mean target; it can be negative on poor predictions.

This is a learning project, not a tool for making high-stakes decisions about individual students.
