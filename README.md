# Machine Learning Internship Projects

A collection of three machine learning projects developed as part of a two-week machine learning internship.

## Projects

### 1. Student Score Prediction
**Type:** Supervised Learning — Regression

Predict student exam scores based on study hours using Linear Regression.

- Data cleaning and exploratory visualisation
- Training and testing data split
- Model evaluation using MAE, RMSE and R²
- Actual vs. predicted score visualisation

**Technologies:** Python, Pandas, Matplotlib, Scikit-learn

### 2. Customer Segmentation
**Type:** Unsupervised Learning — Clustering

Group mall customers according to annual income and spending score using K-Means.

- Feature scaling and data exploration
- Elbow method for cluster selection
- Customer cluster visualisation
- Cluster averages and silhouette score

**Technologies:** Python, Pandas, Matplotlib, Scikit-learn

### 3. Loan Approval Prediction
**Type:** Supervised Learning — Classification

Build a classification model to predict loan approval outcomes from applicant information.

- Missing-value handling
- Categorical feature encoding
- Logistic Regression classification
- Evaluation using precision, recall, F1-score and a confusion matrix

**Technologies:** Python, Pandas, Matplotlib, Scikit-learn

## Repository Structure

```text
machine-learning-internship-projects/
├── 01_student_score_prediction/
│   ├── data/
│   ├── main.py
│   └── README.md
├── 02_customer_segmentation/
│   ├── data/
│   ├── main.py
│   └── README.md
├── 03_loan_approval_prediction/
│   ├── data/
│   ├── main.py
│   └── README.md
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

Python 3.10 or newer is recommended.

1. Clone this repository.
2. Open the repository folder in a terminal.
3. Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

4. Download each required dataset and place its CSV in the corresponding project's `data/` folder, using the filename specified in that project's README.
5. Navigate into the project folder and run:

```bash
python main.py
```

## Learning Outcomes

- Data preprocessing and exploratory data analysis
- Regression, clustering and classification
- Model evaluation and interpretation
- Data visualisation with Matplotlib
- Organising Python projects for reproducibility

## Notes

This repository is intended for educational purposes. Model performance depends on the dataset and experimental setup. The loan approval model is not intended for real-world lending decisions.

**Author:** Add your name here.
