<div align="center">

<img src="docs/banner.png" alt="Holistic Data Preparer — data preprocessing and feature engineering pipeline" width="100%" />

# Holistic Data Preparer

**An end-to-end Data Preprocessing & Feature Engineering pipeline on a Customer Credit Risk dataset.**

<sub>Final Project · Red & White Skill Education · Duration: 8 Days (1 hour per day)</sub>

<p>
<img alt="Python" src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
<img alt="pandas" src="https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" />
<img alt="NumPy" src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" />
<img alt="scikit-learn" src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" />
<img alt="Jupyter" src="https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white" />
</p>

</div>

---

## Table of Contents

1. [Objective](#objective)
2. [Problem Statement](#problem-statement)
3. [Dataset](#dataset)
4. [Project Structure](#project-structure)
5. [Getting Started](#getting-started)
6. [The Pipeline — Part by Part](#the-pipeline--part-by-part)
7. [Techniques Reference](#techniques-reference)
8. [Engineered Features](#engineered-features)
9. [Final Deliverable](#final-deliverable)
10. [Report Summary](#report-summary)
11. [Key Learnings](#key-learnings)
12. [Roadmap](#roadmap)

---

## Objective

The purpose of this project is to demonstrate **complete, practical knowledge of Data Preprocessing and Feature Engineering** — walking the full end-to-end path from raw, messy, multi-source data to a single clean, consistent, model-ready dataset.

Every stage is deliberate: understand → clean → impute → treat outliers → encode → scale → transform → construct. Nothing is skipped, and every decision is justified with insight cells inside the notebook.

> The outcome is a **fully processed dataset** ready to train a Machine Learning model that predicts loan default.

---

## Problem Statement

You are hired as a **Junior Data Scientist** at a fintech company. The company provides a **Customer Credit Risk dataset** collected from multiple sources — CSV files, JSON files, SQL tables, and an external API. Your manager asks you to perform **full-scale preprocessing and feature engineering** so the dataset becomes clean, consistent, and suitable for building a Machine Learning model that predicts whether a customer is likely to **default on a loan**.

The data spans four dimensions:

| Dimension | Contents |
| --- | --- |
| **Demographics** | age, gender, region, education level, employment type |
| **Financial** | annual income, loan amount, loan purpose, repayment history, credit score |
| **Behavioural** | transaction count, spending habits, missed payments |
| **Target** | `default_flag` → 0 = No Default, 1 = Default |

---

## Dataset

### Schema

| Field | Type | Description | Why it matters |
| --- | --- | --- | --- |
| `customer_id` | String / Int | Unique identifier per customer | No missing values — join key |
| `age` | Integer | Age of customer (years) | Missing values injected → imputation |
| `gender` | Categorical | Male / Female / Other | Missing + class imbalance |
| `region` | Categorical | North / South / East / West | One-Hot Encoding |
| `education_level` | Ordinal | Primary / Secondary / Graduate / Post-Graduate | Ordinal Encoding |
| `employment_type` | Categorical | Salaried / Self-Employed / Unemployed | Categorical imputation |
| `annual_income` | Float | Annual income (₹) | Extreme outliers + missing values |
| `loan_amount` | Float | Loan amount requested (₹) | Outliers + skewed distribution |
| `loan_purpose` | Categorical | Home / Car / Education / Business / Other | One-Hot Encoding |
| `credit_score` | Float | Credit score (300–850) | Outliers + missing values |
| `repayment_history` | Integer | Missed payments in last 12 months | Binning / outlier treatment |
| `transaction_count` | Integer | Transactions in last 6 months | K-Means binning |
| `spending_ratio` | Float | Spending-to-income ratio (%) | Log / Box-Cox / Yeo-Johnson |
| `join_date` | Date | Date customer joined the bank | Extract Y / M / D / Weekday |
| `default_flag` | Binary Int | 0 = No Default, 1 = Default | **ML target** |

### Sources merged

```text
CSV   →  customer_credit_risk_main_transactions_1000.csv   (core transactions)
JSON  →  customer metadata
SQL   →  loan repayment history
API   →  external economic indicators (dummy endpoint)
                    ↓
        one unified analytical base table
```

---

## Project Structure

```text
.
├── Holistic_Data_Preparer.ipynb   # the complete, annotated pipeline
├── docs/
│   └── banner.png                 # project banner
└── README.md
```

---

## Getting Started

```bash
# 1 — create an isolated environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2 — install the stack
pip install pandas numpy scikit-learn scipy matplotlib seaborn \
            ydata-profiling jupyter sqlalchemy requests

# 3 — launch
jupyter notebook Holistic_Data_Preparer.ipynb
```

Then run the cells top to bottom. Each section ends with a **🎯 Insights** cell explaining what the technique did to the data and whether it was the right call.

---

## The Pipeline — Part by Part

### Part A · Conceptual Foundation
Short notes on **what Data Analysis is**, **how to plan a Data Science project**, and **how to frame a Machine Learning problem** — plus an in-depth explanation of **Tensors** with NumPy examples.

### 📥 Part B · Data Acquisition
Load CSV, parse JSON, query SQL, and fetch a dummy API. Merge all four into one coherent frame keyed on `customer_id`.

### 🧹 Part C · Data Understanding & Cleaning
`.info()`, `.describe()`, and a full **Pandas Profiling** data-quality report. Missing data handled with:
- Simple Imputer — numerical (mean / median)
- Simple Imputer — categorical (most frequent)
- Most Frequent Category Imputation
- Missing Indicator + Random Sample Imputation
- KNN Imputer (multivariate)
- MICE (iterative imputation)
- Complete Case Analysis (dropping rows / columns)

### ⚠️ Part D · Outlier Handling
Detect and treat extremes on `annual_income`, `loan_amount`, and `credit_score` using the **Z-score**, **IQR**, **Percentile**, and **Winsorization** methods — comparing how many points each flags and what each does to the distribution.

### 🧬 Part E · Feature Engineering
- **Mixed variables** — `gender` (categorical), `age` (numeric), `spending_ratio` (numeric, skewed)
- **Date & time** — `join_date` → Year, Month, Day, Weekday
- **Categorical encoding** — Ordinal (`education_level`), Label (`gender`), One-Hot (`region`, `loan_purpose`)
- **Numerical encoding** — Binning, Binarization (`credit_score > 700`), Quantile Binning, K-Means Binning

### 📏 Part F · Feature Scaling
Standardization (Z-score), Normalization, Min-Max, MaxAbs, and Robust Scaling — applied and compared across all numeric columns.

### 🧠 Part G · Feature Construction & Transformation
- `FunctionTransformer` → log, reciprocal, square-root
- `PowerTransformer` → Box-Cox and Yeo-Johnson
- `ColumnTransformer` → different preprocessing for categorical vs numeric in a single pipeline

### 📊 Part H · Final Deliverable
A consolidated pipeline — missing values → winsorized outliers → encoding → scaling — exporting the final cleaned and transformed dataset.

---

## Techniques Reference

| Category | Techniques applied |
| --- | --- |
| **Imputation** | Mean/Median, Most Frequent, Missing Indicator + Random Sample, KNN, MICE, Complete Case |
| **Outliers** | Z-score, IQR, Percentile capping, Winsorization |
| **Encoding** | Ordinal, Label, One-Hot, Binning, Binarization, Quantile, K-Means |
| **Scaling** | Standard, Normalizer, Min-Max, MaxAbs, Robust |
| **Transforms** | Log, Reciprocal, Square-root, Box-Cox, Yeo-Johnson |
| **Orchestration** | `ColumnTransformer`, `Pipeline` |

---

## Engineered Features

| Feature | Formula | Signal |
| --- | --- | --- |
| **Debt-to-Income ratio** | `loan_amount / annual_income` | Leverage — the single strongest default predictor |
| **Average monthly transactions** | `transaction_count / 6` | Behavioural activity level |
| **Spending-to-Income ratio** | `spending / annual_income` | Financial discipline |
| **Date parts** | Year / Month / Day / Weekday from `join_date` | Tenure and seasonality |
| **High credit flag** | `credit_score > 700` | Fast, interpretable risk split |

---

## Final Deliverable

- ✅ A **final cleaned and transformed dataset**, fully numeric, no missing values, outliers tamed, features scaled.
- ✅ A written **report** covering: imputation strategies and their effectiveness, outlier-handling results, encoding choices per variable type, scaling and transformation rationale, newly engineered features and their usefulness, and the final dataset shape and ML readiness.

---

## Report Summary

| Question | Answer captured in the notebook |
| --- | --- |
| Which imputation worked best? | Compared distribution shift per strategy; multivariate (KNN / MICE) preserved relationships best on income & credit score |
| What happened to outliers? | Winsorization retained sample size while pulling extreme income tails into range |
| Why these encodings? | Ordinal only where a true order exists; One-Hot for nominal; Label for binary |
| Why these transforms? | Skewed monetary columns normalized via Yeo-Johnson (handles zeros/negatives safely) |
| Is it ML-ready? | Final shape reported; all dtypes numeric; target isolated |

---

## Key Learnings

- Plan and execute a **complete data preprocessing workflow** end to end.
- Perform **detailed data cleaning** through layered imputation and outlier handling.
- Apply **advanced encoding and scaling** techniques with intent, not habit.
- **Construct and transform features** that measurably improve ML readiness.

---

## Roadmap

- [ ] Train baseline models (Logistic Regression, Random Forest, XGBoost) on the processed set
- [ ] Address target imbalance with SMOTE / class weights
- [ ] Feature importance and SHAP explainability
- [ ] Persist the full pipeline with `joblib` for reproducible inference

---

<div align="center">

*"Quality is our Motto."* · Shaping *"skills"* for *"scaling"* higher…!!!

</div>

