# 🎯 Customer Credit Risk — Data Preprocessing & Feature Engineering

> A complete end-to-end data preprocessing pipeline that ingests data from **CSV, JSON, SQL, and API** sources, cleans it, profiles it, handles outliers, engineers new features, and scales/transforms the data so it is ready for machine learning models.

---

## 🎯 Objective

Conduct **Data Preprocessing and Feature Engineering** on a real-world customer credit-risk dataset.  
The aim is to **understand, clean, transform, and analyze** the data before it can be used for machine learning.

This project emphasizes:

- 📊 Data profiling & quality assessment
- 🔄 Handling multiple data formats
- 🧹 Data cleaning & missing-value imputation
- ⚠️ Outlier detection & treatment
- 🔧 Feature engineering, encoding & scaling
- 🤖 ML problem framing: **Credit Risk / Default Prediction**

---

## 📄 Problem Statement

The dataset contains **customer credit-risk and transactional behavior** collected from multiple sources:

- 📂 CSV — `customer_credit_risk_main_transactions_1000.csv`
- 📑 JSON — `customer_profiles.json`
- 🗄️ SQL — SQLite database (`retail_customers.db`)
- 🌐 API — Random user data from a public API

The goal is to **frame a machine-learning problem** (predict customer default / credit risk) and perform **comprehensive preprocessing & profiling** so the dataset becomes ML-ready.

---

## 📂 Project Files

| 📄 File | 📌 Description |
|---------|----------------|
| 📓 **data_profiler.ipynb** | Main Jupyter Notebook with the full pipeline |
| 📊 **customer_credit_risk_main_transactions_1000.csv** | Main transaction dataset (1000 rows × 11 columns) |
| 📄 **customer_profiles.json** | Customer profile records in JSON |
| 🗄️ **retail_customers.db** | SQLite3 database with customer records |
| 🌐 **randomuser.me API** | External dummy API for extra demographic data |
| 📑 **README.md** | Project documentation (this file) |

### Key columns in the dataset

| Column | Description |
|--------|-------------|
| `customer_id` | Unique customer identifier |
| `age` | Customer age |
| `gender` | Male / Female |
| `region` | Geographic region (North, South, East, West) |
| `education_level` | Education category (Primary, Secondary, Graduate, Postgraduate) |
| `employment_type` | Salaried / Self-Employed / etc. |
| `annual_income` | Annual income in local currency |
| `loan_amount` | Loan amount requested |
| `loan_purpose` | Purpose of the loan (Home, Car, Business, Education, Other) |
| `credit_score` | Credit bureau score |
| `default_flag` | Target variable: 1 = defaulted, 0 = non-defaulted |

---

## 🛠️ Tools Used

<div>

<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white"/>
<img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white"/>
<img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white"/>
<img src="https://img.shields.io/badge/JSON-000000?style=for-the-badge&logo=json&logoColor=white"/>
<img src="https://img.shields.io/badge/SQLite3-003B57?style=for-the-badge&logo=sqlite&logoColor=white"/>
<img src="https://img.shields.io/badge/Requests-FF6F00?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
<img src="https://img.shields.io/badge/Matplotlib-2C2D72?style=for-the-badge&logo=plotly&logoColor=white"/>
<img src="https://img.shields.io/badge/Seaborn-4C9A2A?style=for-the-badge&logo=seaborn&logoColor=white"/>
<img src="https://img.shields.io/badge/ydata_profiling-FF4088?style=for-the-badge&logo=python&logoColor=white"/>

</div>

---

## 🎬 Project Demo

<!-- Replace the link below with your actual demo URL -->
[![Watch Demo](https://img.shields.io/badge/▶️%20Watch%20Demo-Google%20Drive-blue?style=for-the-badge&logo=google-drive)](https://drive.google.com/your-demo-link)

📹 Click the badge above to watch the complete project demonstration.

---

### ♻️ Workflow

```text
Raw Data (CSV / JSON / SQL / API)
        ↓
   Data Acquisition
        ↓
   Data Understanding & Cleaning
        ↓
   Pandas Profiling
        ↓
   Missing Value Imputation
        ↓
   Outlier Detection & Treatment
        ↓
   Feature Engineering (Encoding / Binning)
        ↓
   Feature Scaling
        ↓
   Feature Construction & Transformation
        ↓
   ML-Ready Dataset
```

---

## 📥 Part B : Data Acquisition

### 5️⃣ Import datasets from multiple sources

- 📂 Load CSV files
- 📑 Parse JSON files
- 🗄️ Fetch records from SQL
- 🌐 Fetch data from a dummy API

```python
import pandas as pd
import json
import sqlite3
import requests

# 1. CSV
transactions_df = pd.read_csv("customer_credit_risk_main_transactions_1000.csv")
print(transactions_df.shape)

# 2. JSON
with open("customer_profiles.json") as file_obj:
    profile_records = json.load(file_obj)
df_json = pd.json_normalize(profile_records)

# 3. SQL
conn = sqlite3.connect("retail_customers.db")
sql_df = pd.read_sql_query("SELECT * FROM customers", conn)

# 4. API
resp = requests.get("https://randomuser.me/api/?results=20").json()
df_api = pd.json_normalize(resp["results"])
```

### 🗂️ Merged the data

```python
combined_df = pd.concat([transactions_df, df_json, sql_df], ignore_index=True)
combined_df = combined_df.merge(
    df_api[["gender", "email", "location.city"]],
    left_index=True, right_index=True, how="left"
)
combined_df.info()
```

---

## 🧹 Part C : Data Understanding & Cleaning

### 6️⃣ Perform initial exploration

- 📝 Use `.head()`, `.info()`, `.describe()` to explore
- 🔍 Identify missing values and duplicates

```python
print(combined_df.head())
print(combined_df.info())
print(combined_df.describe(include="all"))
print(combined_df.isnull().sum())
print(combined_df.duplicated().sum())
```

### 7️⃣ Apply data cleaning

- 🔍 Handle missing data
- 🧠 Correct inconsistent data types
- 🗑️ Drop irrelevant columns

```python
combined_df = combined_df.dropna(subset=["age", "annual_income"], how="any")
combined_df["loan_amount"] = combined_df["loan_amount"].fillna(0).astype(float)
combined_df["age"] = combined_df["age"].astype(int)
combined_df["annual_income"] = combined_df["annual_income"].astype(float)
combined_df = combined_df.drop_duplicates()
combined_df = combined_df.drop(columns=["Unused_Col"], errors="ignore")
print("After cleaning:", combined_df.shape)
```

### 📊 Pandas Profiling

Generate an interactive HTML report summarizing missing values, descriptive statistics, correlations, and data-quality warnings.

```python
from ydata_profiling import ProfileReport

report = ProfileReport(combined_df, title="Credit Risk Profiling Report", explorative=True)
report.to_file("credit_risk_profiling_report.html")
print("Report saved.")
```

---

## 🧩 Missing Data Handling

### 8️⃣ Simple Imputer (Numerical: Mean / Median)

```python
from sklearn.impute import SimpleImputer

num_imputer = SimpleImputer(strategy="mean")
combined_df[["annual_income", "loan_amount"]] = num_imputer.fit_transform(
    combined_df[["annual_income", "loan_amount"]]
)
```

##### 🎯 Insights

- Mean imputation fills every numeric gap; the post-imputation null count is **0**.
- Best for features with a roughly symmetric distribution and low missingness.

### 9️⃣ Simple Imputer (Categorical: Most Frequent)

```python
cat_imputer = SimpleImputer(strategy="most_frequent")
combined_df[["gender", "employment_type"]] = cat_imputer.fit_transform(
    combined_df[["gender", "employment_type"]]
)
```

##### 🎯 Insights

- `SimpleImputer(strategy='most_frequent')` clears all missing `gender` and `employment_type` values.
- Quick and preserves the most common category without distorting the distribution.

### 🔟 Missing Indicator + Random Sample Imputation

```python
import numpy as np

# Missing indicator
combined_df["annual_income_missing"] = combined_df["annual_income"].isna().astype(int)

# Random sample imputation
missing_idx = combined_df["annual_income"].isna()
observed = combined_df.loc[~missing_idx, "annual_income"]
combined_df.loc[missing_idx, "annual_income"] = np.random.choice(observed, missing_idx.sum())
```

##### 🎯 Insights

- Random sample imputation fills `annual_income` using values drawn from the observed distribution.
- Adding a missing indicator preserves information about which rows were originally missing.

---

## ⚠️ Part D : Outlier Handling

### 1️⃣1️⃣ Detect and treat outliers using:

### 📈 Z-score Method

```python
from scipy import stats

z_scores = np.abs(stats.zscore(combined_df.select_dtypes(include=[np.number])))
combined_df = combined_df[(z_scores < 3).all(axis=1)]
```

##### 🎯 Insights

- The Z-score rule (`|z| > 3`) flags extreme records, mostly in `annual_income` and `loan_amount`.
- Simple and effective for roughly normal distributions.

### 📉 IQR Method

```python
Q1 = combined_df.quantile(0.25)
Q3 = combined_df.quantile(0.75)
IQR = Q3 - Q1
combined_df = combined_df[~((combined_df < (Q1 - 1.5 * IQR)) | (combined_df > (Q3 + 1.5 * IQR))).any(axis=1)]
```

##### 🎯 Insights

- The IQR rule uses quartiles, making it robust to skew.
- Flags more outliers than Z-score in skewed financial features.

### 📊 Percentile Method

```python
lower = combined_df.quantile(0.01)
upper = combined_df.quantile(0.99)
combined_df = combined_df.clip(lower, upper, axis=1)
```

##### 🎯 Insights

- Percentile trimming at the 1st/99th bounds gives explicit control over how much data is removed.

### ✂️ Winsorization Technique

```python
from scipy.stats import mstats

combined_df["annual_income"] = mstats.winsorize(combined_df["annual_income"], limits=[0.05, 0.05])
```

##### 🎯 Insights

- Winsorization at the 5th/95th percentiles **caps** extreme values instead of deleting rows.
- Keeps the full dataset while reducing the influence of outliers.

---

## 🧬 Part E : Feature Engineering

### 1️⃣2️⃣ Handle variable types

#### 🔄 Mixed Variables (numeric + categorical)

```python
numeric_cols = combined_df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = combined_df.select_dtypes(include=["object"]).columns.tolist()
```

##### 🎯 Insights

- Separating numeric and categorical columns is the first step before applying different preprocessing pipelines.

#### 📅 Date & Time variables → extract Year, Month, Day, Weekday

```python
combined_df["join_date"] = pd.to_datetime(combined_df["join_date"], errors="coerce")
combined_df["join_year"] = combined_df["join_date"].dt.year
combined_df["join_month"] = combined_df["join_date"].dt.month
combined_df["join_day"] = combined_df["join_date"].dt.day
combined_df["join_weekday"] = combined_df["join_date"].dt.weekday
```

##### 🎯 Insights

- Converting `join_date` to datetime unlocks `join_year`, `join_month`, `join_day`, and `join_weekday` as useful categorical/numeric features.

### 1️⃣3️⃣ Encoding categorical variables

#### 🔠 Ordinal Encoding (education levels)

```python
from sklearn.preprocessing import OrdinalEncoder

education_order = [["Primary", "Secondary", "Graduate", "Postgraduate"]]
ord_encoder = OrdinalEncoder(categories=education_order)
combined_df["education_encoded"] = ord_encoder.fit_transform(combined_df[["education_level"]])
```

##### 🎯 Insights

- Ordinal encoding maps education to **0–3 in a meaningful order** (Primary < Secondary < Graduate < Postgraduate).

#### 🔠 Label Encoding (binary features)

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
combined_df["gender_encoded"] = le.fit_transform(combined_df["gender"])
```

##### 🎯 Insights

- Label encoding converts `gender` into compact integers, keeping the column small.

#### 🔠 One-Hot Encoding (regions, loan purpose)

```python
combined_df = pd.get_dummies(combined_df, columns=["region", "loan_purpose"], drop_first=True)
```

##### 🎯 Insights

- One-hot encoding expands `region` and `loan_purpose` into indicator columns suitable for linear models.

### 1️⃣4️⃣ Encoding numerical features

#### 📶 Binning (discretize income into groups)

```python
combined_df["income_bin"] = pd.cut(
    combined_df["annual_income"], bins=3, labels=["Low", "Medium", "High"]
)
```

##### 🎯 Insights

- Equal-width binning turns continuous income into Low/Medium/High groups, making patterns easier to interpret.

#### 📶 Binarization (flag if > threshold)

```python
from sklearn.preprocessing import Binarizer

binarizer = Binarizer(threshold=700)
combined_df["credit_score_flag"] = binarizer.fit_transform(combined_df[["credit_score"]])
```

##### 🎯 Insights

- Binarization at a threshold of 700 creates a crisp `credit_score_flag` separating high and low credit scores.

#### 📶 Quantile Binning

```python
combined_df["income_quantile"] = pd.qcut(combined_df["annual_income"], q=4, labels=["Q1", "Q2", "Q3", "Q4"])
```

##### 🎯 Insights

- Quantile binning produces four roughly equal-sized groups, preventing empty bins.

#### 📶 K-Means Binning

```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=42, n_init="auto")
combined_df["income_kmeans"] = kmeans.fit_predict(combined_df[["annual_income"]])
```

##### 🎯 Insights

- K-Means binning lets the **data itself decide** the cut points instead of imposing fixed boundaries.

---

## 📏 Part F : Feature Scaling

### 1️⃣5️⃣ Apply multiple scaling methods

#### 📐 Standardization (Z-score scaling)

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
combined_df[["annual_income_scaled", "loan_amount_scaled"]] = scaler.fit_transform(
    combined_df[["annual_income", "loan_amount"]]
)
```

##### 🎯 Insights

- Standardization re-centres `annual_income` and `loan_amount` to mean 0 and unit variance.

#### 📐 Normalization

```python
from sklearn.preprocessing import Normalizer

normalizer = Normalizer()
combined_df[["income_norm", "loan_norm"]] = normalizer.fit_transform(
    combined_df[["annual_income", "loan_amount"]]
)
```

##### 🎯 Insights

- `Normalizer` works **row-wise**, scaling each customer's vector to unit norm.

#### 📊 Min-Max Scaling

```python
from sklearn.preprocessing import MinMaxScaler

minmax = MinMaxScaler()
combined_df[["income_minmax", "loan_minmax"]] = minmax.fit_transform(
    combined_df[["annual_income", "loan_amount"]]
)
```

##### 🎯 Insights

- Min-Max scaling compresses every numeric feature into a clean **[0, 1]** range.

#### ➕ MaxAbs Scaling

```python
from sklearn.preprocessing import MaxAbsScaler

maxabs = MaxAbsScaler()
combined_df[["income_maxabs", "loan_maxabs"]] = maxabs.fit_transform(
    combined_df[["annual_income", "loan_amount"]]
)
```

##### 🎯 Insights

- MaxAbs scaling maps data into `[-1, 1]` without shifting the center, which is useful for sparse data.

#### 🛡️ Robust Scaling

```python
from sklearn.preprocessing import RobustScaler

robust = RobustScaler()
combined_df[["income_robust", "loan_robust"]] = robust.fit_transform(
    combined_df[["annual_income", "loan_amount"]]
)
```

##### 🎯 Insights

- Robust scaling centers on the **median** and scales by the **IQR**, so extreme income values have minimal impact.

---

## 🧠 Part G : Feature Construction & Transformation

### 1️⃣6️⃣ Apply transformations

#### 🔧 FunctionTransformer → log, reciprocal, square root

```python
from sklearn.preprocessing import FunctionTransformer

log_transform = FunctionTransformer(np.log1p, validate=True)
combined_df["income_log"] = log_transform.fit_transform(combined_df[["annual_income"]])
```

##### 🎯 Insights

- `FunctionTransformer` applies log, reciprocal, and square-root transforms to reduce skewness.

#### ⚡ PowerTransformer → Box-Cox and Yeo-Johnson

```python
from sklearn.preprocessing import PowerTransformer

pt = PowerTransformer(method="yeo-johnson")
combined_df[["income_power", "loan_power"]] = pt.fit_transform(
    combined_df[["annual_income", "loan_amount"]]
)
```

##### 🎯 Insights

- `PowerTransformer` estimates the optimal lambda automatically, stabilizing variance.

#### 🧮 ColumnTransformer → different preprocessing for different columns

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), ["age", "annual_income", "loan_amount"]),
        ("cat", OneHotEncoder(drop="first"), ["gender", "region", "loan_purpose"]),
    ]
)
X_processed = preprocessor.fit_transform(combined_df)
```

##### 🎯 Insights

- `ColumnTransformer` applies StandardScaler to numeric columns and OneHotEncoder to categoricals in one step.

### 1️⃣7️⃣ Construct new features

#### 💰 Debt-to-Income ratio

```python
combined_df["debt_to_income_ratio"] = combined_df["loan_amount"] / combined_df["annual_income"]
```

##### 🎯 Insights

- `debt_to_income_ratio` compresses two columns into one meaningful risk indicator.

#### 💰 Average monthly transactions

```python
combined_df["avg_monthly_transaction"] = combined_df["transaction_count"] / 12
```

##### 🎯 Insights

- `avg_monthly_transaction` converts a raw total into a monthly rate, making it comparable across customers.

#### 💰 Spending-to-Income ratio

```python
combined_df["spending_to_income_ratio"] = combined_df["total_spending"] / combined_df["annual_income"]
```

##### 🎯 Insights

- `spending_to_income_ratio` links behavioural spending back to earning capacity.

---

## 📊 Part H : Final Deliverable

### 1️⃣8️⃣ Provide a final cleaned and transformed dataset

```python
combined_df.to_csv("final_credit_risk_ml_ready.csv", index=False)
print("Final ML-ready dataset saved. Shape:", combined_df.shape)
```

### 1️⃣9️⃣ Write a report summarizing:

- 🧩 Missing-value strategies used and their effectiveness
- ⚠️ Outlier handling results
- 🔠 Encoding choices and why each was selected
- 📏 Scaling decisions for model compatibility
- 🛠️ New engineered features and their business meaning
- 📈 Final dataset shape and readiness for ML

---

## 📂 Project Workflow

1. **Data Acquisition** → Import data from CSV, JSON, SQL, and API
2. **Data Cleaning** → Handle missing values, duplicates, and inconsistent data types
3. **Data Profiling** → Generate profiling reports for better understanding
4. **Missing Value Imputation** → Mean, most-frequent, and random-sample strategies
5. **Outlier Treatment** → Z-score, IQR, percentile, and Winsorization
6. **Feature Engineering** → Encoding, binning, and datetime feature extraction
7. **Feature Scaling** → Standardization, Min-Max, Robust, and more
8. **Feature Transformation** → Log, Box-Cox, Yeo-Johnson, and ColumnTransformer
9. **Feature Construction** → Debt-to-income, monthly transaction rate, spending ratios
10. **Problem Framing** → Define ML task: **Credit Risk / Default Prediction**

---

## 📈 Results & Insights

- ✅ Cleaned dataset with consistent data types and no duplicates
- ✅ Multiple missing-value strategies applied and evaluated
- ✅ Outliers detected and treated using Z-score, IQR, percentile, and Winsorization
- ✅ Categorical features encoded with ordinal, label, and one-hot encoders
- ✅ Numerical features binned and binarized for interpretability
- ✅ Feature scaling applied across multiple methods for model compatibility
- ✅ New engineered features capture risk signals (debt-to-income, spending-to-income)
- ✅ Clear ML problem framing: **Predict Customer Default / Credit Risk**

---

## 📌 Expected Outcomes

- A **structured, ML-ready dataset** saved as `final_credit_risk_ml_ready.csv`
- **Data-quality profiling report** generated via `ydata_profiling`
- **Documentation** of all cleaning, imputation, outlier, and engineering steps
- A reusable **preprocessing pipeline** that can be applied to new credit-risk data

---

## ⚙️ Installation & Setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/customer-credit-risk-preprocessing.git
cd customer-credit-risk-preprocessing
pip install -r requirements.txt
```

### Required packages

```text
pandas
numpy
scikit-learn
scipy
matplotlib
seaborn
ydata-profiling
requests
```

### Run the notebook

```bash
jupyter notebook data_profiler.ipynb
```

---

## 🙏 Thank You

Thank you for taking the time to explore this project!  
Your feedback, suggestions, and contributions are always welcome.

⭐ If you found this project helpful, don’t forget to **star the repository** and share it with others.
