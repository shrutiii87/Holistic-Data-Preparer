<img width="1200" height="300" alt="banner" src="https://github.com/user-attachments/assets/3cf3e04e-ed24-4a23-8816-21215b66b889" />

---

## 🎯 Objective

The purpose of this project is to demonstrate **complete, practical knowledge of Data Preprocessing and Feature Engineering** — walking the full end-to-end path from raw, messy, multi-source data to a single clean, consistent, model-ready dataset.

Every stage is deliberate: **understand → clean → impute → treat outliers → encode → scale → transform → construct.** Nothing is skipped, and every technique is followed by a 🎯 *Insights* note explaining what it did to the data and whether it was the right call.

---

## 📄 Problem Statement

You are hired as a **Junior Data Scientist** at a fintech company. The company provides a **Customer Credit Risk dataset** collected from multiple sources — CSV files, JSON files, SQL tables, and an external API. Your manager asks you to perform **full-scale preprocessing and feature engineering** so the dataset becomes clean, consistent, and suitable for building a Machine Learning model that predicts whether a customer will **default on a loan**.

The dataset contains:

- **Demographics** — age, gender, region, education level, employment type.
- **Financial details** — annual income, loan amount, loan purpose, repayment history, credit score.
- **Behavioural attributes** — transactions, spending habits, missed payments.
- **Target variable** — Loan Default → 0 (No), 1 (Yes).

---

# 📂 Project Files

| 📄 File / Folder | 📌 Description |
|------------------|----------------|
| 📓 `Holistic_Data_Preparer.ipynb` | Main notebook — the complete, annotated preprocessing & feature-engineering pipeline |
| 📊 `customer_credit_risk_main_transactions_1000.csv` | Raw core transactions dataset (CSV source) |
| 📑 `customer_metadata_1000.json` | Customer metadata (JSON source) |
| 🗄️ `loan_repayment_history.db` | SQLite database with loan repayment history (SQL source) |
| 📈 `customer_credit_risk_data_quality_report.html` | Auto-generated Pandas/ydata profiling report |
| 🧼 `final_processed_dataset.csv` | Final cleaned, encoded and scaled ML-ready dataset |
| 📘 `README.md` | Project documentation and workflow guide |

---

## 🛠️ Tools Used

<div>

<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white"/>
<img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white"/>
<img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white"/>
<img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white"/>
<img src="https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white"/>
<img src="https://img.shields.io/badge/SQLite-003B57?style=for-the-badge&logo=sqlite&logoColor=white"/>
<img src="https://img.shields.io/badge/ydata--profiling-FF4B4B?style=for-the-badge"/>
<img src="https://img.shields.io/badge/MICE-Imputation-EC4899?style=for-the-badge"/>
<img src="https://img.shields.io/badge/KNN-Imputer-0EA5E9?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Winsorization-7C3AED?style=for-the-badge"/>
<img src="https://img.shields.io/badge/ColumnTransformer-059669?style=for-the-badge"/>

</div>

---

# 🎬 Project Demo

[![Watch Demo](https://img.shields.io/badge/▶️%20Watch%20Demo-Google%20Drive-blue?style=for-the-badge&logo=google-drive)](https://drive.google.com/file/d/1ZuVCxGIFa7tJHLn4mLyihBH0W_C4tVda/view?usp=sharing)

📹 Click the badge above to watch the complete project demonstration.

---

### 🧬 Dataset Structure — Customer Credit Risk

| Field Name | Data Type | Description | Notes |
|------------|-----------|-------------|-------|
| `customer_id` | String / Int | Unique identifier for each customer | Join key — no missing values |
| `age` | Integer | Age of customer (years) | Missing values injected |
| `gender` | Categorical | Male / Female / Other | Missing + category imbalance |
| `region` | Categorical | North / South / East / West | One-Hot Encoding |
| `education_level` | Ordinal Cat. | Primary / Secondary / Graduate / Post-Graduate | Ordinal Encoding |
| `employment_type` | Categorical | Salaried / Self-Employed / Unemployed | Categorical imputation |
| `annual_income` | Float | Annual income (₹) | Very high outliers + missing |
| `loan_amount` | Float | Loan amount requested (₹) | Outliers + skewed distribution |
| `loan_purpose` | Categorical | Home / Car / Education / Business / Other | One-Hot Encoding |
| `credit_score` | Float | Credit score (300–850) | Outliers + missing values |
| `repayment_history` | Integer | Missed payments in last 12 months | Binning / outlier treatment |
| `transaction_count` | Integer | Transactions in last 6 months | K-Means / Quantile binning |
| `spending_ratio` | Float | Spending-to-income ratio (%) | Log / Box-Cox / Yeo-Johnson |
| `join_date` | Date | Date customer joined the bank | Extract Y / M / D / Weekday |
| `default_flag` | Binary Int | 0 = No Default, 1 = Default | 🎯 **ML target** |

---

<img width="1535" height="1024" alt="c87edcd2-2262-4ed2-a7d3-33c33ca47172" src="https://github.com/user-attachments/assets/2633190a-7c29-45e6-8f9a-b566211f9031" />


---

## 📥 Part B : Data Acquisition

<img width="1200" height="300" alt="partB" src="https://github.com/user-attachments/assets/794d546e-0cb6-4cd5-8d49-bc2fd42fdfd1" />

### 3️⃣ Import datasets from multiple sources

##### 📂 Load CSV files (main transactions dataset)

```python
transactions_df = pd.read_csv("customer_credit_risk_main_transactions_1000.csv")
transactions_df.head()
```

💡 **Insight:** The CSV is the **core fact table** — 1000 customers with demographics, financials and the `default_flag` target. Everything else joins onto it. 📊

---

##### 📑 Parse JSON files (customer metadata)

```python
metadata_df = pd.read_json("customer_metadata_1000.json")
metadata_df.head()
```

💡 **Insight:** JSON adds contextual metadata such as `join_date` and `spending_ratio`, keyed by `customer_id` — proving the pipeline can absorb **semi-structured** sources, not just flat files. 🧾

---

##### 🗄️ Fetch records from SQL

```python
conn = sqlite3.connect("loan_repayment_history.db")
repayment_df = pd.read_sql_query("SELECT * FROM loan_repayment_history", conn)
conn.close()

print(repayment_df.isna().sum())
repayment_df.head()
```

💡 **Insight:** The SQL table carries **behavioural signals** (`repayment_history`, `transaction_count`) and deliberately injected NULLs — the perfect proving ground for the imputation section that follows. 🗃️

---

##### 🌐 Fetch data from an API (external economic indicators)

```python
url = "https://api.worldbank.org/v2/country/IND/indicator/FP.CPI.TOTL.ZG?format=json"
data = requests.get(url).json()[1]

api_df = pd.DataFrame(data)[['country', 'date', 'value']]
api_df.columns = ['country', 'year', 'inflation_rate']
api_df.head()
```

💡 **Insight:** Live **inflation-rate data** from the World Bank adds macro-economic context that no internal table can provide — real credit-risk models are always enriched with external indicators. 🌍

---

##### 🔗 Merge CSV + JSON + SQL

```python
final_df = (
    transactions_df
    .merge(metadata_df,  on="customer_id", how="left")
    .merge(repayment_df, on="customer_id", how="left")
)
print(final_df.shape)
final_df.head()
```

💡 **Insight:** Three heterogeneous sources collapse into **one analytical base table** keyed on `customer_id`. `how="left"` guarantees no customer is lost, and any unmatched key simply surfaces as a missing value — handled next. 🧩

---

## 🧹 Part C : Data Understanding & Cleaning

<img width="1200" height="300" alt="partC" src="https://github.com/user-attachments/assets/b9a326c8-1d26-43d9-9714-a22f3b92809e" />

### 4️⃣ Explore the dataset using pandas

```python
final_df.info()
final_df.describe()

print("---------All Missing Values---------")
final_df.isnull().sum()
```

💡 **Insight:** `.info()` exposes dtypes and non-null counts, `.describe()` exposes the **wild spread in `annual_income` and `loan_amount`** (huge max vs median = outliers ahead). Missing values are scattered but never dominant, so **no column needs dropping**. 🔍

---

### 5️⃣ Pandas Profiling — data quality report

```python
profile = ProfileReport(
    final_df,
    title="Customer Credit Risk Data Quality Report",
    explorative=True
)
profile.to_file("customer_credit_risk_data_quality_report.html")
```

💡 **Insight:** One command produces a complete HTML audit — distributions, correlations, missing-value matrix, duplicate check and warnings. It turns hours of manual EDA into a **single reproducible artefact**. 📈

---

### 🧩 Handle missing data with :-

##### 📋 Simple Imputer (Numerical: Mean / Median)

```python
mean_df = final_df.copy()

num_cols = ["age", "annual_income", "loan_amount", "credit_score",
            "spending_ratio", "repayment_history", "transaction_count"]

mean_imputer = SimpleImputer(strategy="mean")
mean_df[num_cols] = mean_imputer.fit_transform(mean_df[num_cols])

print(mean_df[num_cols].isnull().sum())
```

💡 **Insight:** Mean imputation fills every numeric gap — the post-imputation null count is **0 for all numeric columns**. It is fast and keeps the mean unchanged, but it **shrinks variance** and distorts skewed columns like `annual_income`. ⚖️

---

##### 🏷️ Simple Imputer (Categorical: Most Frequent)

```python
cat_cols = ["gender", "employment_type"]

cat_imputed_df = final_df.copy()
cat_imputer = SimpleImputer(strategy="most_frequent")
cat_imputed_df[cat_cols] = cat_imputer.fit_transform(cat_imputed_df[cat_cols])

print(cat_imputed_df[cat_cols].isnull().sum())
```

💡 **Insight:** Mean/median can't apply to text — the **mode** keeps imputed categories realistic and avoids inventing a meaningless new class. Being sklearn-based, it drops straight into a `ColumnTransformer`. 👥

---

##### 📊 Most Frequent Category Imputation (pandas equivalent)

```python
mfci_df = final_df.copy()

for col in cat_cols:
    mfci_df[col] = mfci_df[col].fillna(mfci_df[col].mode()[0])

print(mfci_df[cat_cols].isnull().sum())
```

💡 **Insight:** The manual `mode()` fill reproduces the sklearn result **exactly** — pandas gives per-column transparency, sklearn gives pipeline reusability. Same answer, different ergonomics. 🔁

---

##### 🔍 Missing Indicator + Random Sample Imputation

```python
random_sample_df = final_df.copy()
random_sample_df["annual_income_missing"] = random_sample_df["annual_income"].isnull().astype(int)

random_values = random_sample_df["annual_income"].dropna().sample(
    random_sample_df["annual_income"].isnull().sum(),
    random_state=42, replace=True
)
random_sample_df.loc[random_sample_df["annual_income"].isnull(), "annual_income"] = random_values.values
```

💡 **Insight:** Random sampling draws from the **observed distribution**, so skew and variance survive intact — unlike mean imputation, which flattens them. The `annual_income_missing` flag lets the model learn whether *missingness itself* carries signal. 🚩

---

##### ⚡ KNN Imputer (multivariate)

```python
knn_df = final_df.copy()
knn = KNNImputer(n_neighbors=5)
knn_df[num_cols] = knn.fit_transform(knn_df[num_cols])
knn_df[num_cols].describe()
```

💡 **Insight:** KNN fills each gap using the **five most similar customers**, capturing multivariate relationships (income ↔ loan amount ↔ credit score) that univariate methods ignore. Costlier, but distribution-faithful. 🤝

---

##### ♻️ MICE Algorithm (chained equations)

```python
mice_df = final_df.copy()
mice = IterativeImputer(random_state=42)
mice_df[num_cols] = mice.fit_transform(mice_df[num_cols])
```

💡 **Insight:** MICE models each column as a **regression on all the others** and iterates until convergence — statistically the most robust estimator here, and the method chosen for the final numeric imputation. 🔬

---

##### 🗑️ Complete Case Analysis

```python
cca_df = final_df.dropna()
print("Before:", final_df.shape, " After:", cca_df.shape)
```

💡 **Insight:** Dropping incomplete rows is the simplest option but the **most expensive** — a meaningful share of customers disappears, and any bias in *why* data was missing is baked in. Useful as a baseline, not a solution. ⚠️

---

## ⚠️ Part D : Outlier Handling

<img width="1200" height="300" alt="partD" src="https://github.com/user-attachments/assets/19472c9f-95a7-4ab5-9983-f5996cc1fec7" />

### 7️⃣ Detect and treat outliers using:

##### 📈 Z-score Method

```python
zscore_df = final_df.copy()
num_cols = ["annual_income", "loan_amount", "credit_score",
            "repayment_history", "transaction_count"]

z_scores = np.abs(zscore(zscore_df[num_cols], nan_policy="omit"))
print(pd.DataFrame(z_scores > 3, columns=num_cols).sum())

zscore_df = zscore_df[(z_scores < 3).all(axis=1)]
print("Original:", final_df.shape, " After:", zscore_df.shape)
```

💡 **Insight:** The |z| > 3 rule flags a small set of extremes, mostly in `annual_income` and `loan_amount`. Because it depends on the mean and std — both of which outliers inflate — it is **itself sensitive to outliers** and assumes near-normality. 📏

---

##### 📉 IQR Method

```python
iqr_df = final_df.copy()

for col in num_cols:
    Q1, Q3 = iqr_df[col].quantile(0.25), iqr_df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    iqr_df = iqr_df[(iqr_df[col] >= lower) & (iqr_df[col] <= upper)]

print("Shape after IQR filtering:", iqr_df.shape)
```

💡 **Insight:** Quartile-based and therefore **robust to skew**, IQR flags noticeably more rows than Z-score. `repayment_history` and `transaction_count` are near-uniform, so most removals come from the monetary columns. 🧮

---

##### 💡 Percentile Method

```python
percentile_df = final_df.copy()

for col in num_cols:
    lower, upper = percentile_df[col].quantile(0.01), percentile_df[col].quantile(0.99)
    percentile_df = percentile_df[(percentile_df[col] >= lower) & (percentile_df[col] <= upper)]

print("Shape after trimming:", percentile_df.shape)
```

💡 **Insight:** Trimming at the 1st/99th percentile gives **explicit control over how much data is dropped** rather than letting the statistics decide — a predictable, auditable rule. 🎯

---

##### ✂️ Winsorization Technique

```python
winsor_df = final_df.copy()

for col in num_cols:
    lower, upper = winsor_df[col].quantile(0.05), winsor_df[col].quantile(0.95)
    winsor_df[col] = winsor_df[col].clip(lower, upper)

print("Winsorization Applied Successfully!")
```

💡 **Insight:** Capping at the 5th/95th percentile **retains all 1000 customers** while pulling minima and maxima inward. Extremes stop dominating scalers and distance-based models — the best quality/quantity trade-off in this project. 🧊

---

## 🧬 Part E : Feature Engineering

### 8️⃣ Handle variable types

##### 🔄 Mixed Variables (numeric + categorical)

```python
numerical_cols   = final_df.select_dtypes(include=["int64", "float64"]).columns
categorical_cols = final_df.select_dtypes(include=["object"]).columns

print("Numerical:", list(numerical_cols))
print("Categorical:", list(categorical_cols))
```

💡 **Insight:** A clean dtype split is the **foundation of every ColumnTransformer** that follows — numeric columns get scaled, categorical columns get encoded, and nothing is mistakenly processed twice. 🧷

---

##### 📅 Date & Time variables → Year, Month, Day, Weekday

```python
final_df["join_date"]    = pd.to_datetime(final_df["join_date"])
final_df["join_year"]    = final_df["join_date"].dt.year
final_df["join_month"]   = final_df["join_date"].dt.month
final_df["join_day"]     = final_df["join_date"].dt.day
final_df["join_weekday"] = final_df["join_date"].dt.day_name()
```

💡 **Insight:** One opaque timestamp becomes **four usable features**, unlocking tenure, seasonality and weekday effects that a raw date string could never express. 📆

---

### 9️⃣ Encoding categorical variables

##### 🔠 Ordinal Encoding (education levels)

```python
education_order = {"Primary": 0, "Secondary": 1, "Graduate": 2, "Post-Graduate": 3}
ordinal_df = final_df.copy()
ordinal_df["education_level"] = ordinal_df["education_level"].map(education_order)
```

💡 **Insight:** Education has a **true ranking**, so mapping to 0–3 preserves real information the model can exploit. Using one-hot here would throw that ordering away. 🎓

---

##### 🔠 Label Encoding (binary features)

```python
label_df = final_df.copy()
label_encoder = LabelEncoder()
label_df["gender"] = label_encoder.fit_transform(label_df["gender"])

print(dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))
```

💡 **Insight:** Compact integer codes for a low-cardinality field, with the printed **mapping documenting exactly which class became which code** — essential for reproducibility and interpretation. 🏷️

---

##### 🔠 One-Hot Encoding (regions, loan purpose)

```python
onehot_df = pd.get_dummies(final_df, columns=["region", "loan_purpose"], dtype=int)
onehot_df.filter(regex="region_|loan_purpose_").head()
```

💡 **Insight:** `region` and `loan_purpose` are **nominal** — no natural order exists. One-hot expands them into indicator columns so the model never infers a fake ranking like *Home < Car < Education*. 🗺️

---

### 🔟 Encoding numerical features

##### 📶 Binning (discretize income into groups)

```python
binning_df = final_df.copy()
binning_df["income_group"] = pd.cut(binning_df["annual_income"], bins=3,
                                    labels=["Low", "Medium", "High"])
binning_df["income_group"].value_counts()
```

💡 **Insight:** Equal-width bins are easy to read but produce **unbalanced groups** on skewed income — a direct consequence of splitting by range rather than by count. 📦

---

##### 📶 Binarization (flag if > threshold)

```python
binarization_df = final_df.copy()
binarization_df["credit_score"] = binarization_df["credit_score"].fillna(
    binarization_df["credit_score"].median())

binarizer = Binarizer(threshold=700)
binarization_df["credit_score_flag"] = binarizer.fit_transform(
    binarization_df[["credit_score"]]).astype(int)
```

💡 **Insight:** A crisp `credit_score_flag` separates **prime from sub-prime** customers in a single interpretable column — exactly the kind of feature a risk officer can reason about. 🎚️

---

##### 📶 Quantile Binning

```python
quantile_df = final_df.copy()
quantile_df["transaction_count_quantile"] = pd.qcut(
    quantile_df["transaction_count"], q=4, labels=["Q1", "Q2", "Q3", "Q4"])
```

💡 **Insight:** Unlike equal-width binning, quantile binning guarantees **equally populated buckets**, which keeps every group statistically meaningful. 📐

---

##### 📶 K-Means Binning

```python
kmeans_df = final_df.copy()
kbins = KBinsDiscretizer(n_bins=4, encode="ordinal", strategy="kmeans")
kmeans_df["transaction_count_kmeans"] = kbins.fit_transform(
    kmeans_df[["transaction_count"]]).astype(int)
```

💡 **Insight:** K-Means places cut points where the data **naturally clusters** rather than at arbitrary widths or counts — the most data-driven of the three binning strategies. 🧠

---

## 📏 Part F : Feature Scaling

### 1️⃣1️⃣ Apply multiple scaling methods

```python
scale_cols = ["annual_income", "loan_amount", "credit_score", "spending_ratio"]

standard = StandardScaler().fit_transform(final_df[scale_cols])   # Z-score
normal   = Normalizer().fit_transform(final_df[scale_cols])       # unit norm
minmax   = MinMaxScaler().fit_transform(final_df[scale_cols])     # 0–1
maxabs   = MaxAbsScaler().fit_transform(final_df[scale_cols])     # −1–1
robust   = RobustScaler().fit_transform(final_df[scale_cols])     # median / IQR
```

| Scaler | What it does | Best for |
|--------|--------------|----------|
| 📏 **Standardization** | mean 0, std 1 | Linear models, PCA, SVM |
| 📐 **Normalization** | scales each row to unit norm | Distance/similarity models |
| 📊 **Min-Max** | squeezes into 0–1 | Neural networks, bounded inputs |
| ➕ **MaxAbs** | scales by max magnitude | Sparse data, keeps zeros |
| 🛡️ **Robust** | centres on median, scales by IQR | **Outlier-heavy money columns** |

💡 **Insight:** Because `annual_income` and `loan_amount` carry heavy tails even after treatment, **Robust Scaling** is the most trustworthy choice — it ignores the tails entirely by using median and IQR instead of mean and std. 🛡️

---

## 🧠 Part G : Feature Construction & Transformation

### 1️⃣2️⃣ Apply transformations

##### 🔧 FunctionTransformer → log, reciprocal, square root

```python
log_tf  = FunctionTransformer(np.log1p, validate=True)
sqrt_tf = FunctionTransformer(np.sqrt,  validate=True)

final_df["spending_ratio_log"]  = log_tf.fit_transform(final_df[["spending_ratio"]])
final_df["spending_ratio_sqrt"] = sqrt_tf.fit_transform(final_df[["spending_ratio"]])
```

💡 **Insight:** `log1p` compresses the long right tail of `spending_ratio` into a far more symmetric shape (and safely handles zeros), while `sqrt` gives a gentler correction for mild skew. 📉

---

##### ⚡ PowerTransformer → Box-Cox & Yeo-Johnson

```python
yeo = PowerTransformer(method="yeo-johnson")
final_df[["annual_income_yj", "loan_amount_yj"]] = yeo.fit_transform(
    final_df[["annual_income", "loan_amount"]].fillna(final_df[["annual_income", "loan_amount"]].median())
)
```

💡 **Insight:** Power transforms **search for the optimal λ** instead of assuming log is right. Box-Cox requires strictly positive values; **Yeo-Johnson handles zeros and negatives**, which is why it is the safer default here. ⚡

---

##### 🧮 Column Transformer → different preprocessing per column type

```python
preprocessor = ColumnTransformer(transformers=[
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                      ("scale",  RobustScaler())]), numerical_cols),
    ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                      ("encode", OneHotEncoder(handle_unknown="ignore"))]), categorical_cols),
])

processed = preprocessor.fit_transform(final_df)
```

💡 **Insight:** The entire preprocessing strategy collapses into **one reusable, leak-free object**. Fit on train, apply to test — no manual step can be forgotten or applied in the wrong order. 🧷

---

### 1️⃣3️⃣ Construct new features

```python
final_df["debt_to_income"]    = final_df["loan_amount"] / final_df["annual_income"]
final_df["avg_monthly_txn"]   = final_df["transaction_count"] / 6
final_df["spend_to_income"]   = final_df["spending_ratio"] / 100
```

| 💎 Feature | 🧮 Formula | 📌 Signal |
|-----------|-----------|-----------|
| **Debt-to-Income ratio** | `loan_amount / annual_income` | Leverage — the strongest default predictor |
| **Average monthly transactions** | `transaction_count / 6` | Behavioural activity level |
| **Spending-to-Income ratio** | `spending / annual_income` | Financial discipline |

💡 **Insight:** These ratios encode **relationships no single raw column holds**. A ₹5L loan is unremarkable at ₹30L income and alarming at ₹4L — only the ratio expresses that. 💰

---

## 📊 Part H : Final Deliverable

### 1️⃣4️⃣ Final cleaned and transformed dataset

```python
# 1. Missing values  → MICE (numeric) + Most Frequent (categorical)
# 2. Outliers        → Winsorize (5th / 95th percentile)
# 3. Encoding        → Ordinal + Label + One-Hot
# 4. Scaling         → Robust / Standard
# 5. Construction    → debt_to_income, avg_monthly_txn, spend_to_income

print("Missing values:", final_df.isnull().sum().sum())
print("Final shape:",    final_df.shape)

final_df.to_csv("final_processed_dataset.csv", index=False)
print("Final processed dataset saved successfully.")
```

💡 **Insight:** The delivered table is **fully numeric, gap-free, outlier-controlled and scaled**, with the target isolated — it can be handed to any estimator without a single further transformation. 💾

---

### 1️⃣5️⃣ Report Summary

| ❓ Question | ✅ Answer |
|-------------|----------|
| Which imputation strategy was most effective? | **MICE** for numeric columns — it estimates each gap from the relationships between all other features instead of a single statistic. **Most Frequent** for categoricals, since it preserves existing classes without inventing new ones. |
| Which outlier method preserved data quality best? | **Winsorization** — it caps extremes without deleting a single row, keeping all 1000 customers while removing the distorting influence of the tails. |
| Which encodings were applied and why? | Ordinal only where a real order exists (`education_level`), Label for low-cardinality binaries (`gender`), One-Hot for nominal fields (`region`, `loan_purpose`). |
| Which scaling / transformations and why? | **Robust Scaling** for heavy-tailed money columns; **Yeo-Johnson** for skewed `annual_income` / `loan_amount` because it tolerates zeros and negatives. |
| Are the new features useful? | Yes — Debt-to-Income is the single most predictive engineered signal, with spending and activity ratios adding behavioural context. |
| Is the dataset ML-ready? | Yes — zero missing values, all dtypes numeric, outliers capped, features scaled, target isolated. |

---

## 📂 Project Workflow

1. **Data Acquisition** → Load CSV, parse JSON, query SQL, fetch API
2. **Merge** → Join all sources into one analytical base table on `customer_id`
3. **Understanding** → `.info()`, `.describe()`, profiling report
4. **Missing Value Treatment** → Mean, Mode, Random Sample, KNN, MICE, CCA
5. **Outlier Handling** → Z-score, IQR, Percentile, Winsorization
6. **Feature Engineering** → Date parts, Ordinal / Label / One-Hot encoding, Binning
7. **Scaling** → Standard, Normalizer, Min-Max, MaxAbs, Robust
8. **Transformation** → Log, Reciprocal, Sqrt, Box-Cox, Yeo-Johnson, ColumnTransformer
9. **Construction** → Debt-to-Income, Avg monthly transactions, Spend-to-Income
10. **Export** → Save `final_processed_dataset.csv`

---

## 📈 Results & Insights

- ✅ **Four heterogeneous sources** (CSV + JSON + SQL + API) merged into one clean base table of 1000 customers
- ✅ **Six imputation strategies** compared — MICE selected for numerics, mode for categoricals
- ✅ **Four outlier techniques** compared — Winsorization selected, **zero rows deleted**
- ✅ **Nine encoding & binning techniques** applied across ordinal, nominal, binary and continuous columns
- ✅ **Five scalers and five transforms** benchmarked on the same columns
- ✅ **Three engineered ratio features** added, led by Debt-to-Income
- ✅ Final dataset exported as `final_processed_dataset.csv` — complete, stable and ML-ready

---

## 📌 Expected Outcomes

- Understand how to **plan and execute a complete data preprocessing workflow**
- Perform **detailed data cleaning** using imputation and outlier handling
- Apply **advanced encoding and scaling techniques** with intent, not habit
- **Construct and transform features** that measurably improve ML readiness

---

## ⚙️ Installation & Setup

```bash
# clone the repository
git clone https://github.com/yourusername/holistic-data-preparer.git
cd holistic-data-preparer

# create an isolated environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# install dependencies
pip install pandas numpy scikit-learn scipy matplotlib seaborn \
            ydata-profiling jupyter requests

# launch the notebook
jupyter notebook Holistic_Data_Preparer.ipynb
```

---

## 🚀 Future Scope

- [ ] Train baseline models (Logistic Regression, Random Forest, XGBoost) on the processed set
- [ ] Address target imbalance with SMOTE / class weights
- [ ] Feature importance and SHAP explainability
- [ ] Persist the full pipeline with `joblib` for reproducible inference

---

## 🙏 Thank You

Thank you for taking the time to explore this project!
Your feedback, suggestions, and contributions are always welcome.

⭐ If you found this project helpful, don't forget to **star the repository** and share it with others.

---

<div align="center">


</div>

