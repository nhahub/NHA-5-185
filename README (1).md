# Darra AI — Week 1: Data Collection, Initial Cleaning & Advanced EDA

> DEPI Project · Credit-risk early-warning system (Darra AI)

Week 1 focused on collecting the datasets, understanding their structure and limitations, doing **only minimal cleaning**, and running an in-depth **exploratory data analysis** to decide what Weeks 2+ need (split, target engineering, feature engineering, modeling).

---

## 🎯 Goal

Understand the data **before** modeling: its quality, its risks (imbalance, leakage, sentinel codes), and the financial relationships between income, debt, utilization, payment behavior and default — and turn that into concrete decisions for the next stage.

## 📦 Datasets

| | **Give Me Some Credit (GMSC)** | **UCI Default of Credit Card Clients (Taiwan)** |
|---|---|---|
| Source | Kaggle competition | UCI Machine Learning Repository |
| Rows × Columns | 150,000 × 12 | 30,000 × 25 |
| Granularity | One snapshot per borrower | 6 months of history per customer (Apr–Sep 2005) |
| Target | `SeriousDlqin2yrs` (90+ days delinquent within 2 years) | `default` (default next month) |
| Positive rate | **6.68%** (≈ 14 : 1 imbalance, severe) | **22.12%** (≈ 3.5 : 1 imbalance, moderate) |
| Purpose in Darra AI | Long-term risk + "what-if" scenarios (income, debt ratio, dependents) | Monthly monitoring + 30/90-day risk trend |
| Notebook | `Main_EDA.ipynb` | `Taiwan_EDA.ipynb` |

## 🧹 Initial Cleaning (obvious issues only)

Following the Week 1 rules, no advanced preprocessing or transformations were applied. Issues were **detected and documented**, not silently fixed:

- Fixed/inspected data types and column names (`Unnamed: 0` → `ID`, `PAY_0` → `PAY_1`, trimmed headers).
- Checked exact duplicates and duplicates ignoring the ID column.
- Flagged impossible values (e.g. `age = 0`) and sentinel codes (96/98).
- All imputation, capping, scaling and feature engineering is deferred to the **Week 2 unified sklearn pipeline** (fit on train only, to avoid leakage).

---

## 🔍 Key Findings

### Give Me Some Credit

| Area | Finding |
|---|---|
| **Imbalance** | Only 6.68% defaulted → accuracy is misleading; use AUC-PR / recall, class weights |
| **Missing values** | `MonthlyIncome` 19.82%, `NumberOfDependents` 2.62%; default rate differs between missing vs. present (5.61% vs 6.95% for income) → missingness carries signal, keep a missing-indicator |
| **Duplicates** | 0 duplicate IDs, but **609 rows** identical on every real feature → leakage risk across splits |
| **Impossible values** | 1 row with `age = 0` |
| **Sentinel codes** | **269 rows** have 96/98 in all three "days past due" columns — default rate **54.65% vs 6.68%** overall → treat as a flagged group, not as real counts |
| **Ratio errors** | `RevolvingUtilization` should be in [0,1] but reaches 50,708 (3,321 rows > 1); `DebtRatio` reaches 329,664 (35,137 rows > 1) → cap/winsorize |
| **Strongest signals** | Spearman vs. target: `NumberOfTimes90DaysLate` 0.34, `60-89DaysPastDue` 0.28, `30-59DaysPastDue` 0.26, `RevolvingUtilization` 0.24, `age` −0.12 |
| **Relationships** | Default rate rises monotonically with utilization and past-due counts, falls with age and income; the three past-due counters are highly inter-correlated |

### UCI Taiwan Credit Card

| Area | Finding |
|---|---|
| **Imbalance** | 22.12% default rate (≈ 3.5 : 1) → stratified CV, PR-AUC |
| **Missing / duplicates** | No missing cells, 0 duplicate IDs, 35 duplicates ignoring ID |
| **Categorical codes** | `EDUCATION` / `MARRIAGE` contain undocumented levels ("Unknown", "Others") → clean/merge in the pipeline, do not drop rows |
| **Payment status** | `PAY_k` mixes meanings (−2 no consumption, −1 paid duly, 0 revolving, ≥1 months delayed). Default rate: ~13–17% for ≤0, **34%** at 1-month delay, **69%** at 2-month delay |
| **Persistence** | P(late now \| late last month) = **89.9%** vs 11.1% if not late last month |
| **Credit limit** | Lower `LIMIT_BAL` → higher default (AUC of `LIMIT_BAL` alone = 0.618) |
| **Bills** | Six `BILL_AMT` columns are almost collinear (mean pairwise corr 0.89, VIF up to ~27); some negative bills present |
| **Payment behavior** | Payment-to-bill ratio and utilization (`BILL_AMT / LIMIT_BAL`) are far more informative than raw amounts |
| **Trajectory** | Customers by payment trend → never late: **11.7%** default, improving: 34.3%, late/flat: 38.4%, **deteriorating: 52.8%** |
| **History-aware features** | 23 exploratory engineered features (late-month counts, delay trend, utilization, payment ratios) tested against raw columns to guide Week 2 |

---

## ⚠️ Data Issues & Requirements for Next Stages

1. **Leakage risk:** duplicated rows (609 GMSC / 35 UCI) → handle with grouped/stratified splitting; UCI should be split **by customer ID**.
2. **Sentinel codes (96/98)** and **impossible ratios** → handle inside the pipeline with caps learned from train only.
3. **Missingness is informative** → median imputation + missing-indicator, fit on train only.
4. **UCI categorical cleaning** (EDUCATION/MARRIAGE) and **month ordering** needed before building rolling 3-month features.
5. **Strong multicollinearity** in `BILL_AMT*` → engineered aggregates / regularization / SHAP grouped by feature family.
6. **UCI covers a single 6-month window** → trend features are limited; longer history would improve temporal modeling.
7. **Two different targets** (GMSC: 2-year serious delinquency; UCI: next-month default) → UCI needs **30-day and 90-day target engineering**.

## 📈 Visualizations

Both notebooks include: target distribution, missing-value bars, univariate distributions (log-scaled where skewed), boxplots + IQR outlier analysis, default-rate-by-bin plots, Spearman correlation heatmaps, missingness-vs-target plots, and (UCI) payment-status transition matrix, trajectory archetypes and signal-strength charts.

## 🗂️ Repository Structure

```
├── notebooks/
│   ├── Main_EDA.ipynb          # Give Me Some Credit — full EDA
│   └── Taiwan_EDA.ipynb        # UCI Taiwan — full EDA
├── data/
│   ├── raw/                    # (not committed — see Dataset Sources)
│   └── cleaned/
├── reports/
│   ├── data_report.md
│   ├── eda_findings.md
│   └── data_issues_requirements.md
└── README.md
```

## 🔗 Dataset Sources

- Give Me Some Credit — Kaggle: https://www.kaggle.com/c/GiveMeSomeCredit
- Default of Credit Card Clients — UCI: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients

## ➡️ Next Stage (Week 2)

Train/Validation/Test split → target & feature engineering → unified sklearn pipeline (preprocessing + model) → Logistic Regression baseline vs. XGBoost + Optuna → calibration & threshold → SHAP explainability.

## 👥 Team

Youssef Mohamed · Mariem Attia · Mariam Ebrahim · Mariana · Mariem Abdal Khalek · Nervana Mohamed
