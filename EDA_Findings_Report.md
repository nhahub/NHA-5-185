# Darra AI — Week 1 EDA Findings Report

This report summarizes what we found in each dataset: structure, data quality, target behavior, risk signals, and the decisions they imply for Week 2.

---

# 1. Give Me Some Credit (GMSC)

## 1.1 Dataset overview

| Item | Value |
|---|---|
| Source | Kaggle "Give Me Some Credit" competition |
| Shape | 150,000 rows × 12 columns |
| Granularity | One snapshot per borrower (no time dimension) |
| Target | `SeriousDlqin2yrs` (1 = 90+ days delinquent within 2 years) |
| Role in Darra AI | Long-term risk model + what-if scenarios (income, debt ratio, dependents) |

**Columns:** `ID`, `SeriousDlqin2yrs`, `RevolvingUtilizationOfUnsecuredLines`, `age`, `NumberOfTime30-59DaysPastDueNotWorse`, `DebtRatio`, `MonthlyIncome`, `NumberOfOpenCreditLinesAndLoans`, `NumberOfTimes90DaysLate`, `NumberRealEstateLoansOrLines`, `NumberOfTime60-89DaysPastDueNotWorse`, `NumberOfDependents`

## 1.2 Data quality findings

| # | Issue | Evidence | Severity |
|---|---|---|---|
| 1 | Missing `MonthlyIncome` | 29,731 rows (19.82%) | High |
| 2 | Missing `NumberOfDependents` | 3,924 rows (2.62%) | Low |
| 3 | Duplicates | 0 duplicate IDs, but **609 rows** identical on all real features | Medium (leakage across splits) |
| 4 | Impossible age | 1 row with `age = 0` | Low |
| 5 | **Sentinel codes 96/98** | 269 rows have 96/98 in all three past-due columns | High |
| 6 | Invalid utilization | `RevolvingUtilization` should be in [0,1]; 3,321 rows > 1, 241 rows > 10, max 50,708 | High |
| 7 | Invalid debt ratio | `DebtRatio` > 1 in 35,137 rows, > 10 in 28,877 rows, max 329,664 | High |

## 1.3 Target

- Default rate **6.68%**, imbalance ≈ **14 : 1** (severe).
- Accuracy is misleading → use AUC-PR, recall, class weights.

## 1.4 Distributions & outliers

- `age`: roughly bell-shaped, centered in the 40s–50s.
- `RevolvingUtilization`, `DebtRatio`, `MonthlyIncome`: extremely right-skewed (log scale needed to read them).
- The three past-due counters and `NumberOfDependents`: zero-inflated discrete counts.
- IQR rule flags the most outliers in `DebtRatio` (≈20.9%) and the past-due counters (≈16%). Most of these are linked to the sentinel codes and unit errors above, not genuine extreme customers.

## 1.5 Relationship with the target

| Feature | Spearman with target |
|---|---|
| `NumberOfTimes90DaysLate` | **0.34** |
| `NumberOfTime60-89DaysPastDueNotWorse` | 0.28 |
| `NumberOfTime30-59DaysPastDueNotWorse` | 0.26 |
| `RevolvingUtilizationOfUnsecuredLines` | 0.24 |
| `age` | −0.12 |
| `MonthlyIncome` | −0.07 |

- Default rate rises monotonically with utilization and with past-due counts.
- Younger borrowers default more; default falls steadily with age.
- Lower income → mildly higher default.
- `NumberOfOpenCreditLinesAndLoans` and `NumberRealEstateLoansOrLines` show flat / non-monotonic relationships (weak signals).
- The three past-due counters are highly correlated with each other (same behavior at different severities).

## 1.6 Missingness analysis

| Column | Default rate when missing | When present |
|---|---|---|
| `MonthlyIncome` | 5.61% | 6.95% |
| `NumberOfDependents` | 4.56% | 6.74% |

Missingness is **not completely random** (rates differ), so it carries signal.

## 1.7 Sentinel group

269 rows (≈0.18%) with 96/98 codes have a default rate of **54.65%** (≈8× the base rate) and 45% missing income. They behave like a distinct high-risk group.

## 1.8 Decisions for Week 2

1. Handle `age ≤ 0` as invalid (set to missing / drop in pipeline).
2. Flag 96/98 rows with an indicator and clip the counters.
3. Cap `RevolvingUtilization` and `DebtRatio` (cap values learned from **train only**).
4. Median imputation + missing-indicator for `MonthlyIncome` and `NumberOfDependents`.
5. Handle duplicates carefully in the split to avoid leakage.
6. Use AUC-PR, `class_weight` / `scale_pos_weight`, StratifiedKFold.
7. Log-transform skewed features for the linear baseline.

---

# 2. UCI Default of Credit Card Clients (Taiwan)

## 2.1 Dataset overview

| Item | Value |
|---|---|
| Source | UCI Machine Learning Repository |
| Shape | 30,000 rows × 25 columns |
| Granularity | One row per customer with 6 months of history |
| Time coverage | 6 monthly snapshots (Apr–Sep 2005) |
| Target | `default` (default payment next month) |
| Role in Darra AI | Monthly monitoring + 30/90-day risk trend |

**Column groups:**
- Identity / demographics: `ID`, `SEX`, `EDUCATION`, `MARRIAGE`, `AGE`
- Credit limit: `LIMIT_BAL`
- Payment status (6 months): `PAY_1` … `PAY_6` (`PAY_0` renamed to `PAY_1`)
- Bill amounts: `BILL_AMT1` … `BILL_AMT6`
- Payment amounts: `PAY_AMT1` … `PAY_AMT6`
- Target: `default`

## 2.2 Data quality findings

| # | Issue | Evidence | Severity |
|---|---|---|---|
| 1 | Missing values | None (0 missing cells) | None |
| 2 | Duplicates | 0 duplicate IDs, 35 duplicate rows ignoring ID | Low |
| 3 | Undocumented category levels | `EDUCATION` has "Unknown"/"Others"; `MARRIAGE` has "Unknown"/"Others" | Medium |
| 4 | Negative bill amounts | 6.4% of customers have at least one negative bill (min ≈ −339,603) | Medium |
| 5 | Mixed meaning in `PAY_k` | −2 no consumption, −1 paid duly, 0 revolving, ≥1 months delayed | High (needs encoding care) |
| 6 | Extreme amounts | Heavy right tails in `BILL_AMT*` and `PAY_AMT*` | Medium |
| 7 | Short history | Only 6 months, single window | Limitation |

## 2.3 Target

- Default rate **22.12%**, imbalance ≈ **3.5 : 1** (moderate).
- Use stratified / grouped CV, AUC-PR, recall at a chosen precision, class weights.

## 2.4 Demographics

| Variable | Share | Default rate |
|---|---|---|
| Female | 60.4% | 20.8% |
| Male | 39.6% | 24.2% |
| Graduate school | 35.3% | 19.2% |
| University | 46.8% | 23.7% |
| High school | 16.4% | 25.2% |
| Others | 0.41% | 5.7% |
| Unknown | 1.15% | 7.5% |
| Married | 45.5% | 23.5% |
| Single | 53.2% | 20.9% |
| Marriage: Others | 1.08% | 26.0% |
| Marriage: Unknown | 0.18% | 9.3% |

All demographic variables are statistically significant (chi² p < 0.001) but with **small effect sizes** (Cramér's V: EDUCATION 0.073, age bins 0.056, SEX 0.040, MARRIAGE 0.035). Demographics are weak predictors compared to payment behavior.

## 2.5 Credit limit

- Median `LIMIT_BAL`: **90,000** for defaulters vs **150,000** for non-defaulters.
- AUC of `LIMIT_BAL` alone = **0.618** (Mann-Whitney p ≈ 1e-189).

## 2.6 Payment status (strongest signal)

| `PAY_1` status | Default rate |
|---|---|
| −2 (no consumption) | 13.2% |
| −1 (paid duly) | 16.8% |
| 0 (revolving) | 12.8% |
| 1 month late | **34.0%** |
| 2 months late | **69.1%** |
| 3 months late | 75.8% |
| 4 months late | 68.4% |

- Lateness is highly persistent: P(late now | late last month) = **89.9%** vs **11.1%** if not late last month.
- About 33.6% of customers were late at least once in the 6 months; 21.9% had a "no consumption" month.
- The code 0 ("revolving") is not in the original documentation, and is the most common status (≈70% of customers have it at least once).

## 2.7 Bills, payments and utilization

- Six `BILL_AMT` columns are almost collinear: mean pairwise Spearman **0.84** (Pearson ≈ 0.89), VIF up to **~27**.
- `PAY_*` block mean pairwise correlation 0.67; `PAY_AMT*` block 0.51.
- Raw bill amounts are weak predictors; **bill relative to the limit** (utilization) is informative.
- Payment-to-bill ratio (how much of the previous bill was paid) separates risk better than raw payment amounts. 4.7% of customers have no valid ratio (never had a positive bill).

## 2.8 History-aware (engineered) features — exploratory

23 features were explored (late-month counts, max/mean delay, delay trend, utilization, bill volatility, payment ratios, zero-payment months). Their signal was compared against raw columns to justify history-aware modeling. Compact engineered set has low VIF (≤ 3.3) vs raw bills (up to ~27).

| Trajectory | Customers | Default rate |
|---|---|---|
| Never late | 19,931 | **11.7%** |
| Improving | 1,894 | 34.3% |
| Late, flat trend | 4,605 | 38.4% |
| Deteriorating | 3,570 | **52.8%** |

This supports the "improving / stable / deteriorating" traffic-light idea for monthly monitoring.

## 2.9 Decisions for Week 2

1. Clean `EDUCATION` / `MARRIAGE` (merge undocumented levels into "other", do not drop rows).
2. Order the months correctly (PAY_6 oldest → PAY_1 latest) before building rolling 3-month features.
3. Split with **StratifiedGroupKFold by customer ID**.
4. Replace six raw bills with aggregates (avg bill, trend, volatility, utilization); cap negative/extreme values.
5. Keep raw `PAY_k` for trees; create delay / revolving / no-consumption flags for the linear model.
6. Build 30-day and 90-day targets (target engineering).
7. Use AUC-PR, recall-focused threshold, class weights.

---

# 3. Cross-dataset comparison

| Aspect | GMSC | UCI Taiwan |
|---|---|---|
| Rows | 150,000 | 30,000 |
| Default rate | 6.68% | 22.12% |
| Imbalance | Severe (14:1) | Moderate (3.5:1) |
| Missing values | Yes (income 19.8%, dependents 2.6%) | None |
| Temporal info | None | 6 months |
| Main data problems | Sentinel codes, impossible ratios, duplicates | Undocumented codes, negative bills, collinearity |
| Strongest signal | Past-due counts, utilization | Recent payment status, delay trend |
| Weak signal | Open credit lines, real estate loans | Demographics (SEX, MARRIAGE) |

# 4. Data leakage risks

- **GMSC:** 609 duplicate feature rows can appear in both train and test, inflating scores.
- **UCI:** repeated customer information (6 months of the same customer) → must split/CV by customer ID, and 30/90-day targets must be built so that no future month feeds the features.
- **Both:** imputers, caps, scalers and calibrators must be fit on train only (inside the pipeline).

# 5. Additional data needed

- Longer payment history for UCI (the 6-month window limits trend modeling).
- Actual income / expense / savings data (neither dataset has expenses or savings, which are needed for savings coverage and recommendations).
- A data dictionary for UCI's undocumented codes (PAY = 0, EDUCATION 0/5/6, MARRIAGE = 0).
- Information on how the 96/98 sentinel codes were generated in GMSC.
