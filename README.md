# Darra AI (درع)

From Crisis Prediction to Financial Resilience: an early-warning system for personal financial health.

## What it does
- Scores payment-distress risk (a proxy for cash-flow crisis) with calibrated probabilities.
- UCI: 30-day and 90-day risk cards, trend, risk band, plain-language drivers (90-day is lower confidence).
- GMSC: long-term structural risk (24-month horizon). The two views are never averaged.
- Scenario engine (what-if) that changes raw inputs only.
- Best Actions engine: top 3 actions with change size, risk before → after, effort level.
- Savings coverage (rule-based): runway_months, shock_coverage, savings left, status band.
- Streamlit dashboard: Overview → Why (SHAP) → What-If → Best Actions → Coverage, with 3 demo profiles.

## Rules
- Models, calibrators and thresholds are frozen in Week 3. No retraining.
- The test set is never used to choose bands, thresholds or action cut-offs.
- Every prediction goes through the saved pipeline (raw input in, calibrated probability out).
- SHAP explains the model and is not causal: use "associated with higher risk".
- Show calibrated probabilities only. Do not rely on accuracy.

## Deliverables
1. Scoring layer (GMSC + UCI) with unified JSON output
2. Scenario engine and scenario library
3. Recommendations engine and top-3 actions report
4. Savings coverage module
5. 30/90-day risk view (UCI) and long-term risk view (GMSC)
6. Streamlit app
7. Test report and 3 demo profiles
8. Updated notebooks and .py modules

## Project plan
- Weeks 1–3: data collection, cleaning, feature engineering
- Weeks 4–6: model training, tuning, SHAP
- Weeks 7–9: what-if optimizer, LLM action-plan layer (optional)
- Weeks 10+: dashboard and jury testing

## Setup
```bash
pip install -r requirements.txt   # pandas, numpy, xgboost, shap, scipy, streamlit
streamlit run app.py              # replace app.py with the actual entry file
```

## Team
Youssef Mohamed, Mariem Attia, Mariam Ebrahim, Mariana, Mariem Abdal Khalek, Nervana Mohamed

## Known limitations
To be recorded by the team for the final report.
