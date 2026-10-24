# Bank Churn Prediction (Playground S4E1)

**Competition:** [https://www.kaggle.com/competitions/playground-series-s4e1](https://www.kaggle.com/competitions/playground-series-s4e1)
**Kaggle notebook:** [https://www.kaggle.com/code/farizodastanaqulova/bank-churn-s4e1-lightgbm](https://www.kaggle.com/code/farizodastanaqulova/bank-churn-s4e1-lightgbm)

| Type | Metric | Public score |
|------|--------|--------------|
| Binary classification | ROC AUC | **0.887** |

## Task
Predict whether a bank customer closes their account (churns).

## Approach
- Dropped `id` and `CustomerId`; kept `Geography`, `Gender`, and `Surname` as categoricals (LightGBM handles them natively).
- LightGBM, 5-fold stratified CV, out-of-fold AUC, test probabilities averaged across folds.

## Files
- `notebook.ipynb` — the Kaggle notebook (reads the attached competition data, runs CV, writes `submission.csv`).
- `kernel-metadata.json` — Kaggle kernel config, including the competition data source.

## Run
The notebook runs as-is on Kaggle with the competition attached. Locally, see the
[repo README](../../README.md): install requirements, place your Kaggle API token,
accept the competition rules, then the shared pipeline `scripts/model_all.py`
reproduces the submission.
