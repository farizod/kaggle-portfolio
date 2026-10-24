# Loan Approval Prediction (Playground S4E10)

**Competition:** [https://www.kaggle.com/competitions/playground-series-s4e10](https://www.kaggle.com/competitions/playground-series-s4e10)
**Kaggle notebook:** [https://www.kaggle.com/code/farizodastanaqulova/loan-approval-s4e10-lightgbm](https://www.kaggle.com/code/farizodastanaqulova/loan-approval-s4e10-lightgbm)

| Type | Metric | Public score |
|------|--------|--------------|
| Binary classification | ROC AUC | **0.960** |

## Task
Predict whether a loan application is approved.

## Approach
- Categorical features (home ownership, loan intent, loan grade, prior default flag) handled natively by LightGBM.
- LightGBM, 5-fold stratified CV, test probabilities averaged across folds.

## Files
- `notebook.ipynb` — the Kaggle notebook (reads the attached competition data, runs CV, writes `submission.csv`).
- `kernel-metadata.json` — Kaggle kernel config, including the competition data source.

## Run
The notebook runs as-is on Kaggle with the competition attached. Locally, see the
[repo README](../../README.md): install requirements, place your Kaggle API token,
accept the competition rules, then the shared pipeline `scripts/model_all.py`
reproduces the submission.
