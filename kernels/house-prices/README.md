# House Prices - Advanced Regression Techniques

**Competition:** [https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
**Kaggle notebook:** [https://www.kaggle.com/code/farizodastanaqulova/house-prices-lightgbm](https://www.kaggle.com/code/farizodastanaqulova/house-prices-lightgbm)

| Type | Metric | Public score |
|------|--------|--------------|
| Regression | RMSE on log(SalePrice) | **0.128** |

## Task
Predict the final sale price of residential homes from 79 explanatory features.

## Approach
- `log1p` transform on the target (the metric is RMSE in log space).
- Missing numerics -> median; missing categoricals -> the literal `"None"` (often means 'feature absent'); all object columns kept as categoricals.
- LightGBM regressor, 5-fold CV, predictions inverse-transformed with `expm1`.

## Files
- `notebook.ipynb` — the Kaggle notebook (reads the attached competition data, runs CV, writes `submission.csv`).
- `kernel-metadata.json` — Kaggle kernel config, including the competition data source.

## Run
The notebook runs as-is on Kaggle with the competition attached. Locally, see the
[repo README](../../README.md): install requirements, place your Kaggle API token,
accept the competition rules, then the shared pipeline `scripts/model_all.py`
reproduces the submission.
