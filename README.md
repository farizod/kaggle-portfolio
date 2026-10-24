# Kaggle Competitions Portfolio

Solutions for 7 Kaggle competitions — tabular classification/regression and
computer vision — built with **LightGBM** and lightweight feature engineering.
Each competition has a self-contained, reproducible notebook that reads the
competition data, runs cross-validation, and writes a submission file.

**Kaggle profile:** [kaggle.com/farizodastanaqulova](https://www.kaggle.com/farizodastanaqulova)

## Results

| # | Competition | Type | Metric | Score | Notebook |
|---|-------------|------|--------|-------|----------|
| 1 | [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic) | Binary classification | Accuracy | **0.799** | [notebook](kernels/spaceship-titanic/notebook.ipynb) · [Kaggle](https://www.kaggle.com/code/farizodastanaqulova/spaceship-titanic-lightgbm-0-799) |
| 2 | [Forest Cover Type Prediction](https://www.kaggle.com/competitions/forest-cover-type-prediction) | Multi-class (7) | Accuracy | **0.747** | [notebook](kernels/forest-cover-type-prediction/notebook.ipynb) · [Kaggle](https://www.kaggle.com/code/farizodastanaqulova/forest-cover-type-prediction-lightgbm-0-747) |
| 3 | [Bank Churn (Playground S4E1)](https://www.kaggle.com/competitions/playground-series-s4e1) | Binary classification | ROC AUC | **0.887** | [notebook](kernels/s4e1/notebook.ipynb) · [Kaggle](https://www.kaggle.com/code/farizodastanaqulova/bank-churn-s4e1-lightgbm) |
| 4 | [Obesity Risk (Playground S4E2)](https://www.kaggle.com/competitions/playground-series-s4e2) | Multi-class (7) | Accuracy | **0.913** | [notebook](kernels/s4e2/notebook.ipynb) · [Kaggle](https://www.kaggle.com/code/farizodastanaqulova/obesity-risk-s4e2-lightgbm) |
| 5 | [Loan Approval (Playground S4E10)](https://www.kaggle.com/competitions/playground-series-s4e10) | Binary classification | ROC AUC | **0.960** | [notebook](kernels/s4e10/notebook.ipynb) · [Kaggle](https://www.kaggle.com/code/farizodastanaqulova/loan-approval-s4e10-lightgbm) |
| 6 | [House Prices](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) | Regression | RMSE (log) | **0.128** | [notebook](kernels/house-prices/notebook.ipynb) · [Kaggle](https://www.kaggle.com/code/farizodastanaqulova/house-prices-lightgbm) |
| 7 | [Digit Recognizer](https://www.kaggle.com/competitions/digit-recognizer) | Multi-class (10), images | Accuracy | **0.973** | [notebook](kernels/digit-recognizer/notebook.ipynb) · [Kaggle](https://www.kaggle.com/code/farizodastanaqulova/digit-recognizer-mnist-lightgbm-0-973) |

*Scores are public leaderboard scores.*

## Approach

- **Model:** LightGBM (gradient-boosted trees) for every competition — strong,
  fast, and handles mixed categorical/numeric tabular data natively.
- **Validation:** 5-fold stratified cross-validation (KFold for regression),
  test predictions averaged across folds. Digit Recognizer uses a 10% holdout
  with early stopping.
- **Feature engineering (Spaceship Titanic):** passenger group & group size from
  `PassengerId`, deck/number/side from `Cabin`, total spend and a no-spend flag.
- **House Prices:** `log1p` target transform, median imputation for numerics,
  `"None"` for missing categoricals, `expm1` on predictions.
- **Binary (AUC) tasks:** predicted probabilities; **multi-class:** averaged
  class probabilities with argmax; **Spaceship:** boolean prediction.

## Repository layout

```
scripts/
  model_all.py     # training pipeline: build models + submissions for all 7
kernels/<comp>/
  README.md              # that competition's write-up
  notebook.ipynb         # the public Kaggle notebook
  kernel-metadata.json   # Kaggle kernel config (competition data source)
subs/                # generated submission CSVs
requirements.txt     # Python dependencies
LICENSE              # MIT
```

Each competition folder has its own README with the task, approach, and score.

## Reproduce

```bash
pip install -r requirements.txt
# place your Kaggle API token at ~/.kaggle/kaggle.json, accept each competition's rules, then:
python scripts/model_all.py all   # downloads happen separately; see data/ note below
```

> Competition datasets are **not** included in this repo (Kaggle terms). Download
> them with the Kaggle API (`kaggle competitions download -c <slug>`) after
> accepting each competition's rules.

## Tech

Python · pandas · scikit-learn · LightGBM · Kaggle API

## License

[MIT](LICENSE) © 2026 Farizod Astanaqulova
