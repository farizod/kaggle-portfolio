# Obesity Risk Prediction (Playground S4E2)

**Competition:** [https://www.kaggle.com/competitions/playground-series-s4e2](https://www.kaggle.com/competitions/playground-series-s4e2)
**Kaggle notebook:** [https://www.kaggle.com/code/farizodastanaqulova/obesity-risk-s4e2-lightgbm](https://www.kaggle.com/code/farizodastanaqulova/obesity-risk-s4e2-lightgbm)

| Type | Metric | Public score |
|------|--------|--------------|
| Multi-class classification (7 classes) | Accuracy | **0.913** |

## Task
Predict an individual's obesity-risk category (7 classes) from lifestyle and body-measurement features.

## Approach
- Target label-encoded; all object/string columns (eating habits, transport, family history, etc.) kept as categoricals.
- LightGBM multi-class, 5-fold stratified CV, probabilities averaged, mapped back to the original class names.

## Files
- `notebook.ipynb` — the Kaggle notebook (reads the attached competition data, runs CV, writes `submission.csv`).
- `kernel-metadata.json` — Kaggle kernel config, including the competition data source.

## Run
The notebook runs as-is on Kaggle with the competition attached. Locally, see the
[repo README](../../README.md): install requirements, place your Kaggle API token,
accept the competition rules, then the shared pipeline `scripts/model_all.py`
reproduces the submission.
