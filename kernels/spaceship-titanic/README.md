# Spaceship Titanic

**Competition:** [https://www.kaggle.com/competitions/spaceship-titanic](https://www.kaggle.com/competitions/spaceship-titanic)
**Kaggle notebook:** [https://www.kaggle.com/code/farizodastanaqulova/spaceship-titanic-lightgbm-0-799](https://www.kaggle.com/code/farizodastanaqulova/spaceship-titanic-lightgbm-0-799)

| Type | Metric | Public score |
|------|--------|--------------|
| Binary classification | Accuracy | **0.799** |

## Task
Predict which passengers were transported to an alternate dimension after the Spaceship Titanic's collision with a spacetime anomaly.

## Approach
- Feature engineering from `PassengerId` (group id + group size), `Cabin` (deck / number / side), and the five spend columns (total spend + a no-spend flag).
- `CryoSleep` and `VIP` cast to numeric; `HomePlanet`, `Destination`, `Deck`, `Side` kept as categoricals.
- LightGBM, 5-fold stratified CV, test probabilities averaged across folds, thresholded at 0.5 to a boolean.

## Files
- `notebook.ipynb` — the Kaggle notebook (reads the attached competition data, runs CV, writes `submission.csv`).
- `kernel-metadata.json` — Kaggle kernel config, including the competition data source.

## Run
The notebook runs as-is on Kaggle with the competition attached. Locally, see the
[repo README](../../README.md): install requirements, place your Kaggle API token,
accept the competition rules, then the shared pipeline `scripts/model_all.py`
reproduces the submission.
