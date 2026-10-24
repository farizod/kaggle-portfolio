# Forest Cover Type Prediction

**Competition:** [https://www.kaggle.com/competitions/forest-cover-type-prediction](https://www.kaggle.com/competitions/forest-cover-type-prediction)
**Kaggle notebook:** [https://www.kaggle.com/code/farizodastanaqulova/forest-cover-type-prediction-lightgbm-0-747](https://www.kaggle.com/code/farizodastanaqulova/forest-cover-type-prediction-lightgbm-0-747)

| Type | Metric | Public score |
|------|--------|--------------|
| Multi-class classification (7 classes) | Accuracy | **0.747** |

## Task
Predict the forest cover type (one of 7 classes) for 30x30m patches from cartographic variables only.

## Approach
- 54 cartographic features used as-is (elevation, slope, soil types, wilderness areas, hillshade, distances).
- LightGBM multi-class (softmax), 5-fold stratified CV, class probabilities averaged, argmax for the final label.

## Files
- `notebook.ipynb` — the Kaggle notebook (reads the attached competition data, runs CV, writes `submission.csv`).
- `kernel-metadata.json` — Kaggle kernel config, including the competition data source.

## Run
The notebook runs as-is on Kaggle with the competition attached. Locally, see the
[repo README](../../README.md): install requirements, place your Kaggle API token,
accept the competition rules, then the shared pipeline `scripts/model_all.py`
reproduces the submission.
