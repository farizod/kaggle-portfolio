# Digit Recognizer (MNIST)

**Competition:** [https://www.kaggle.com/competitions/digit-recognizer](https://www.kaggle.com/competitions/digit-recognizer)
**Kaggle notebook:** [https://www.kaggle.com/code/farizodastanaqulova/digit-recognizer-mnist-lightgbm-0-973](https://www.kaggle.com/code/farizodastanaqulova/digit-recognizer-mnist-lightgbm-0-973)

| Type | Metric | Public score |
|------|--------|--------------|
| Multi-class classification (10 classes), image data | Accuracy | **0.973** |

## Task
Classify 28x28 grayscale images of handwritten digits (0-9).

## Approach
- 784 raw pixel values scaled to [0, 1].
- LightGBM multi-class with a 10% stratified holdout and early stopping (faster than full k-fold on 784 features).
- A convolutional network would score higher (~0.99); LightGBM is used here to keep one consistent, dependency-light approach across all competitions.

## Files
- `notebook.ipynb` — the Kaggle notebook (reads the attached competition data, runs CV, writes `submission.csv`).
- `kernel-metadata.json` — Kaggle kernel config, including the competition data source.

## Run
The notebook runs as-is on Kaggle with the competition attached. Locally, see the
[repo README](../../README.md): install requirements, place your Kaggle API token,
accept the competition rules, then the shared pipeline `scripts/model_all.py`
reproduces the submission.
