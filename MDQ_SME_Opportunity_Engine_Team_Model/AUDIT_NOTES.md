# Audit Notes

## Source Of Truth

The only original case inputs are the supplied synthetic parquet files and the Mastercard case statement.

The notebooks in `notebooks/` were supplied by the team and contain analytical work built on top of those inputs.

## Data-Derived Outputs

- transaction aggregates by `card_number`
- behavioral features
- Isolation Forest score
- One-Class SVM score
- rank-averaged SME opportunity score
- lead ranking

## Demo Rules Added For The Telegram Product

- segment assignment
- migration stage labels
- recommended products
- explanation-card ordering and signal-strength bars
- illustrative revenue scenarios

These demo rules are intentionally deterministic and inspectable. They are not claimed to be learned by the ML ensemble.

## Model Metric Limitation

The hidden-SME ground truth inside the consumer-card dataset is not available locally. Reported AUC, precision, recall, F1, and confusion matrix are proxy metrics that compare holdout known-business cards with the consumer pool. They are useful for validation but are not the official hidden-SME leaderboard result.

## Notebook Branches

- `block5_ml_modeling.ipynb`, `block6_evaluation.ipynb`, and `block7_sme_detection.ipynb` implement a supervised CatBoost/XGBoost branch.
- `task.ipynb` contains a later one-class Isolation Forest + One-Class SVM branch.
- This Telegram comparison build uses the one-class branch because it avoids directly fitting the detector with all consumer cards labeled as negatives.
