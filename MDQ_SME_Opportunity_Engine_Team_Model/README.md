# AI SME Opportunity Engine - Team Model Version

Separate comparison build of the Telegram banking assistant for Mastercard Data Quest 2026.

This version uses the team's hidden-SME modeling approach from `notebooks/task.ipynb`: a one-class ensemble trained only on known business cards.

## Honest Model Description

The model does not claim to know which consumers are definitely entrepreneurs.

It learns what typical business-card behavior looks like and ranks consumer cards by similarity to that behavior. The output `SME Score` is a percentile-like opportunity rank from 0 to 100, not a calibrated probability.

The product message stays the same:

> We do not punish hidden SMEs. We identify growing business behavior and connect clients with useful financial products.

## Why This Version Exists

The original bot build remains unchanged in:

```text
../MDQ_SME_Opportunity_Engine/
```

This comparison build lives in:

```text
../MDQ_SME_Opportunity_Engine_Team_Model/
```

The notebooks supplied by the team are preserved in `notebooks/`.

## Team Model Flow

1. Load business cards, consumer cards, and merchant reference parquet files.
2. Aggregate transaction behavior into card-level features.
3. Use the business-card population only for model fitting.
4. Split known business cards 80/20.
5. Fit preprocessing only on the 80% business training subset.
6. Train `IsolationForest` and `OneClassSVM`.
7. Evaluate whether the models recognize unseen holdout business cards.
8. Rank all consumers by business-likeness using a rank-averaged ensemble.
9. Add explainable segments, product offers, and bank revenue estimates for the bot.

## Main Behavioral Signals

- turnover and average ticket
- transaction frequency and active days
- B2B MCC spending: ads, cloud, suppliers, logistics, accounting
- recurring payments
- merchant concentration
- MCC concentration
- online/offline activity
- international services
- weekday, weekend, business-hour, and night behavior
- monthly turnover growth

## Run Locally

```powershell
pip install -r requirements.txt
python train_pipeline.py
copy .env.example .env
notepad .env
python -m bot.main
```

Paste the BotFather token into `BOT_TOKEN` before starting the bot.

The loader accepts source parquet files in the project folder, `data/raw/`, the parent folder, or Downloads. Files with suffixes such as `(1)` are supported.

## Generated Bot Files

- `data/final_predictions.parquet`
- `data/client_features.parquet`
- `data/client_explanations.parquet`
- `data/top_leads.parquet`
- `data/dashboard_metrics.json`
- `models/team_one_class_ensemble.joblib`
- `models/model_metrics.json`

The Telegram bot never trains a model live. It reads the generated CSV/JSON files.

`data/top_leads.parquet` is a compact top-100 export for manual review. Inside Telegram, the
`Top SME Leads` screen paginates through the full ranked portfolio of 80,000 consumer cards.

## Truthfulness Boundary

The following outputs are calculated from the supplied synthetic parquet files:

- card-level behavioral features
- the Isolation Forest and One-Class SVM scores
- the rank-averaged SME opportunity score
- the top-lead ordering

The following bot outputs are transparent demo rules added on top of the model score:

- business segment names
- migration stage labels
- product suggestions
- behavioral explanation cards
- revenue scenarios

Revenue scenarios use assumed margins for demonstration. They are not observed bank income and must be validated against actual bank tariffs before external use.

## Demo IDs

Run `python train_pipeline.py` once. The pipeline updates this section with real top-ranked consumer IDs:

<!-- DEMO_IDS_START -->
- `5100612020402608`
- `5176476691114937`
- `5201499082819830`
- `5338474007563215`
- `5486737863418710`
- `5176512938679108`
- `5176513825363681`
- `5228596285044584`
<!-- DEMO_IDS_END -->

## Bot Screens

- `/start` main menu
- Analyze Client
- Top SME Leads
- Client Card
- Why Detected
- Product Recommendation
- Revenue Potential
- Portfolio Dashboard
- How It Works

## About The Supplied Notebooks

The supplied `block5_ml_modeling.ipynb`, `block6_evaluation.ipynb`, and `block7_sme_detection.ipynb` contain a supervised CatBoost/XGBoost branch.

The larger `task.ipynb` contains a later one-class branch designed specifically for hidden-SME ranking. This runnable comparison build follows the one-class branch because it matches the hidden-SME business question more closely.

