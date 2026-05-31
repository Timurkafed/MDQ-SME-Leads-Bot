from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
RANDOM_STATE = 42

B2B_MCC = {
    "7311", "7372", "5968", "4816", "7399", "5045", "7392", "4214",
    "4215", "5046", "5111", "5199", "5099", "8931", "7379",
}
LARGE_TXN_THRESHOLD = 1_000_000

# Exact behavioral feature set used by the one-class team-model adapter.
TEAM_MODEL_FEATURES = [
    "total_amount", "mean_amount", "median_amount", "std_amount", "max_amount",
    "min_amount", "transaction_count", "active_days", "period_days",
    "transactions_per_active_day", "transactions_per_period_day", "unique_merchants",
    "top_merchant_share", "unique_mcc", "top_mcc_share", "unique_countries",
    "weekend_share", "business_hours_share", "night_share", "avg_transaction_hour",
    "std_transaction_hour", "avg_days_between_txn", "std_days_between_txn",
    "repeated_amount_share", "b2b_txn_share", "b2b_amount_share", "n_b2b_mcc",
    "p90_amount", "p95_amount", "p99_amount", "mean_to_median_ratio",
    "large_txn_count", "large_txn_share", "merchant_hhi", "mcc_hhi",
    "recurring_share", "recurring_amount_share", "foreign_share", "us_share",
    "ireland_share", "recurring_capable_share", "foreign_merchant_share",
    "us_merchant_share", "ireland_merchant_share",
]

LOG_FEATURES = [
    "total_amount", "mean_amount", "median_amount", "max_amount", "std_amount",
    "p90_amount", "p95_amount", "transaction_count",
]


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading source parquet files...")
    business, consumers, merchants = load_sources()

    print("Building team notebook card-level features...")
    business_features = build_card_features(business, merchants)
    consumer_features = build_card_features(consumers, merchants)

    print("Training team one-class ensemble (Isolation Forest + One-Class SVM)...")
    artifacts, scores, metrics = train_team_ensemble(business_features, consumer_features)

    print("Creating banking-product exports for Telegram...")
    predictions, explanations = enrich_predictions(consumer_features, scores)
    export_outputs(predictions, consumer_features, explanations, metrics)
    joblib.dump(artifacts, MODEL_DIR / "team_one_class_ensemble.joblib")

    demo_ids = predictions.head(10)["card_number"].astype(str).tolist()
    update_readme_demo_ids(demo_ids)

    print("\nDone.")
    print("Model: Isolation Forest + One-Class SVM rank ensemble")
    print(json.dumps(metrics, indent=2))
    print("Demo IDs:", ", ".join(demo_ids[:8]))


def resolve_input_file(stem: str) -> Path:
    search_dirs = [ROOT_DIR, ROOT_DIR / "data" / "raw", ROOT_DIR.parent, Path.home() / "Downloads"]
    for directory in search_dirs:
        if not directory.exists():
            continue
        exact = directory / f"{stem}.parquet"
        if exact.exists():
            return exact
        matches = sorted(directory.glob(f"{stem}*.parquet"))
        if matches:
            return matches[0]
    raise FileNotFoundError(
        f"Missing {stem}.parquet. Put it in this project, data/raw, the parent folder, or Downloads."
    )


def load_sources() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    business_path = resolve_input_file("business_cards_MDQ")
    consumer_path = resolve_input_file("consumer_cards_MDQ")
    merchants_path = resolve_input_file("merchants_reference")
    print(f"  business:  {business_path}")
    print(f"  consumer:  {consumer_path}")
    print(f"  merchants: {merchants_path}")

    business = pd.read_parquet(business_path)
    consumers = pd.read_parquet(consumer_path)
    merchants = pd.read_parquet(merchants_path)
    for frame in (business, consumers):
        frame["card_number"] = frame["card_number"].astype(str)
        frame["merchant_id"] = frame["merchant_id"].astype(str)
        frame["mcc"] = pd.to_numeric(frame["mcc"], errors="coerce").fillna(-1).astype(int)
        frame["transaction_amount_kzt"] = pd.to_numeric(
            frame["transaction_amount_kzt"], errors="coerce"
        ).fillna(0.0)
    merchants["merchant_id"] = merchants["merchant_id"].astype(str)
    return business, consumers, merchants


def build_card_features(tx: pd.DataFrame, merchants: pd.DataFrame) -> pd.DataFrame:
    frame = add_transaction_signals(tx, merchants)
    g = frame.groupby("card_number", observed=True, sort=False)

    features = g.agg(
        total_amount=("amount", "sum"),
        mean_amount=("amount", "mean"),
        median_amount=("amount", "median"),
        std_amount=("amount", "std"),
        max_amount=("amount", "max"),
        min_amount=("amount", "min"),
        transaction_count=("amount", "size"),
        active_days=("transaction_day", "nunique"),
        first_transaction=("transaction_ts", "min"),
        last_transaction=("transaction_ts", "max"),
        unique_merchants=("merchant_id", "nunique"),
        unique_mcc=("mcc", "nunique"),
        unique_countries=("country", "nunique"),
        weekend_share=("is_weekend", "mean"),
        business_hours_share=("is_business_hours", "mean"),
        night_share=("is_night", "mean"),
        avg_transaction_hour=("hour", "mean"),
        std_transaction_hour=("hour", "std"),
        recurring_share=("is_recurring_bool", "mean"),
        recurring_capable_share=("recurring_capable_bool", "mean"),
        foreign_share=("is_foreign", "mean"),
        us_share=("is_us", "mean"),
        ireland_share=("is_ireland", "mean"),
        foreign_merchant_share=("is_foreign_merchant", "mean"),
        us_merchant_share=("is_us_merchant", "mean"),
        ireland_merchant_share=("is_ireland_merchant", "mean"),
        online_share=("is_online", "mean"),
        offline_share=("is_offline", "mean"),
        tokenized_share=("tokenized_bool", "mean"),
        top_mcc=("mcc", top_value),
        top_merchant_category=("mcc_category", top_value),
    )
    features["period_days"] = (
        (features["last_transaction"] - features["first_transaction"]).dt.days + 1
    ).clip(lower=1)
    features["transactions_per_active_day"] = (
        features["transaction_count"] / features["active_days"].replace(0, np.nan)
    )
    features["transactions_per_period_day"] = features["transaction_count"] / features["period_days"]

    merchant_counts = frame.groupby(["card_number", "merchant_id"], observed=True).size()
    merchant_totals = merchant_counts.groupby(level=0).sum()
    merchant_shares = merchant_counts / merchant_totals
    features["top_merchant_share"] = merchant_shares.groupby(level=0).max()
    features["merchant_hhi"] = (merchant_shares ** 2).groupby(level=0).sum()

    mcc_counts = frame.groupby(["card_number", "mcc"], observed=True).size()
    mcc_totals = mcc_counts.groupby(level=0).sum()
    mcc_shares = mcc_counts / mcc_totals
    features["top_mcc_share"] = mcc_shares.groupby(level=0).max()
    features["mcc_hhi"] = (mcc_shares ** 2).groupby(level=0).sum()

    amount_g = frame.groupby("card_number", observed=True)["amount"]
    features["p90_amount"] = amount_g.quantile(0.90)
    features["p95_amount"] = amount_g.quantile(0.95)
    features["p99_amount"] = amount_g.quantile(0.99)
    features["mean_to_median_ratio"] = (
        features["mean_amount"] / features["median_amount"].replace(0, np.nan)
    )

    b2b = frame[frame["is_b2b"]]
    b2b_amount = b2b.groupby("card_number", observed=True)["amount"].sum()
    features["b2b_txn_share"] = g["is_b2b"].mean()
    features["b2b_amount_share"] = b2b_amount / features["total_amount"].replace(0, np.nan)
    features["n_b2b_mcc"] = b2b.groupby("card_number", observed=True)["mcc"].nunique()

    recurring_amount = frame[frame["is_recurring_bool"]].groupby(
        "card_number", observed=True
    )["amount"].sum()
    features["recurring_amount_share"] = (
        recurring_amount / features["total_amount"].replace(0, np.nan)
    )

    features["large_txn_count"] = g["is_large"].sum()
    features["large_txn_share"] = g["is_large"].mean()
    features["repeated_amount_share"] = repeated_amount_share(frame)

    ordered = frame.sort_values(["card_number", "transaction_ts"]).copy()
    ordered["days_between_txn"] = ordered.groupby("card_number", observed=True)[
        "transaction_ts"
    ].diff().dt.days
    interval = ordered.groupby("card_number", observed=True)["days_between_txn"].agg(["mean", "std"])
    features["avg_days_between_txn"] = interval["mean"]
    features["std_days_between_txn"] = interval["std"]

    features["monthly_growth"] = monthly_growth(frame)
    features["active_months"] = frame.groupby("card_number", observed=True)["month"].nunique().clip(lower=1)

    # Presentation aliases keep the Telegram layer independent from notebook naming.
    features["tx_count"] = features["transaction_count"]
    features["total_turnover"] = features["total_amount"]
    features["avg_tx_amount"] = features["mean_amount"]
    features["median_tx_amount"] = features["median_amount"]
    features["max_tx_amount"] = features["max_amount"]
    features["foreign_tx_share"] = features["foreign_share"]
    features["merchant_concentration"] = features["top_merchant_share"]
    features["mcc_diversity"] = (1 - features["mcc_hhi"]).clip(0, 1)

    features = features.replace([np.inf, -np.inf], np.nan).fillna(0).reset_index()
    features["card_number"] = features["card_number"].astype(str)
    # Keep the notebook's deterministic card order. The business train/holdout
    # split depends on row positions, so order is part of reproducibility.
    return features.sort_values("card_number").reset_index(drop=True)


def add_transaction_signals(tx: pd.DataFrame, merchants: pd.DataFrame) -> pd.DataFrame:
    frame = tx.copy()
    merchant_ref = merchants[["merchant_id", "merchant_country", "recurring_capable"]].copy()
    frame = frame.merge(merchant_ref, on="merchant_id", how="left")

    frame["amount"] = frame["transaction_amount_kzt"].abs()
    frame["transaction_ts"] = pd.to_datetime(frame["transaction_timestamp"], errors="coerce")
    frame["transaction_ts"] = frame["transaction_ts"].fillna(
        pd.to_datetime(frame["transaction_date"], errors="coerce")
    )
    frame["transaction_day"] = frame["transaction_ts"].dt.normalize()
    frame["month"] = frame["transaction_ts"].dt.to_period("M")
    frame["hour"] = frame["transaction_ts"].dt.hour.fillna(0)
    frame["is_weekend"] = frame["transaction_ts"].dt.dayofweek.ge(5).fillna(False)
    frame["is_business_hours"] = frame["hour"].between(9, 18)
    frame["is_night"] = frame["hour"].between(0, 6)

    frame["is_recurring_bool"] = to_bool(frame["is_recurring"])
    frame["tokenized_bool"] = to_bool(frame["tokenized"])
    frame["recurring_capable_bool"] = to_bool(frame["recurring_capable"])
    frame["is_large"] = frame["amount"].ge(LARGE_TXN_THRESHOLD)
    frame["is_b2b"] = frame["mcc"].astype(str).isin(B2B_MCC)

    country = frame["country"].astype(str)
    merchant_country = frame["merchant_country"].fillna("").astype(str)
    frame["is_foreign"] = country.ne("Kazakhstan")
    frame["is_us"] = country.eq("US")
    frame["is_ireland"] = country.eq("Ireland")
    frame["is_foreign_merchant"] = merchant_country.ne("Kazakhstan")
    frame["is_us_merchant"] = merchant_country.eq("US")
    frame["is_ireland_merchant"] = merchant_country.eq("Ireland")

    channel = frame["channel"].fillna("").astype(str).str.lower()
    frame["is_online"] = channel.str.contains("online|ecom|internet|web|mobile|app", regex=True)
    frame["is_offline"] = ~frame["is_online"]
    frame["mcc_category"] = frame["mcc"].map(categorize_mcc)
    return frame


def to_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.fillna(False).astype(str).str.lower().isin({"true", "1", "yes", "y", "t"})


def top_value(series: pd.Series) -> Any:
    mode = series.mode()
    return mode.iloc[0] if not mode.empty else "Unknown"


def repeated_amount_share(frame: pd.DataFrame) -> pd.Series:
    counts = frame.groupby(["card_number", "amount"], observed=True).size().rename("n")
    repeated = counts[counts.gt(1)].groupby(level=0).sum()
    totals = frame.groupby("card_number", observed=True).size()
    return repeated.reindex(totals.index, fill_value=0) / totals


def monthly_growth(frame: pd.DataFrame) -> pd.Series:
    monthly = (
        frame.groupby(["card_number", "month"], observed=True)["amount"]
        .sum()
        .rename("amount")
        .reset_index()
        .sort_values(["card_number", "month"])
    )
    summary = monthly.groupby("card_number", observed=True).agg(
        first=("amount", "first"), last=("amount", "last"), months=("month", "nunique")
    )
    growth = np.where(
        summary["months"].gt(1) & summary["first"].gt(0),
        (summary["last"] - summary["first"]) / summary["first"],
        0,
    )
    return pd.Series(growth, index=summary.index).clip(-1, 5)


def prep_matrix(features: pd.DataFrame) -> np.ndarray:
    matrix = features[TEAM_MODEL_FEATURES].astype(float).copy()
    for column in LOG_FEATURES:
        matrix[column] = np.log1p(np.clip(matrix[column], 0, None))
    return matrix.values


def train_team_ensemble(
    business: pd.DataFrame,
    consumers: pd.DataFrame,
) -> tuple[dict[str, Any], np.ndarray, dict[str, Any]]:
    biz_train, biz_holdout = train_test_split(
        business, test_size=0.20, random_state=RANDOM_STATE
    )

    train_raw = prep_matrix(biz_train)
    holdout_raw = prep_matrix(biz_holdout)
    consumer_raw = prep_matrix(consumers)

    scaler = StandardScaler().fit(train_raw)
    x_train = scaler.transform(train_raw)
    x_holdout = scaler.transform(holdout_raw)
    x_consumer = scaler.transform(consumer_raw)

    iso_result = select_isolation_forest(x_train, x_holdout, x_consumer)
    ocs_result = select_one_class_svm(x_train, x_holdout, x_consumer)

    iso = iso_result["model"]
    ocsvm = ocs_result["model"]
    iso_train = iso_result["scores_train"]
    iso_holdout = iso_result["scores_holdout"]
    iso_consumer = iso_result["scores_consumer"]
    ocs_train = ocs_result["scores_train"]
    ocs_holdout = ocs_result["scores_holdout"]
    ocs_consumer = ocs_result["scores_consumer"]

    # Submission score follows the notebook: rank-average consumers from 0 to 1.
    consumer_score = 0.5 * (
        rankdata(iso_consumer) / len(iso_consumer)
        + rankdata(ocs_consumer) / len(ocs_consumer)
    )

    y_proxy = np.concatenate([np.ones(len(x_holdout)), np.zeros(len(x_consumer))])
    iso_pool = np.concatenate([iso_holdout, iso_consumer])
    ocs_pool = np.concatenate([ocs_holdout, ocs_consumer])
    ensemble_pool = 0.5 * (
        rankdata(iso_pool) / len(iso_pool)
        + rankdata(ocs_pool) / len(ocs_pool)
    )
    proxy_pred = ensemble_pool >= 0.95

    iso_tau = float(np.percentile(iso_train, 10))
    ocs_tau = float(np.percentile(ocs_train, 10))
    metrics = {
        "metric_scope": "Proxy evaluation: holdout business cards as positives and consumer pool as negatives",
        "roc_auc": float(roc_auc_score(y_proxy, ensemble_pool)),
        "precision": float(precision_score(y_proxy, proxy_pred, zero_division=0)),
        "recall": float(recall_score(y_proxy, proxy_pred, zero_division=0)),
        "f1_score": float(f1_score(y_proxy, proxy_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_proxy, proxy_pred).tolist(),
        "holdout_business_rows": int(len(x_holdout)),
        "consumer_rows": int(len(x_consumer)),
        "isolation_forest_holdout_recall": float(np.mean(iso_holdout >= iso_tau)),
        "isolation_forest_consumer_pass_rate": float(np.mean(iso_consumer >= iso_tau)),
        "ocsvm_holdout_recall": float(np.mean(ocs_holdout >= ocs_tau)),
        "ocsvm_consumer_pass_rate": float(np.mean(ocs_consumer >= ocs_tau)),
        "isolation_forest_params": iso_result["params"],
        "ocsvm_params": ocs_result["params"],
        "score_interpretation": "Consumer percentile-like opportunity rank, not a calibrated probability",
    }
    artifacts = {
        "model_name": "Isolation Forest + One-Class SVM rank ensemble",
        "feature_columns": TEAM_MODEL_FEATURES,
        "log_features": LOG_FEATURES,
        "scaler": scaler,
        "isolation_forest": iso,
        "one_class_svm": ocsvm,
    }
    return artifacts, consumer_score, metrics


def evaluate_config(
    model: Any,
    x_train: np.ndarray,
    x_holdout: np.ndarray,
    x_consumer: np.ndarray,
    params: dict[str, Any],
) -> dict[str, Any]:
    score_fn = model.score_samples if hasattr(model, "score_samples") else model.decision_function
    scores_train = score_fn(x_train)
    scores_holdout = score_fn(x_holdout)
    scores_consumer = score_fn(x_consumer)
    tau = float(np.percentile(scores_train, 10))
    recall_holdout = float(np.mean(scores_holdout >= tau))
    pass_rate_consumers = float(np.mean(scores_consumer >= tau))
    return {
        "params": params,
        "model": model,
        "scores_train": scores_train,
        "scores_holdout": scores_holdout,
        "scores_consumer": scores_consumer,
        "recall_holdout_biz": recall_holdout,
        "pass_rate_consumers": pass_rate_consumers,
        "gap": recall_holdout - pass_rate_consumers,
    }


def select_isolation_forest(
    x_train: np.ndarray,
    x_holdout: np.ndarray,
    x_consumer: np.ndarray,
) -> dict[str, Any]:
    results = []
    for n_estimators in [100, 200, 300, 500]:
        for contamination in ["auto", 0.05, 0.10]:
            params = {"n_estimators": n_estimators, "contamination": contamination}
            model = IsolationForest(
                **params, random_state=RANDOM_STATE, n_jobs=-1
            ).fit(x_train)
            results.append(evaluate_config(model, x_train, x_holdout, x_consumer, params))
    return max(results, key=lambda result: result["gap"])


def select_one_class_svm(
    x_train: np.ndarray,
    x_holdout: np.ndarray,
    x_consumer: np.ndarray,
) -> dict[str, Any]:
    rng = np.random.RandomState(RANDOM_STATE)
    sample_idx = rng.choice(len(x_train), min(8000, len(x_train)), replace=False)
    x_fit = x_train[sample_idx]
    results = []
    for nu in [0.05, 0.10, 0.20]:
        for gamma in ["scale", "auto"]:
            params = {"nu": nu, "gamma": gamma}
            model = OneClassSVM(kernel="rbf", **params).fit(x_fit)
            results.append(evaluate_config(model, x_train, x_holdout, x_consumer, params))
    return max(results, key=lambda result: result["gap"])


def enrich_predictions(
    consumers: pd.DataFrame, score: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = consumers.copy()
    frame["sme_rank_score"] = score
    frame["sme_score"] = np.rint(frame["sme_rank_score"] * 100).clip(0, 100).astype(int)
    frame["score_band"] = frame["sme_rank_score"].map(score_band)
    frame["migration_stage"] = frame["sme_rank_score"].map(migration_stage)

    benchmarks = build_benchmarks(frame)
    frame["segment"] = frame.apply(lambda row: segment_client(row, benchmarks), axis=1)
    frame["recommended_product"] = frame["segment"].map(product_for_segment)

    revenue = frame.apply(estimate_revenue, axis=1, result_type="expand")
    for column in revenue.columns:
        frame[column] = revenue[column]

    explanations = frame.apply(
        lambda row: generate_explanation(row, benchmarks), axis=1, result_type="expand"
    )
    explanations.insert(0, "card_number", frame["card_number"].values)

    prediction_columns = [
        "card_number", "sme_rank_score", "sme_score", "score_band", "segment",
        "migration_stage", "recommended_product", "illustrative_monthly_revenue_kzt",
        "illustrative_annual_revenue_kzt", "revenue_scenario_assumption", "tx_count", "active_days",
        "total_turnover", "avg_tx_amount", "online_share", "recurring_share",
        "unique_merchants", "unique_mcc", "mcc_diversity", "merchant_concentration",
        "b2b_txn_share", "b2b_amount_share", "monthly_growth", "top_mcc",
        "top_merchant_category",
    ]
    predictions = frame[prediction_columns].sort_values(
        ["sme_score", "sme_rank_score", "total_turnover"], ascending=[False, False, False]
    )
    return predictions, explanations


def build_benchmarks(frame: pd.DataFrame) -> dict[str, float]:
    columns = [
        "total_turnover", "avg_tx_amount", "tx_count", "monthly_growth",
        "b2b_amount_share", "recurring_share", "online_share", "merchant_concentration",
    ]
    result = {}
    for column in columns:
        series = pd.to_numeric(frame[column], errors="coerce").fillna(0)
        result[f"{column}_p90"] = float(series.quantile(0.90))
    return result


def segment_client(row: pd.Series, benchmarks: dict[str, float]) -> str:
    category = str(row["top_merchant_category"]).lower()
    if row["recurring_share"] >= 0.18 or (
        row["recurring_capable_share"] >= 0.55 and row["recurring_share"] >= 0.08
    ):
        return "Subscription / Recurring Business"
    if row["online_share"] >= 0.60 and any(
        signal in category for signal in ["digital", "e-commerce", "ads", "software", "retail"]
    ):
        return "Online Seller"
    if row["b2b_amount_share"] >= 0.20 or row["foreign_share"] >= 0.15:
        return "Freelancer / Digital Professional"
    if row["offline_share"] >= 0.60 and any(
        signal in category for signal in ["local", "food", "beauty", "auto", "transport", "retail"]
    ):
        return "Local Service Provider"
    if row["monthly_growth"] >= 0.25 or row["total_turnover"] >= benchmarks["total_turnover_p90"]:
        return "Growing Microbusiness"
    return "Online Seller" if row["online_share"] >= 0.50 else "Growing Microbusiness"


def product_for_segment(segment: str) -> str:
    return {
        "Online Seller": "Online acquiring + payment links + QR payments",
        "Freelancer / Digital Professional": "Business card + international transfers",
        "Local Service Provider": "POS terminal + QR payments",
        "Growing Microbusiness": "Working capital loan + SME account",
        "Subscription / Recurring Business": "Recurring payment tools + invoicing",
    }.get(segment, "SME account + QR payments")


def score_band(score: float) -> str:
    if score >= 0.95:
        return "HIGH"
    if score >= 0.90:
        return "MEDIUM"
    return "LOW"


def migration_stage(score: float) -> str:
    if score >= 0.99:
        return "Priority SME Outreach"
    if score >= 0.95:
        return "SME Growth Candidate"
    if score >= 0.90:
        return "Microbusiness Lite"
    return "Consumer Nurture"


def estimate_revenue(row: pd.Series) -> pd.Series:
    monthly_turnover = float(row["total_turnover"]) / max(float(row["active_months"]), 1)
    confidence_multiplier = 0.55 + float(row["sme_score"]) / 200
    segment = str(row["segment"])
    config = {
        "Online Seller": (2500, 0.012, "1.2% acquiring margin + monthly service fee"),
        "Freelancer / Digital Professional": (2200, 0.004, "business card interchange + international transfer margin"),
        "Local Service Provider": (3000, 0.011, "POS/QR transaction margin + terminal/service fee"),
        "Growing Microbusiness": (5000, 0.018, "SME account fee + working capital interest spread"),
        "Subscription / Recurring Business": (2800, 0.009, "recurring payment fee + invoicing service margin"),
    }
    base, rate, assumption = config[segment]
    monthly = max((base + monthly_turnover * rate) * confidence_multiplier, 1000)
    return pd.Series({
        "illustrative_monthly_revenue_kzt": int(round(monthly)),
        "illustrative_annual_revenue_kzt": int(round(monthly * 12)),
        "revenue_scenario_assumption": assumption,
    })


def generate_explanation(row: pd.Series, benchmarks: dict[str, float]) -> pd.Series:
    candidates = [
        (
            ratio_signal(row["b2b_amount_share"], 0.50),
            "Business-infrastructure spending",
            f"{format_pct(row['b2b_amount_share'])} of turnover goes to B2B MCC categories such as ads, cloud, or suppliers.",
        ),
        (
            ratio_signal(row["merchant_concentration"], 0.40),
            "Supplier concentration pattern",
            f"Top merchant concentration is {format_pct(row['merchant_concentration'])}, similar to repeat supplier spending.",
        ),
        (
            ratio_signal(row["recurring_share"], 0.20),
            "Recurring operating expenses",
            f"{format_pct(row['recurring_share'])} of transactions are recurring payments.",
        ),
        (
            ratio_signal(row["online_share"], 0.70),
            "Digital activity footprint",
            f"{format_pct(row['online_share'])} of transactions happen online.",
        ),
        (
            ratio_signal(row["foreign_share"], 0.20),
            "International service usage",
            f"{format_pct(row['foreign_share'])} of transactions are foreign payments.",
        ),
        (
            ratio_signal(row["total_turnover"], benchmarks["total_turnover_p90"]),
            "High turnover pattern",
            f"Observed turnover is {format_kzt(row['total_turnover'])}.",
        ),
        (
            ratio_signal(max(float(row["monthly_growth"]), 0), 0.50),
            "Growth trajectory",
            f"Monthly turnover growth is {format_pct(row['monthly_growth'])}.",
        ),
    ]
    selected = sorted(candidates, key=lambda item: item[0], reverse=True)[:3]
    result: dict[str, Any] = {"dominant_signal": selected[0][1]}
    for index, (impact, reason, detail) in enumerate(selected, start=1):
        result[f"reason_{index}"] = reason
        result[f"reason_detail_{index}"] = detail
        result[f"impact_{index}"] = round(impact * 100)
    return pd.Series(result)


def ratio_signal(value: Any, target: float) -> float:
    return max(0.0, min(float(value) / max(float(target), 1e-9), 1.0))


def format_pct(value: Any) -> str:
    return f"{float(value) * 100:.0f}%"


def format_kzt(value: Any) -> str:
    return f"{float(value):,.0f} KZT"


def categorize_mcc(mcc: Any) -> str:
    code = int(mcc)
    mapping = {
        7311: "Digital Ads / Marketing", 7372: "Digital / IT Services",
        5968: "Subscription / Membership", 4816: "Digital / Information Services",
        7399: "Business Services", 5045: "Computer Equipment / Wholesale",
        4215: "Delivery / Logistics", 7379: "IT Services", 8931: "Accounting Services",
        5411: "Grocery / Local Retail", 5812: "Restaurants / Food Service",
        5814: "Fast Food / Food Service", 7230: "Beauty / Local Service",
        7299: "Personal Services", 7538: "Auto Service", 4121: "Taxi / Mobility",
    }
    if code in mapping:
        return mapping[code]
    if 5000 <= code <= 5599:
        return "Retail / Wholesale"
    if 7000 <= code <= 7299:
        return "Local Services"
    if 7300 <= code <= 7399:
        return "Business Services"
    if 8000 <= code <= 8999:
        return "Professional Services"
    return "Other"


def export_outputs(
    predictions: pd.DataFrame,
    features: pd.DataFrame,
    explanations: pd.DataFrame,
    metrics: dict[str, Any],
) -> None:
    predictions.to_parquet(
        DATA_DIR / "final_predictions.parquet", index=False, compression="snappy"
    )
    features.to_parquet(
        DATA_DIR / "client_features.parquet", index=False, compression="snappy"
    )
    explanations.to_parquet(
        DATA_DIR / "client_explanations.parquet", index=False, compression="snappy"
    )
    predictions.head(100).to_parquet(
        DATA_DIR / "top_leads.parquet", index=False, compression="snappy"
    )

    high = predictions[predictions["sme_rank_score"] >= 0.95]
    medium = predictions[
        (predictions["sme_rank_score"] >= 0.90) & (predictions["sme_rank_score"] < 0.95)
    ]
    dashboard = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_name": "Isolation Forest + One-Class SVM rank ensemble",
        "model_metrics": metrics,
        "score_note": "SME Score is a percentile-like business-likeness opportunity rank, not a probability.",
        "recommendation_note": "Segments and product suggestions are transparent demo rules, not ML predictions.",
        "revenue_note": "Revenue values are illustrative scenarios based on assumed margins, not observed bank income or validated tariffs.",
        "total_consumer_clients": int(len(predictions)),
        "high_potential_clients": int(len(high)),
        "medium_potential_clients": int(len(medium)),
        "average_sme_score": float(predictions["sme_score"].mean()),
        "average_sme_rank_score": float(predictions["sme_rank_score"].mean()),
        "illustrative_monthly_revenue_kzt_high": int(high["illustrative_monthly_revenue_kzt"].sum()),
        "illustrative_monthly_revenue_kzt_top_100": int(predictions.head(100)["illustrative_monthly_revenue_kzt"].sum()),
        "segments": predictions["segment"].value_counts().to_dict(),
        "score_bands": predictions["score_band"].value_counts().to_dict(),
        "demo_ids": predictions.head(10)["card_number"].astype(str).tolist(),
    }
    (DATA_DIR / "dashboard_metrics.json").write_text(
        json.dumps(dashboard, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (MODEL_DIR / "model_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def update_readme_demo_ids(demo_ids: list[str]) -> None:
    path = ROOT_DIR / "README.md"
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    start = "<!-- DEMO_IDS_START -->"
    end = "<!-- DEMO_IDS_END -->"
    rows = "\n".join(f"- `{card}`" for card in demo_ids[:8])
    replacement = f"{start}\n{rows}\n{end}"
    if start in text and end in text:
        text = text.split(start)[0] + replacement + text.split(end)[1]
    else:
        text += f"\n\n## Demo IDs\n\n{replacement}\n"
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
