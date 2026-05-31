from __future__ import annotations

from typing import Any


CHART_METRICS = [
    ("B2B spend share", "b2b_amount_share"),
    ("Online share", "online_share"),
    ("Recurring share", "recurring_share"),
    ("Foreign share", "foreign_tx_share"),
    ("Merchant focus", "merchant_concentration"),
    ("Business hours", "business_hours_share"),
    ("Weekend share", "weekend_share"),
]


def build_behavior_chart(client: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for label, key in CHART_METRICS:
        rows.append({"label": label, "value": to_float(client.get(key, 0))})
    return rows


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
