from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT_DIR / "data"
PAGE_SIZE = 10


class ScoringService:
    def __init__(self, data_dir: Path | None = None) -> None:
        self.data_dir = Path(data_dir or os.getenv("DATA_DIR", DEFAULT_DATA_DIR)).resolve()
        self.predictions = self._read_parquet("final_predictions.parquet")
        self.features = self._read_parquet("client_features.parquet")
        self.explanations = self._read_parquet("client_explanations.parquet")
        self.top = self._read_parquet("top_leads.parquet")
        self.dashboard_data = self._read_json("dashboard_metrics.json")

        self.predictions_by_card = {row["card_number"]: row for row in self.predictions}
        self.features_by_card = {row["card_number"]: row for row in self.features}
        self.explanations_by_card = {row["card_number"]: row for row in self.explanations}

    def _read_parquet(self, filename: str) -> list[dict[str, Any]]:
        path = self.data_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run `python train_pipeline.py` before starting the bot."
            )
        frame = pd.read_parquet(path)
        if "card_number" in frame.columns:
            frame["card_number"] = frame["card_number"].astype(str)
        return frame.to_dict(orient="records")

    def _read_json(self, filename: str) -> dict[str, Any]:
        path = self.data_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run `python train_pipeline.py` before starting the bot."
            )
        return json.loads(path.read_text(encoding="utf-8"))

    def get_client(self, card_number: str) -> dict[str, Any] | None:
        normalized = normalize_card(card_number)
        exact = self.predictions_by_card.get(normalized)
        if exact is None:
            matches = [card for card in self.predictions_by_card if card.endswith(normalized)]
            if len(matches) == 1:
                exact = self.predictions_by_card[matches[0]]
            else:
                return None

        card = exact["card_number"]
        client: dict[str, Any] = {}
        client.update(self.features_by_card.get(card, {}))
        client.update(exact)
        client.update(self.explanations_by_card.get(card, {}))
        client["card_number"] = card
        return client

    def suggest_ids(self, query: str, limit: int = 5) -> list[str]:
        normalized = normalize_card(query)
        matches = [
            card
            for card in self.predictions_by_card
            if card.endswith(normalized) or normalized in card
        ]
        if matches:
            return matches[:limit]
        return self.demo_ids(limit)

    def demo_ids(self, limit: int = 5) -> list[str]:
        ids = self.dashboard_data.get("demo_ids") or []
        if ids:
            return [str(item) for item in ids[:limit]]
        return [row["card_number"] for row in self.top[:limit]]

    def top_leads(self, page: int = 0, page_size: int = PAGE_SIZE) -> list[dict[str, Any]]:
        start = max(page, 0) * page_size
        end = start + page_size
        return self.predictions[start:end]

    def top_count(self) -> int:
        return len(self.predictions)

    def dashboard(self) -> dict[str, Any]:
        return self.dashboard_data


def normalize_card(card_number: str) -> str:
    return "".join(ch for ch in str(card_number).strip() if ch.isdigit())


@lru_cache(maxsize=1)
def get_scoring_service() -> ScoringService:
    return ScoringService()
