from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"


@dataclass(frozen=True)
class BotConfig:
    bot_token: str
    data_dir: Path


def load_config() -> BotConfig:
    load_dotenv(ROOT_DIR / ".env")
    token = os.getenv("BOT_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "BOT_TOKEN is missing. Create .env from .env.example and paste your Telegram bot token."
        )

    data_dir = Path(os.getenv("DATA_DIR", DATA_DIR)).resolve()
    return BotConfig(bot_token=token, data_dir=data_dir)

