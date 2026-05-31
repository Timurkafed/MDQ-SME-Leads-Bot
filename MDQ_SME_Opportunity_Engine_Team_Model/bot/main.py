from __future__ import annotations

import asyncio
import logging

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.types import BotCommand

from bot.config import load_config
from bot.handlers import analyze, callbacks, dashboard, start, top


async def setup_bot_commands(bot: Bot) -> None:
    commands = [
        BotCommand(command="start", description="🏠 Main menu"),
        BotCommand(command="analyze", description="🔍 Analyze client by card ID"),
        BotCommand(command="top", description="🏆 Top SME leads"),
        BotCommand(command="dashboard", description="📊 Portfolio dashboard"),
        BotCommand(command="help", description="ℹ️ How it works"),
    ]
    await bot.set_my_commands(commands)
    await bot.set_my_description(
        "AI assistant for discovering hidden SME-like behavior in consumer card portfolios."
    )
    await bot.set_my_short_description("AI SME Opportunity Engine")


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config()

    bot = Bot(
        token=config.bot_token,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    dp = Dispatcher()
    dp.include_router(start.router)
    dp.include_router(top.router)
    dp.include_router(dashboard.router)
    dp.include_router(analyze.router)
    dp.include_router(callbacks.router)

    await setup_bot_commands(bot)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())

