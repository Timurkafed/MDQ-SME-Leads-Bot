from __future__ import annotations

from aiogram import F, Router
from aiogram.filters import Command
from aiogram.types import Message

from bot.services.scoring_service import get_scoring_service
from bot.ui.keyboards import BTN_DASHBOARD, dashboard_keyboard
from bot.ui.renderers import render_dashboard


router = Router(name="dashboard")


@router.message(Command("dashboard"))
@router.message(F.text == BTN_DASHBOARD)
async def dashboard_screen(message: Message) -> None:
    service = get_scoring_service()
    await message.answer(
        render_dashboard(service.dashboard()),
        reply_markup=dashboard_keyboard(),
    )

