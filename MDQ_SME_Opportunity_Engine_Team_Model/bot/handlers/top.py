from __future__ import annotations

from aiogram import F, Router
from aiogram.filters import Command
from aiogram.types import Message

from bot.services.scoring_service import get_scoring_service
from bot.ui.keyboards import BTN_TOP, top_leads_keyboard
from bot.ui.renderers import render_top_leads


router = Router(name="top")


@router.message(Command("top"))
@router.message(F.text == BTN_TOP)
async def top_leads_screen(message: Message) -> None:
    service = get_scoring_service()
    page = 0
    leads = service.top_leads(page=page)
    await message.answer(
        render_top_leads(leads, page=page, total=service.top_count()),
        reply_markup=top_leads_keyboard(leads, page=page, total=service.top_count()),
    )

