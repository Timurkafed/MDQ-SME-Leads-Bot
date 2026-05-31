from __future__ import annotations

from aiogram import F, Router
from aiogram.enums import ChatAction
from aiogram.filters import Command
from aiogram.types import Message

from bot.services.scoring_service import get_scoring_service
from bot.ui.keyboards import BTN_ANALYZE, client_keyboard, main_inline_menu
from bot.ui.renderers import render_analyze_prompt, render_client_card, render_not_found


router = Router(name="analyze")


@router.message(Command("analyze"))
@router.message(F.text == BTN_ANALYZE)
async def analyze_prompt(message: Message) -> None:
    service = get_scoring_service()
    await message.answer(
        render_analyze_prompt(service.demo_ids()),
        reply_markup=main_inline_menu(),
    )


@router.message(F.text.regexp(r"^\s*\d{4,19}\s*$"))
async def analyze_card(message: Message) -> None:
    service = get_scoring_service()
    query = (message.text or "").strip()

    await message.bot.send_chat_action(chat_id=message.chat.id, action=ChatAction.TYPING)
    loading = await message.answer("🔎 <b>Analyzing client behavior...</b>")

    client = service.get_client(query)
    if client is None:
        await loading.edit_text(
            render_not_found(query, service.suggest_ids(query)),
            reply_markup=main_inline_menu(),
        )
        return

    await loading.edit_text(
        render_client_card(client),
        reply_markup=client_keyboard(client["card_number"]),
    )

