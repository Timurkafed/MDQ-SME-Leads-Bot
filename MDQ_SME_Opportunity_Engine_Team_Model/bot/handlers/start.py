from __future__ import annotations

from aiogram import F, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import Message

from bot.ui.keyboards import BTN_HOW, main_inline_menu, main_reply_keyboard
from bot.ui.renderers import render_how_it_works, render_main_menu


router = Router(name="start")


@router.message(CommandStart())
async def start_screen(message: Message) -> None:
    await message.answer(
        render_main_menu(),
        reply_markup=main_reply_keyboard(),
    )
    await message.answer(
        "Choose a workspace action:",
        reply_markup=main_inline_menu(),
    )


@router.message(Command("help"))
@router.message(F.text == BTN_HOW)
async def how_it_works_screen(message: Message) -> None:
    await message.answer(
        render_how_it_works(),
        reply_markup=main_inline_menu(),
    )

