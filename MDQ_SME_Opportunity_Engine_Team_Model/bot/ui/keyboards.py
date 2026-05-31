from __future__ import annotations

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup, KeyboardButton, ReplyKeyboardMarkup


BTN_ANALYZE = "🔍 Analyze Client"
BTN_TOP = "🏆 Top SME Leads"
BTN_DASHBOARD = "📊 Dashboard"
BTN_HOW = "ℹ️ How It Works"


def main_reply_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text=BTN_ANALYZE), KeyboardButton(text=BTN_TOP)],
            [KeyboardButton(text=BTN_DASHBOARD), KeyboardButton(text=BTN_HOW)],
        ],
        resize_keyboard=True,
        is_persistent=True,
        input_field_placeholder="Enter client card number or choose an action...",
    )


def main_inline_menu() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text=BTN_ANALYZE, callback_data="analyze"),
                InlineKeyboardButton(text=BTN_TOP, callback_data="top:0"),
            ],
            [
                InlineKeyboardButton(text=BTN_DASHBOARD, callback_data="dashboard"),
                InlineKeyboardButton(text=BTN_HOW, callback_data="how"),
            ],
        ]
    )


def client_keyboard(card_number: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="📊 Behavior Chart", callback_data=f"chart:{card_number}"),
                InlineKeyboardButton(text="🔬 Why Detected", callback_data=f"why:{card_number}"),
            ],
            [
                InlineKeyboardButton(text="💳 Product Offer", callback_data=f"product:{card_number}"),
                InlineKeyboardButton(text="💰 Revenue Potential", callback_data=f"revenue:{card_number}"),
            ],
            [
                InlineKeyboardButton(text="◀️ Back", callback_data="top:0"),
                InlineKeyboardButton(text="🏠 Menu", callback_data="menu"),
            ],
        ]
    )


def top_leads_keyboard(
    leads: list[dict[str, str]],
    page: int,
    total: int,
    page_size: int = 10,
) -> InlineKeyboardMarkup:
    rows: list[list[InlineKeyboardButton]] = []
    for index, lead in enumerate(leads, start=page * page_size + 1):
        card = lead["card_number"]
        score = lead.get("sme_score", "0")
        segment = str(lead.get("segment", ""))[:18]
        rows.append(
            [
                InlineKeyboardButton(
                    text=f"{index}. #{card[-6:]} · {score}/100 · {segment}",
                    callback_data=f"client:{card}",
                )
            ]
        )

    nav: list[InlineKeyboardButton] = []
    if page > 0:
        nav.append(InlineKeyboardButton(text="◀️ Prev", callback_data=f"top:{page - 1}"))
    if (page + 1) * page_size < total:
        nav.append(InlineKeyboardButton(text="Next ▶️", callback_data=f"top:{page + 1}"))
    if nav:
        rows.append(nav)

    rows.append([InlineKeyboardButton(text="🏠 Menu", callback_data="menu")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def dashboard_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="🏆 Top SME Leads", callback_data="top:0"),
                InlineKeyboardButton(text="🔍 Analyze Client", callback_data="analyze"),
            ],
            [InlineKeyboardButton(text="🏠 Menu", callback_data="menu")],
        ]
    )
