from __future__ import annotations

from aiogram import F, Router
from aiogram.types import CallbackQuery

from bot.services.chart_service import build_behavior_chart
from bot.services.explanation_service import get_reason_rows
from bot.services.recommendation_engine import product_offer
from bot.services.scoring_service import get_scoring_service
from bot.ui.keyboards import (
    client_keyboard,
    dashboard_keyboard,
    main_inline_menu,
    top_leads_keyboard,
)
from bot.ui.renderers import (
    render_analyze_prompt,
    render_behavior_chart,
    render_client_card,
    render_dashboard,
    render_how_it_works,
    render_main_menu,
    render_not_found,
    render_product_offer,
    render_revenue_potential,
    render_top_leads,
    render_why_detected,
)


router = Router(name="callbacks")


@router.callback_query(F.data == "menu")
async def menu_callback(callback: CallbackQuery) -> None:
    await callback.answer()
    await callback.message.edit_text(render_main_menu(), reply_markup=main_inline_menu())


@router.callback_query(F.data == "analyze")
async def analyze_callback(callback: CallbackQuery) -> None:
    await callback.answer("Open client analyzer")
    service = get_scoring_service()
    await callback.message.edit_text(
        render_analyze_prompt(service.demo_ids()),
        reply_markup=main_inline_menu(),
    )


@router.callback_query(F.data == "how")
async def how_callback(callback: CallbackQuery) -> None:
    await callback.answer()
    await callback.message.edit_text(render_how_it_works(), reply_markup=main_inline_menu())


@router.callback_query(F.data == "dashboard")
async def dashboard_callback(callback: CallbackQuery) -> None:
    await callback.answer("Refreshing dashboard")
    service = get_scoring_service()
    await callback.message.edit_text(
        render_dashboard(service.dashboard()),
        reply_markup=dashboard_keyboard(),
    )


@router.callback_query(F.data.startswith("top:"))
async def top_callback(callback: CallbackQuery) -> None:
    service = get_scoring_service()
    page = int((callback.data or "top:0").split(":", maxsplit=1)[1])
    page = max(page, 0)
    leads = service.top_leads(page=page)
    await callback.answer("Loading top leads")
    await callback.message.edit_text(
        render_top_leads(leads, page=page, total=service.top_count()),
        reply_markup=top_leads_keyboard(leads, page=page, total=service.top_count()),
    )


@router.callback_query(F.data.startswith("client:"))
async def client_callback(callback: CallbackQuery) -> None:
    card_number = (callback.data or "").split(":", maxsplit=1)[1]
    service = get_scoring_service()
    client = service.get_client(card_number)
    await callback.answer("Opening client card")
    if client is None:
        await callback.message.edit_text(
            render_not_found(card_number, service.demo_ids()),
            reply_markup=main_inline_menu(),
        )
        return
    await callback.message.edit_text(
        render_client_card(client),
        reply_markup=client_keyboard(client["card_number"]),
    )


@router.callback_query(F.data.startswith("why:"))
async def why_callback(callback: CallbackQuery) -> None:
    card_number = (callback.data or "").split(":", maxsplit=1)[1]
    service = get_scoring_service()
    client = service.get_client(card_number)
    await callback.answer("Explaining detection")
    if client is None:
        await callback.message.edit_text(render_not_found(card_number, service.demo_ids()), reply_markup=main_inline_menu())
        return
    await callback.message.edit_text(
        render_why_detected(client, get_reason_rows(client)),
        reply_markup=client_keyboard(client["card_number"]),
    )


@router.callback_query(F.data.startswith("product:"))
async def product_callback(callback: CallbackQuery) -> None:
    card_number = (callback.data or "").split(":", maxsplit=1)[1]
    service = get_scoring_service()
    client = service.get_client(card_number)
    await callback.answer("Preparing product offer")
    if client is None:
        await callback.message.edit_text(render_not_found(card_number, service.demo_ids()), reply_markup=main_inline_menu())
        return
    await callback.message.edit_text(
        render_product_offer(client, product_offer(client)),
        reply_markup=client_keyboard(client["card_number"]),
    )


@router.callback_query(F.data.startswith("revenue:"))
async def revenue_callback(callback: CallbackQuery) -> None:
    card_number = (callback.data or "").split(":", maxsplit=1)[1]
    service = get_scoring_service()
    client = service.get_client(card_number)
    await callback.answer("Calculating bank value")
    if client is None:
        await callback.message.edit_text(render_not_found(card_number, service.demo_ids()), reply_markup=main_inline_menu())
        return
    await callback.message.edit_text(
        render_revenue_potential(client),
        reply_markup=client_keyboard(client["card_number"]),
    )


@router.callback_query(F.data.startswith("chart:"))
async def chart_callback(callback: CallbackQuery) -> None:
    card_number = (callback.data or "").split(":", maxsplit=1)[1]
    service = get_scoring_service()
    client = service.get_client(card_number)
    await callback.answer("Building behavior chart")
    if client is None:
        await callback.message.edit_text(render_not_found(card_number, service.demo_ids()), reply_markup=main_inline_menu())
        return
    await callback.message.edit_text(
        render_behavior_chart(client, build_behavior_chart(client)),
        reply_markup=client_keyboard(client["card_number"]),
    )
