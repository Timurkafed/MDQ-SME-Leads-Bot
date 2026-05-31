from __future__ import annotations

from html import escape
from typing import Any


SEPARATOR = "━━━━━━━━━━━━━━━━━━"


def render_main_menu() -> str:
    return (
        "🏦 <b>AI SME Opportunity Engine</b>\n"
        f"{SEPARATOR}\n\n"
        "Rank hidden SME-like behavior inside consumer card portfolios and connect clients "
        "with useful banking products.\n\n"
        "💬 <b>Business principle</b>\n"
        "We do not punish hidden SMEs. We identify growing business behavior and help the bank serve it.\n\n"
        "Choose an action below or enter a client card number."
    )


def render_analyze_prompt(demo_ids: list[str]) -> str:
    ids = "\n".join(f"• <code>{escape(card)}</code>" for card in demo_ids[:5])
    return (
        "🔍 <b>Analyze Client</b>\n"
        f"{SEPARATOR}\n\n"
        "Enter a full card number to open a banking-style client card.\n\n"
        "Demo IDs:\n"
        f"{ids}"
    )


def render_client_card(client: dict[str, Any]) -> str:
    score = as_int(client.get("sme_score"))
    band_icon, band_label = score_badge(score, str(client.get("score_band", "")))
    card_number = escape(str(client.get("card_number", "")))
    segment = escape(str(client.get("segment", "Unknown")))
    migration_stage = escape(str(client.get("migration_stage", "Emerging SME Behavior")))
    product = escape(str(client.get("recommended_product", "SME account + QR payments")))

    return (
        f"🔍 <b>CLIENT #{card_number}</b>\n"
        f"{SEPARATOR}\n"
        f"SME Opportunity Score: <b>{score}/100</b> {band_icon} {band_label}\n"
        f"Segment: <b>{segment}</b>\n"
        f"Migration Stage: <b>{migration_stage}</b>\n\n"
        "📊 <b>Behavioral Profile</b>\n"
        f"• transactions: <code>{as_int(client.get('tx_count'))}</code>\n"
        f"• active days: <code>{as_int(client.get('active_days'))}</code>\n"
        f"• average transaction: <code>{fmt_kzt(client.get('avg_tx_amount'))}</code>\n"
        f"• online share: <code>{fmt_pct(client.get('online_share'))}</code>\n"
        f"• recurring share: <code>{fmt_pct(client.get('recurring_share'))}</code>\n"
        f"• B2B spending share: <code>{fmt_pct(client.get('b2b_amount_share'))}</code>\n"
        f"• MCC diversity: <code>{fmt_pct(client.get('mcc_diversity'))}</code>\n"
        f"• merchant pattern: <code>{as_int(client.get('unique_merchants'))} merchants / {escape(str(client.get('top_merchant_category', 'Other')))}</code>\n\n"
        "💡 <b>Why detected?</b>\n"
        f"1. {escape(str(client.get('reason_1', 'High SME similarity')))}\n"
        f"2. {escape(str(client.get('reason_2', 'Business-like transaction pattern')))}\n"
        f"3. {escape(str(client.get('reason_3', 'Portfolio opportunity signal')))}\n\n"
        "🎯 <b>Recommended Product</b>\n"
        f"{product}\n"
        "<i>Rule-based demo suggestion, not an ML prediction.</i>\n\n"
        "💰 <b>Illustrative Revenue Scenario</b>\n"
        f"Monthly scenario estimate: <code>{fmt_kzt(client.get('illustrative_monthly_revenue_kzt'))}</code>\n"
        "<i>Uses assumed margins; validate with actual bank tariffs.</i>"
    )


def render_top_leads(leads: list[dict[str, str]], page: int, total: int) -> str:
    if not leads:
        return (
            "🏆 <b>Top SME Leads</b>\n"
            f"{SEPARATOR}\n\n"
            "No leads found. Run <code>python train_pipeline.py</code> to refresh exports."
        )

    lines = []
    for index, lead in enumerate(leads, start=page * 10 + 1):
        card = escape(str(lead.get("card_number", "")))
        score = as_int(lead.get("sme_score"))
        segment = escape(str(lead.get("segment", "")))
        lines.append(f"{index}. <code>#{card}</code> - <b>{score}/100</b> - {segment}")

    return (
        "🏆 <b>Top SME Leads</b>\n"
        f"{SEPARATOR}\n"
        f"Showing {page * 10 + 1}-{page * 10 + len(leads)} of {total}\n\n"
        + "\n".join(lines)
        + "\n\nTap a lead button below to open the client card."
    )


def render_dashboard(data: dict[str, Any]) -> str:
    metrics = data.get("model_metrics", {}) or {}
    segments = data.get("segments", {}) or {}
    top_segments = sorted(segments.items(), key=lambda item: item[1], reverse=True)[:5]
    segment_lines = "\n".join(
        f"• {escape(str(name))}: <code>{count}</code>" for name, count in top_segments
    )

    return (
        "📊 <b>Portfolio Dashboard</b>\n"
        f"{SEPARATOR}\n\n"
        "🏦 <b>Portfolio</b>\n"
        f"• consumer clients scored: <code>{as_int(data.get('total_consumer_clients'))}</code>\n"
        f"• high-potential clients: <code>{as_int(data.get('high_potential_clients'))}</code>\n"
        f"• medium-potential clients: <code>{as_int(data.get('medium_potential_clients'))}</code>\n"
        f"• average SME score: <code>{as_float(data.get('average_sme_score')):.1f}/100</code>\n\n"
        "💰 <b>Illustrative Revenue Scenario</b>\n"
        f"• high-potential monthly scenario: <code>{fmt_kzt(data.get('illustrative_monthly_revenue_kzt_high'))}</code>\n"
        f"• top-100 monthly scenario: <code>{fmt_kzt(data.get('illustrative_monthly_revenue_kzt_top_100'))}</code>\n"
        "• <i>assumed margins, not observed bank income</i>\n\n"
        "🤖 <b>Model Quality</b>\n"
        f"• model: <code>{escape(str(data.get('model_name', 'Unknown')))}</code>\n"
        f"• ROC-AUC: <code>{fmt_metric(metrics.get('roc_auc'))}</code>\n"
        f"• precision: <code>{fmt_metric(metrics.get('precision'))}</code>\n"
        f"• recall: <code>{fmt_metric(metrics.get('recall'))}</code>\n"
        f"• F1-score: <code>{fmt_metric(metrics.get('f1_score'))}</code>\n\n"
        "ℹ️ <i>Quality metrics are proxy evaluation: unseen business cards vs consumer pool.</i>\n"
        "ℹ️ <i>SME Score is an opportunity rank, not a calibrated probability.</i>\n\n"
        "🧭 <b>Segments</b>\n"
        f"{segment_lines or 'No segment data yet.'}"
    )


def render_how_it_works() -> str:
    return (
        "ℹ️ <b>How It Works</b>\n"
        f"{SEPARATOR}\n\n"
        "1. Merge business and consumer card transactions with merchant reference data.\n"
        "2. Aggregate behavior by card: turnover, activity, B2B spending, recurrence, merchant concentration, growth.\n"
        "3. Train a one-class ensemble only on known business cards.\n"
        "4. Rank consumer cards by similarity to business behavior.\n"
        "5. Add transparent demo rules for behavioral explanation, segment, product suggestion, and an illustrative revenue scenario.\n\n"
        "✅ <b>Important</b>\n"
        "The bot does not train live during the demo. It reads precomputed Parquet/JSON outputs from the ML pipeline.\n"
        "The score is a ranked opportunity signal, not proof that a client is an entrepreneur."
    )


def render_why_detected(client: dict[str, Any], reasons: list[dict[str, Any]]) -> str:
    card = escape(str(client.get("card_number", "")))
    lines = []
    for index, row in enumerate(reasons, start=1):
        impact = as_int(row.get("impact"))
        lines.append(
            f"{index}. <b>{escape(str(row.get('reason', 'Signal')))}</b>\n"
            f"   {progress_bar(impact / 100)} <code>{impact}%</code>\n"
            f"   <i>{escape(str(row.get('detail', '')))}</i>"
        )

    return (
        f"🔬 <b>Why Detected - #{card}</b>\n"
        f"{SEPARATOR}\n\n"
        "Rule-based explanation based on strongest behavioral signals:\n\n"
        + "\n\n".join(lines)
        + "\n\nThis is an opportunity signal, not a penalty flag."
    )


def render_product_offer(client: dict[str, Any], offer: dict[str, str]) -> str:
    card = escape(str(client.get("card_number", "")))
    return (
        f"💳 <b>Rule-Based Product Suggestion - #{card}</b>\n"
        f"{SEPARATOR}\n\n"
        f"Segment: <b>{escape(str(client.get('segment', 'Unknown')))}</b>\n\n"
        f"🎯 <b>{escape(offer['title'])}</b>\n"
        f"{escape(offer['why'])}\n\n"
        "🚀 <b>Next Best Action</b>\n"
        f"{escape(offer['next_step'])}\n\n"
        f"Opportunity rank: <code>{as_int(client.get('sme_score'))}/100</code>\n"
        "<i>This suggestion is a transparent demo rule, not an ML prediction.</i>"
    )


def render_revenue_potential(client: dict[str, Any]) -> str:
    card = escape(str(client.get("card_number", "")))
    monthly = fmt_kzt(client.get("illustrative_monthly_revenue_kzt"))
    annual = fmt_kzt(client.get("illustrative_annual_revenue_kzt"))
    turnover = fmt_kzt(client.get("total_turnover"))
    assumption = escape(str(client.get("revenue_scenario_assumption", "Illustrative product margin assumption")))

    return (
        f"💰 <b>Illustrative Revenue Scenario - #{card}</b>\n"
        f"{SEPARATOR}\n\n"
        f"Monthly scenario: <b>{monthly}</b>\n"
        f"Annualized scenario: <b>{annual}</b>\n\n"
        "📌 <b>Basis</b>\n"
        f"• observed turnover: <code>{turnover}</code>\n"
        f"• product: <code>{escape(str(client.get('recommended_product', 'SME product')))}</code>\n"
        f"• assumption: <i>{assumption}</i>\n\n"
        "⚠️ <i>This is not observed bank income. Validate the assumptions with actual bank tariffs before external use.</i>"
    )


def render_behavior_chart(client: dict[str, Any], chart_rows: list[dict[str, Any]]) -> str:
    card = escape(str(client.get("card_number", "")))
    rows = []
    for row in chart_rows:
        value = as_float(row.get("value"))
        rows.append(
            f"{escape(str(row.get('label'))):<16} {progress_bar(value)} <code>{fmt_pct(value)}</code>"
        )

    return (
        f"📊 <b>Behavior Chart - #{card}</b>\n"
        f"{SEPARATOR}\n\n"
        "<pre>"
        + "\n".join(strip_html(row) for row in rows)
        + "</pre>\n\n"
        f"Top category: <b>{escape(str(client.get('top_merchant_category', 'Other')))}</b>\n"
        f"Top MCC: <code>{escape(str(client.get('top_mcc', 'Unknown')))}</code>"
    )


def render_not_found(query: str, suggestions: list[str]) -> str:
    suggestion_lines = "\n".join(f"• <code>{escape(card)}</code>" for card in suggestions[:5])
    return (
        "⚠️ <b>Client Not Found</b>\n"
        f"{SEPARATOR}\n\n"
        f"No exported client matches <code>{escape(query)}</code>.\n\n"
        "Try one of these demo IDs:\n"
        f"{suggestion_lines}"
    )


def score_badge(score: int, band: str = "") -> tuple[str, str]:
    normalized = band.upper()
    if normalized == "HIGH" or (not normalized and score >= 95):
        return "🔴", "HIGH"
    if normalized == "MEDIUM" or (not normalized and score >= 90):
        return "🟡", "MEDIUM"
    return "🟢", "LOW"


def progress_bar(value: float, width: int = 10) -> str:
    value = max(0.0, min(float(value), 1.0))
    filled = round(value * width)
    return "█" * filled + "░" * (width - filled)


def strip_html(value: str) -> str:
    return value.replace("<code>", "").replace("</code>", "")


def fmt_kzt(value: Any) -> str:
    return f"{as_float(value):,.0f} KZT"


def fmt_pct(value: Any) -> str:
    return f"{as_float(value) * 100:.0f}%"


def fmt_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{as_float(value):.3f}"


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: Any, default: int = 0) -> int:
    return int(round(as_float(value, default)))
