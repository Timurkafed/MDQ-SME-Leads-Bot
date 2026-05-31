from __future__ import annotations

from typing import Any


PRODUCT_COPY = {
    "Online Seller": {
        "title": "Online acquiring + payment links + QR payments",
        "why": "High online activity suggests the client can accept digital payments directly.",
        "next_step": "Offer a one-click SME onboarding flow with payment links and QR acceptance.",
    },
    "Freelancer / Digital Professional": {
        "title": "Business card + international transfers",
        "why": "The pattern includes digital services, foreign transactions, or professional-service MCCs.",
        "next_step": "Offer a business card, FX-friendly transfers, and simple income tracking.",
    },
    "Local Service Provider": {
        "title": "POS terminal + QR payments",
        "why": "Offline and local-service behavior points to face-to-face payment acceptance.",
        "next_step": "Offer QR-first acceptance with POS terminal upgrade for growing volume.",
    },
    "Growing Microbusiness": {
        "title": "Working capital loan + SME account",
        "why": "Turnover, growth, and transaction frequency indicate a scaling microbusiness.",
        "next_step": "Offer SME account migration and a pre-approved working capital limit.",
    },
    "Subscription / Recurring Business": {
        "title": "Recurring payment tools + invoicing",
        "why": "Recurring payments indicate memberships, repeat services, or billing cycles.",
        "next_step": "Offer invoices, subscription charging, and automated payment reminders.",
    },
}


def product_offer(client: dict[str, Any]) -> dict[str, str]:
    segment = str(client.get("segment", "Growing Microbusiness"))
    return PRODUCT_COPY.get(segment, PRODUCT_COPY["Growing Microbusiness"])

