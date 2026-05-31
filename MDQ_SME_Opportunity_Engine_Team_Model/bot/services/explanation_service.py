from __future__ import annotations

from typing import Any


def get_reason_rows(client: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for index in range(1, 4):
        rows.append(
            {
                "reason": client.get(f"reason_{index}", "Portfolio similarity"),
                "detail": client.get(
                    f"reason_detail_{index}",
                    "Behavior is close to known SME card patterns.",
                ),
                "impact": client.get(f"impact_{index}", "0"),
            }
        )
    return rows

