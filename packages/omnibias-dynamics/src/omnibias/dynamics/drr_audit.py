# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Audited inventory of the 121 quadratic DRR graphics.

The shipped :mod:`omnibias.dynamics.hilbert16_ledger` tracks the 15
remaining open/conditional cases explicitly. This module records the full
121-graphic corpus with primary-source status so ``drr_published_corpus``
cannot be earned while silently resting on unreplayed external theorems.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from omnibias.dynamics.hilbert16_ledger import DRR_CASES

GraphicStatus = Literal["published", "conditional", "preprint", "open", "ledger"]

__all__ = [
    "DRRGraphicRecord",
    "PUBLISHED_GRAPHIC_COUNT",
    "TOTAL_DRR_GRAPHIC_COUNT",
    "audit_report",
    "default_graphic_inventory",
    "ledger_case_names",
]


TOTAL_DRR_GRAPHIC_COUNT = 121
PUBLISHED_GRAPHIC_COUNT = TOTAL_DRR_GRAPHIC_COUNT - len(DRR_CASES)


@dataclass(frozen=True)
class DRRGraphicRecord:
    name: str
    status: GraphicStatus
    source: str
    in_ledger: bool

    def to_payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "status": self.status,
            "source": self.source,
            "in_ledger": self.in_ledger,
        }


def ledger_case_names() -> frozenset[str]:
    return frozenset(DRR_CASES)


def default_graphic_inventory() -> tuple[DRRGraphicRecord, ...]:
    """Return the dated 121-graphic inventory."""
    ledger = ledger_case_names()
    records: list[DRRGraphicRecord] = []
    for name in DRR_CASES:
        if name in {"I_12^1", "I_13^1", "I_14^1", "DF_1a", "DF_2a"}:
            status: GraphicStatus = "conditional"
            source = "published external theorem; not independently replayed here"
        elif name == "H_14^3":
            status = "preprint"
            source = "Lu arXiv:2607.13785v3 preprint claim"
        else:
            status = "open"
            source = "explicitly open or WIP in primary sources"
        records.append(DRRGraphicRecord(name, status, source, True))
    published_prefix = "published_quadratic_graphic"
    for index in range(1, PUBLISHED_GRAPHIC_COUNT + 1):
        records.append(
            DRRGraphicRecord(
                f"{published_prefix}_{index:03d}",
                "published",
                "Roussarie program corpus; not individually replayed in this repository",
                False,
            )
        )
    if len(records) != TOTAL_DRR_GRAPHIC_COUNT:
        raise RuntimeError("DRR graphic inventory count mismatch")
    return tuple(records)


def audit_report() -> dict[str, object]:
    inventory = default_graphic_inventory()
    by_status: dict[str, int] = {}
    for record in inventory:
        by_status[record.status] = by_status.get(record.status, 0) + 1
    return {
        "total": len(inventory),
        "ledger_cases": len(ledger_case_names()),
        "published_corpus": PUBLISHED_GRAPHIC_COUNT,
        "by_status": by_status,
        "records": [record.to_payload() for record in inventory],
        "drr_published_corpus_discharged": False,
        "scope": (
            "Inventory only; earning drr_published_corpus requires independent "
            "replay or discharge of every published graphic, not this catalog."
        ),
    }
