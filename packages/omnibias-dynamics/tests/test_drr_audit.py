# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
from __future__ import annotations

from omnibias.dynamics.drr_audit import (
    TOTAL_DRR_GRAPHIC_COUNT,
    audit_report,
    default_graphic_inventory,
    ledger_case_names,
)
from omnibias.dynamics.hilbert16_ledger import DRR_CASES


def test_inventory_has_121_graphics_with_15_ledger_cases() -> None:
    inventory = default_graphic_inventory()
    assert len(inventory) == TOTAL_DRR_GRAPHIC_COUNT
    assert len(ledger_case_names()) == len(DRR_CASES)
    assert sum(1 for record in inventory if record.in_ledger) == len(DRR_CASES)


def test_audit_report_is_honest_about_corpus_discharge() -> None:
    report = audit_report()
    assert report["total"] == TOTAL_DRR_GRAPHIC_COUNT
    assert report["drr_published_corpus_discharged"] is False
