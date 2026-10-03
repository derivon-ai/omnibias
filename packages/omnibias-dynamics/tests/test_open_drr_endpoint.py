# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
from __future__ import annotations

from omnibias.dynamics.open_drr_endpoint import (
    attack_coalescing_chart,
    attack_open_endpoint,
    audit_drr_graphics,
    g1_chart_reaudit,
)


def test_g1_chart_reaudit_stays_route_specific() -> None:
    report = g1_chart_reaudit("I_2^1/I_4^1")
    assert report["parent_g1_passed"] is False
    assert report["route_specific"] is True


def test_drr_inventory_has_121_graphics() -> None:
    report = audit_drr_graphics()
    assert report["total"] == 121


def test_coalescing_chart_campaign_stays_blocked() -> None:
    report = attack_coalescing_chart()
    assert report["status"] == "blocked"


def test_open_endpoint_campaign_stays_blocked() -> None:
    report = attack_open_endpoint("DF_1b")
    assert report.status in {"open", "conditional", "blocked"}
    assert report.g1_route_specific_obstruction is True
