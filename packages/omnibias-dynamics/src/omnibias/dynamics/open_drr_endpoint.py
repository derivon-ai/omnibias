# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Open DRR endpoint campaigns and chart-dependent G1 re-audits."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

from omnibias.dynamics.chart_cells import chart_dependent_g1_reaudit
from omnibias.dynamics.coalescing_boundary import attack_coalescing_boundary
from omnibias.dynamics.drr_audit import audit_report, default_graphic_inventory
from omnibias.dynamics.g1_passage import frozen_exponent_kill_sequence_report
from omnibias.dynamics.slow_fast import SlowFastGraphic, slow_divergence_integral

EndpointName = Literal["DF_1b", "DF_2b", "DH_1", "DH_2"]

__all__ = [
    "EndpointCampaignReport",
    "OPEN_ENDPOINTS",
    "attack_coalescing_chart",
    "attack_open_endpoint",
    "audit_drr_graphics",
    "g1_chart_reaudit",
]


OPEN_ENDPOINTS: tuple[EndpointName, ...] = ("DF_1b", "DF_2b", "DH_1", "DH_2")


@dataclass(frozen=True)
class EndpointCampaignReport:
    name: EndpointName
    status: Literal["blocked", "conditional", "open"]
    detail: str
    g1_route_specific_obstruction: bool
    slow_divergence_certified: bool

    def to_payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
            "g1_route_specific_obstruction": self.g1_route_specific_obstruction,
            "slow_divergence_certified": self.slow_divergence_certified,
        }


def audit_drr_graphics() -> dict[str, object]:
    """Return the dated 121-graphic inventory audit."""
    return audit_report()


def attack_coalescing_chart(chart: str = "I_2^1/I_4^1") -> dict[str, object]:
    """Launch the coalescing-boundary campaign on a named chart."""
    report = attack_coalescing_boundary(chart)
    return report.to_payload()


def g1_chart_reaudit(chart: str = "I_2^1/I_4^1") -> dict[str, object]:
    """Re-audit frozen and chart-dependent majorants on a named chart."""
    return chart_dependent_g1_reaudit(chart)


def attack_open_endpoint(name: EndpointName) -> EndpointCampaignReport:
    """Record the current blocked status for one open DF/DH sibling."""
    inventory = {record.name: record for record in default_graphic_inventory()}
    record = inventory.get(name)
    if record is None:
        raise ValueError(f"unknown endpoint {name!r}")
    graphic = SlowFastGraphic(name, slow_parameter_count=2, fast_parameter_count=1)
    slow = slow_divergence_integral(
        graphic,
        lambda h: Fraction(1, 1) / h,
        height_lo=Fraction(1, 100),
        height_hi=Fraction(1),
    )
    obstruction = frozen_exponent_kill_sequence_report()
    status: Literal["blocked", "conditional", "open"] = (
        "open" if record.status == "open" else "conditional" if record.status in {"conditional", "preprint"} else "blocked"
    )
    return EndpointCampaignReport(
        name,
        status,
        "entry-exit balance and boundary slow-divergence sign remain open",
        obstruction.frozen_exceeds_one,
        slow.value_enclosure[0] <= slow.value_enclosure[1],
    )
