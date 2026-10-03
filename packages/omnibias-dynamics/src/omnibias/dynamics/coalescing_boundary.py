# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Coalescing-root boundary atlas for I_2^1 / I_4^1 saddle-node-at-infinity graphics."""

from __future__ import annotations

from dataclasses import dataclass
from math import log

from omnibias.core.proof.certificate import Cert, make_certificate, verify_certificate_digest
from omnibias.core.proof.realization_replay import source_digest
from omnibias.dynamics.chart_cells import chart_dependent_g1_reaudit
from omnibias.dynamics.g1_passage import frozen_exponent_kill_sequence_report

__all__ = [
    "CoalescingBoundaryReport",
    "attack_coalescing_boundary",
    "chi_blowup_coordinates",
    "epsilon_log_epsilon_enclosure",
]


def chi_blowup_coordinates(
    s: float,
    r: float,
    kappa: float,
) -> tuple[float, float]:
    """Return ``(chi, kappa)`` with ``chi = (s/r) * kappa`` on a declared collar."""
    if r == 0.0:
        raise ValueError("r must be nonzero")
    return (s / r) * kappa, kappa


def epsilon_log_epsilon_enclosure(epsilon: float) -> tuple[float, float]:
    """Outward enclosure of ``epsilon * log(epsilon^-1)`` for small ``epsilon > 0``."""
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    value = epsilon * log(1.0 / epsilon)
    width = max(abs(value) * 1e-12, 1e-15)
    return value - width, value + width


@dataclass(frozen=True)
class CoalescingBoundaryReport:
    chart: str
    status: str
    chi_atlas_declared: bool
    smooth_dependence_recorded: bool
    g1_obstruction_replayed: bool
    detail: str
    source_digest: str
    seal: Cert

    def to_payload(self) -> dict[str, object]:
        return {
            "chart": self.chart,
            "status": self.status,
            "chi_atlas_declared": self.chi_atlas_declared,
            "smooth_dependence_recorded": self.smooth_dependence_recorded,
            "g1_obstruction_replayed": self.g1_obstruction_replayed,
            "detail": self.detail,
        }


def attack_coalescing_boundary(chart: str = "I_2^1/I_4^1") -> CoalescingBoundaryReport:
    """Record the current blocked coalescing-boundary campaign for a named chart."""
    g1 = frozen_exponent_kill_sequence_report()
    chart_audit = chart_dependent_g1_reaudit(chart)
    eps_lo, eps_hi = epsilon_log_epsilon_enclosure(1e-3)
    payload = {
        "type": "coalescing_boundary_campaign",
        "chart": chart,
        "chi_blowup": "chi=(s/r)*kappa",
        "epsilon_log_epsilon_enclosure": [eps_lo, eps_hi],
        "g1_frozen_exceeds_one": g1.frozen_exceeds_one,
        "chart_dependent_reaudit": chart_audit,
        "status": "blocked",
    }
    digest = source_digest(payload)
    seal = make_certificate(
        claim="Coalescing-root boundary atlas campaign (I_2^1/I_4^1); analytic closure open.",
        payload=payload,
        honesty={
            "coalescing_boundary_closed": False,
            "physical_return_membership_proved": False,
            "graphic_finite_cyclicity_proved": False,
            "full_hilbert16_solved": False,
        },
        meta={"transcend_backend": "not_used"},
    )
    return CoalescingBoundaryReport(
        chart,
        "blocked",
        True,
        True,
        g1.frozen_exceeds_one,
        (
            "chi blow-up atlas and (epsilon, epsilon log epsilon^-1) dependence are "
            "declared; uniform endpoint matching and variational remainders remain open"
        ),
        digest,
        seal,
    )


def verify_coalescing_boundary_report(report: CoalescingBoundaryReport) -> bool:
    if not verify_certificate_digest(report.seal):
        return False
    payload = report.seal["payload"]
    return payload.get("type") == "coalescing_boundary_campaign" and source_digest(payload) == report.source_digest
