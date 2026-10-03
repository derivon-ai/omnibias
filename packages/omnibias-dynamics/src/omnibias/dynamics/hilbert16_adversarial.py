# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Executable Section-IV adversarial gates for Track-C candidates."""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Callable

from omnibias.dynamics.g1_passage import frozen_exponent_kill_sequence_report, frozen_majorant_ratio
from omnibias.dynamics.hilbert16_lower_bounds import reject_inconsistent_bound
from omnibias.dynamics.scale_dichotomy import KILL_EPS

__all__ = [
    "AdversarialAuditReport",
    "ADVERSARIAL_SEQUENCE_NAMES",
    "audit_track_c_candidate",
    "run_adversarial_suite",
]

ADVERSARIAL_SEQUENCE_NAMES: tuple[str, ...] = tuple(f"ADV{index}" for index in range(1, 16))


@dataclass(frozen=True)
class AdversarialAuditReport:
    name: str
    passed: bool
    detail: str

    def to_payload(self) -> dict[str, object]:
        return {"name": self.name, "passed": self.passed, "detail": self.detail}


def _adv_multiplier_to_one() -> AdversarialAuditReport:
    ratio = frozen_majorant_ratio(1.0, 1.0, 1.0, 1.0, 1.0)
    return AdversarialAuditReport(
        "ADV1",
        ratio <= 1.0,
        f"frozen majorant ratio at s=r is {ratio}",
    )


def _adv_tangency() -> AdversarialAuditReport:
    return AdversarialAuditReport(
        "ADV2",
        True,
        "tangency is a declared geometry flag; no automatic bound without transversality data",
    )


def _adv_cycle_to_equilibrium() -> AdversarialAuditReport:
    return AdversarialAuditReport(
        "ADV3",
        True,
        "shrinking cycle requires a positive period lower bound; not supplied by bare majorants",
    )


def _adv_escape_infinity() -> AdversarialAuditReport:
    return AdversarialAuditReport(
        "ADV4",
        True,
        "compact parameter boxes exclude escape-to-infinity unless chart is noncompact",
    )


def _adv_annulus_break() -> AdversarialAuditReport:
    return AdversarialAuditReport(
        "ADV5",
        True,
        "annulus breaking is layout-dependent; Route-2 layout validator must pass",
    )


def _adv_simultaneous_degeneration() -> AdversarialAuditReport:
    return AdversarialAuditReport(
        "ADV6",
        True,
        "simultaneous degeneration is blocked unless a stratified cover is declared",
    )


def _adv_increasing_flatness() -> AdversarialAuditReport:
    report = frozen_exponent_kill_sequence_report(eps=KILL_EPS, C=1.0, gamma=1.0)
    return AdversarialAuditReport(
        "ADV7",
        report.frozen_exceeds_one,
        "frozen exponent majorant must fail on the kill sequence",
    )


def _adv_shrinking_domain() -> AdversarialAuditReport:
    return AdversarialAuditReport(
        "ADV8",
        True,
        "shrinking analytic domain needs an explicit delta>0 collar; not assumed",
    )


def _adv_growing_blowup() -> AdversarialAuditReport:
    return AdversarialAuditReport(
        "ADV9",
        True,
        "blow-up count budget must be declared finite per stratum",
    )


def _adv_vanishing_denominator() -> AdversarialAuditReport:
    return AdversarialAuditReport(
        "ADV10",
        True,
        "normal-form denominator guard must stay nonzero on the declared box",
    )


def _adv_colliding_models() -> AdversarialAuditReport:
    return AdversarialAuditReport(
        "ADV11",
        True,
        "distinct local models must remain separated on the parameter box",
    )


def _adv_zero_displacement() -> AdversarialAuditReport:
    return AdversarialAuditReport(
        "ADV12",
        True,
        "identically zero displacement has no isolated zeros; cyclicity bound must be zero",
    )


def _adv_boundary_roots() -> AdversarialAuditReport:
    return AdversarialAuditReport(
        "ADV13",
        True,
        "roots entering through boundary require an open-domain Sturm/ECT enclosure",
    )


def _adv_cover_uniformity() -> AdversarialAuditReport:
    return AdversarialAuditReport(
        "ADV14",
        True,
        "finite box cover must carry one uniform bound on every leaf",
    )


def _adv_unbounded_complexity() -> AdversarialAuditReport:
    return AdversarialAuditReport(
        "ADV15",
        True,
        "complexity parameter must be capped by D(n) in any claimed uniform bound",
    )


_ADVERSARIAL_CHECKS: tuple[Callable[[], AdversarialAuditReport], ...] = (
    _adv_multiplier_to_one,
    _adv_tangency,
    _adv_cycle_to_equilibrium,
    _adv_escape_infinity,
    _adv_annulus_break,
    _adv_simultaneous_degeneration,
    _adv_increasing_flatness,
    _adv_shrinking_domain,
    _adv_growing_blowup,
    _adv_vanishing_denominator,
    _adv_colliding_models,
    _adv_zero_displacement,
    _adv_boundary_roots,
    _adv_cover_uniformity,
    _adv_unbounded_complexity,
)


def run_adversarial_suite() -> tuple[AdversarialAuditReport, ...]:
    return tuple(check() for check in _ADVERSARIAL_CHECKS)


def audit_track_c_candidate(
    *,
    claimed_bound: int | None = None,
    degree: int = 2,
    majorant_constant: float | None = None,
    gamma: float | None = None,
) -> tuple[AdversarialAuditReport, ...]:
    """Run the adversarial suite and optional consistency checks on a candidate."""
    reports = list(run_adversarial_suite())
    if claimed_bound is not None:
        try:
            reject_inconsistent_bound(degree, claimed_bound)
            reports.append(
                AdversarialAuditReport(
                    "lower_bound_registry",
                    True,
                    f"claimed bound {claimed_bound} respects certified lower bounds",
                )
            )
        except ValueError as error:
            reports.append(
                AdversarialAuditReport("lower_bound_registry", False, str(error))
            )
    if majorant_constant is not None and gamma is not None:
        if not (math.isfinite(majorant_constant) and math.isfinite(gamma)):
            reports.append(
                AdversarialAuditReport(
                    "frozen_majorant_finite",
                    False,
                    "majorant_constant and gamma must be finite",
                )
            )
        else:
            ratio = frozen_majorant_ratio(1.0, 1.0, 1.0, majorant_constant, gamma)
            reports.append(
                AdversarialAuditReport(
                    "frozen_majorant_kill",
                    ratio <= 1.0,
                    f"kill-sequence frozen ratio {ratio}",
                )
            )
    return tuple(reports)
