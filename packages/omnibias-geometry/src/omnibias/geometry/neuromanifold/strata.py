# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Scoped stratum diagnostics: exact controls and explicitly numerical evidence."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import reduce
from math import gcd

import numpy as np
from numpy.typing import NDArray
from omnibias.core.realization.rank import ExactRankWitness, certify_matrix_rank

from .geometry import RankReport, RegularQuotientChart, rank_report

Array = NDArray[np.float64]


@dataclass(frozen=True)
class MonomialCurveStratum:
    exponents: tuple[int, ...]
    primitive_exponents: tuple[int, ...]
    parameter_multiplicity: int
    intrinsic_class: str
    parameter_rank_at_origin: int
    first_visible_order: int
    real_image_has_boundary: bool
    scope: str = "declared_monomial_curve_at_origin"


def monomial_curve_stratum(exponents: tuple[int, ...]) -> MonomialCurveStratum:
    """Classify t -> (t**e_i) at zero, using the exact primitive exponent semigroup.

    This supported analytic family distinguishes t², t³, and the cusp (t²,t³).
    An intrinsic algebraic singularity concerns the reduced algebraic curve,
    while an even common power also restricts its real parameterized image.
    General neural maps are not routed through this classification.
    """
    if not exponents or any(type(e) is not int or e < 1 for e in exponents):
        raise ValueError("a nonempty tuple of positive integer exponents is required")
    common = reduce(gcd, exponents)
    primitive = tuple(e // common for e in exponents)
    if min(primitive) == 1:
        intrinsic = "smooth_curve_with_boundary" if common % 2 == 0 else "smooth_curve"
    else:
        intrinsic = "intrinsic_algebraic_curve_singularity"
    return MonomialCurveStratum(
        exponents,
        primitive,
        common,
        intrinsic,
        int(min(exponents) == 1),
        min(exponents),
        common % 2 == 0,
    )


@dataclass(frozen=True)
class StratumDiagnostic:
    observations: RankReport
    complete_representation: RankReport | None
    reasons: tuple[str, ...]
    status: str = "finite_order_diagnostic"


def diagnose_stratum(
    observation_jacobian: Array,
    *,
    complete_jacobian: Array | None = None,
    known_affine_chart: RegularQuotientChart | None = None,
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> StratumDiagnostic:
    """Separate undersampling and an established affine fiber from unresolved rank.

    complete_jacobian must describe a separately declared determining
    representation. Both floating ranks remain estimates; a numerical zero
    product never earns an exact nonlinear symmetry or intrinsic singularity.
    """
    observed = rank_report(observation_jacobian, rtol=rtol, atol=atol)
    complete = None
    reasons = []
    if complete_jacobian is not None:
        if complete_jacobian.shape[1] != observation_jacobian.shape[1]:
            raise ValueError("representations must share the same parameter ordering")
        complete = rank_report(complete_jacobian, rtol=rtol, atol=atol)
        if complete.numerical_rank > observed.numerical_rank:
            reasons.append("numerical_observation_deficiency")
    source = complete_jacobian if complete_jacobian is not None else observation_jacobian
    if known_affine_chart is not None:
        if not known_affine_chart.valid_at(source, rtol=rtol):
            reasons.append("affine_chart_invalidated")
        elif known_affine_chart.kernel:
            reasons.append(
                "kernel_of_declared_affine_map; nonlinear_realization_identity_unresolved"
            )
    if observed.numerical_rank < observation_jacobian.shape[1] and not reasons:
        reasons.append("unresolved_parameter_or_image_singularity")
    if not reasons:
        reasons.append("numerically_regular_for_declared_observations")
    return StratumDiagnostic(observed, complete, tuple(reasons))


def exact_rank_report(matrix: Array) -> tuple[RankReport, ExactRankWitness]:
    """Exact rank of the supplied stored dyadic matrix, with a replayable witness.

    This does not enclose perturbations or promote a sampled Jacobian to a
    global constant-rank statement. Both lower and upper rank bounds are earned
    by a nonzero source minor and exact source factorization.
    """
    witness = certify_matrix_rank([[Fraction(float(v)) for v in row]
                                   for row in np.asarray(matrix, dtype=float)])
    numerical = rank_report(matrix)
    return RankReport(
        numerical.numerical_rank,
        numerical.singular_values,
        numerical.threshold,
        witness.rank,
        witness.rank,
        "exact rank of supplied stored matrix",
    ), witness


__all__ = [
    "MonomialCurveStratum",
    "StratumDiagnostic",
    "diagnose_stratum",
    "exact_rank_report",
    "monomial_curve_stratum",
]
