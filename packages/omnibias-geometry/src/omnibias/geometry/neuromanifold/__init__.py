# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Realization geometry, explicit symmetry charts, and collision proposals.

Numerical producers are permissive. Sound acceptance and finite replay are
provided by downstream omnibias.verify consumers.
"""

from .geometry import (
    ExtrinsicGeometry,
    ObservationMetric,
    RankReport,
    Realization,
    ReducedStep,
    RegularQuotientChart,
    VisibilityReport,
    affine_quotient,
    escape_candidates,
    extrinsic_geometry,
    higher_order_visibility,
    least_squares_hessian,
    normal_acceleration,
    quotient_step,
    rank_report,
)
from .strata import (
    MonomialCurveStratum,
    StratumDiagnostic,
    diagnose_stratum,
    exact_rank_report,
    monomial_curve_stratum,
)
from .symmetry import (
    HiddenAction,
    HomogeneousAnchorChart,
    duplicate_fibers,
    hidden_permutation,
    hidden_sign,
    homogeneous_anchor_chart,
    homogeneous_scaling,
)

__all__ = [
    "ExtrinsicGeometry",
    "HiddenAction",
    "HomogeneousAnchorChart",
    "MonomialCurveStratum",
    "ObservationMetric",
    "RankReport",
    "Realization",
    "ReducedStep",
    "RegularQuotientChart",
    "StratumDiagnostic",
    "VisibilityReport",
    "affine_quotient",
    "diagnose_stratum",
    "duplicate_fibers",
    "escape_candidates",
    "exact_rank_report",
    "extrinsic_geometry",
    "hidden_permutation",
    "hidden_sign",
    "higher_order_visibility",
    "homogeneous_anchor_chart",
    "homogeneous_scaling",
    "least_squares_hessian",
    "monomial_curve_stratum",
    "normal_acceleration",
    "quotient_step",
    "rank_report",
]
