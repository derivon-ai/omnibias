# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Differentiable discrete relaxation, decoding, and certified optimality gaps.

The DiscreteProblem protocol supplies an energy and polynomial representation.
Annealing uses its closed-form gradient; decoding supplies an upper bound and
optional SOS certificates supply a lower bound. The supported result is a certified gap; a zero gap requires matching bounds.

Temperature collapse hardens sigmoid(beta z) for feasibility as beta grows.
This differs from founding bias collapse, the delta -> 0 derivative limit.
Use matching dtypes and JAX_ENABLE_X64=1 for float64 comparisons.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError as _PkgNotFound
from importlib.metadata import version as _pkg_version

from omnibias.discrete._core.bound import (
    gershgorin_min_eig_lower,
    lasserre_lower_bound,
    negative_coeff_lower_bound,
)
from omnibias.discrete._core.decision import mean_normalized_regret, spo_plus_subgradient
from omnibias.discrete._core.decode import (
    brute_force_min,
    decode,
    energy,
    flip_deltas,
    is_binary,
    one_flip_descent,
    round_relaxed,
)
from omnibias.discrete._core.problem import DiscreteProblem, boolean_constraints
from omnibias.discrete._core.relax import INIT_THETA_SCALE, initial_theta
from omnibias.discrete._core.schedule import AnnealSchedule
from omnibias.discrete._core.solution import DiscreteSolution, GapCertificate
from omnibias.discrete._core.union_find import UnionFind, is_forest
from omnibias.discrete.certify import TightenedGap, certify_gap, tighten_gap
from omnibias.discrete.proposers import AnnealDescentProposer

try:
    __version__ = _pkg_version("omnibias-discrete")
except _PkgNotFound:  # pragma: no cover - bare source checkout
    __version__ = "0.0.0+unknown"

# Limit family exposed as package metadata.
__lineage__ = "temperature collapse"

__all__ = [
    "AnnealDescentProposer",
    "AnnealSchedule",
    "DiscreteProblem",
    "DiscreteSolution",
    "GapCertificate",
    "INIT_THETA_SCALE",
    "TightenedGap",
    "UnionFind",
    "__lineage__",
    "__version__",
    "boolean_constraints",
    "brute_force_min",
    "certify_gap",
    "decode",
    "energy",
    "flip_deltas",
    "gershgorin_min_eig_lower",
    "initial_theta",
    "is_binary",
    "is_forest",
    "lasserre_lower_bound",
    "mean_normalized_regret",
    "negative_coeff_lower_bound",
    "one_flip_descent",
    "round_relaxed",
    "spo_plus_subgradient",
    "tighten_gap",
]
