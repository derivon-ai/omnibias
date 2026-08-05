# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Backend-agnostic (numpy / pure-Python) internals of the discrete substrate."""

from __future__ import annotations

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

__all__ = [
    "AnnealSchedule",
    "DiscreteProblem",
    "DiscreteSolution",
    "GapCertificate",
    "INIT_THETA_SCALE",
    "UnionFind",
    "boolean_constraints",
    "brute_force_min",
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
]
