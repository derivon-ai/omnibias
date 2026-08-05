# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Backend-agnostic (numpy) routing internals: decoding, oracle, decision helpers."""

from __future__ import annotations

from omnibias.routing._core.decision import (
    edge_matrix,
    normalized_regret,
    optimal_tour_costs,
    spo_plus_gradient,
)
from omnibias.routing._core.decode import (
    decode_tour,
    held_karp_dp,
    is_valid_tour,
    nearest_neighbor,
    tour_cost,
    two_opt,
)

__all__ = [
    "decode_tour",
    "edge_matrix",
    "held_karp_dp",
    "is_valid_tour",
    "nearest_neighbor",
    "normalized_regret",
    "optimal_tour_costs",
    "spo_plus_gradient",
    "tour_cost",
    "two_opt",
]
