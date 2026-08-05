# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Backend-agnostic (numpy) internals: polytopes, matroids, decoders, oracles."""

from __future__ import annotations

from omnibias.combinatorics._core.decode import (
    DecodeError,
    brute_force_min,
    classical_optimum,
    decode,
    max_flow_value,
    solve_lp,
)
from omnibias.combinatorics._core.matroids import (
    GraphicMatroid,
    Matroid,
    PartitionMatroid,
    UniformMatroid,
    independent_sets,
)
from omnibias.combinatorics._core.polytopes import (
    PolytopeSystem,
    assignment_system,
    matroid_system,
    min_cost_flow_system,
    transport_system,
)

__all__ = [
    "DecodeError",
    "GraphicMatroid",
    "Matroid",
    "PartitionMatroid",
    "PolytopeSystem",
    "UniformMatroid",
    "assignment_system",
    "brute_force_min",
    "classical_optimum",
    "decode",
    "independent_sets",
    "matroid_system",
    "max_flow_value",
    "min_cost_flow_system",
    "solve_lp",
    "transport_system",
]
