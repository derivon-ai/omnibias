# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Learned-facet arrangement LP (theory 03-02).

Wraps the existing interior-point solver and the Neumaier-Shcherbina
bound. Soft membership is temperature collapse (``beta -> inf``,
feasibility), not the founding bias collapse (``delta -> 0``). Do not conflate
the two.
Not a new LP algorithm, not P vs NP.
"""

from __future__ import annotations

from omnibias.convex.arrangement._core import (
    VERTEX_ENUM_MAX_D,
    VERTEX_ENUM_MAX_N,
    DiffMode,
    LearnedPolytope,
    LPOutput,
    dual_objective,
    duality_holds,
    enumerate_vertices,
    honesty_payload,
    knapsack_simplex,
    known_dual_pentagon,
    named_pentagon,
    random_feasible_lp,
    recover_vertex,
    soft_cell_gap_bound,
    soft_membership,
    soft_vertex_dx_dc,
    solve_arrangement_lp,
    sound_lower_bound,
    vertex_optimum,
)

__all__ = [
    "DiffMode",
    "LPOutput",
    "LearnedPolytope",
    "VERTEX_ENUM_MAX_D",
    "VERTEX_ENUM_MAX_N",
    "dual_objective",
    "duality_holds",
    "enumerate_vertices",
    "honesty_payload",
    "knapsack_simplex",
    "known_dual_pentagon",
    "named_pentagon",
    "random_feasible_lp",
    "recover_vertex",
    "soft_cell_gap_bound",
    "soft_membership",
    "soft_vertex_dx_dc",
    "solve_arrangement_lp",
    "sound_lower_bound",
    "vertex_optimum",
]
