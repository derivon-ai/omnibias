# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Torch backend for omnibias-convex."""

from __future__ import annotations

from omnibias.convex.torch.layer import lp_layer, qp_layer
from omnibias.convex.torch.penalty import (
    penalty_descent,
    penalty_gradient,
    solve_lp_penalty,
    solve_qp_penalty,
)
from omnibias.convex.torch.solver import InfeasibleProblemError, solve_lp, solve_qp

__all__ = [
    "InfeasibleProblemError",
    "lp_layer",
    "penalty_descent",
    "penalty_gradient",
    "qp_layer",
    "solve_lp",
    "solve_lp_penalty",
    "solve_qp",
    "solve_qp_penalty",
]
