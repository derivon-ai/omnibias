# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Differentiable Boolean operator surface (torch): gates, spectrum, solver, design."""

from __future__ import annotations

from omnibias.boolean.torch.ops.design import (
    degree_penalty,
    influence_penalty,
    target_spectrum_loss,
)
from omnibias.boolean.torch.ops.gates import (
    linear_threshold,
    soft_and,
    soft_implies,
    soft_majority3,
    soft_nand,
    soft_nor,
    soft_not,
    soft_or,
    soft_xnor,
    soft_xor,
    threshold_and,
    threshold_not,
    threshold_or,
)
from omnibias.boolean.torch.ops.solver import (
    BetaAnnealScheduler,
    BooleanSystem,
    SolveResult,
    brute_force_solutions,
    solve,
)
from omnibias.boolean.torch.ops.spectrum import (
    algebraic_degree_soft,
    influences_diff,
    mobius_coeffs,
    mobius_spectrum,
    walsh_coeffs,
    walsh_spectrum,
)

__all__ = [
    "BetaAnnealScheduler",
    "BooleanSystem",
    "SolveResult",
    "algebraic_degree_soft",
    "brute_force_solutions",
    "degree_penalty",
    "influence_penalty",
    "influences_diff",
    "linear_threshold",
    "mobius_coeffs",
    "mobius_spectrum",
    "soft_and",
    "soft_implies",
    "soft_majority3",
    "soft_nand",
    "soft_nor",
    "soft_not",
    "soft_or",
    "soft_xnor",
    "soft_xor",
    "solve",
    "target_spectrum_loss",
    "threshold_and",
    "threshold_not",
    "threshold_or",
    "walsh_coeffs",
    "walsh_spectrum",
]
