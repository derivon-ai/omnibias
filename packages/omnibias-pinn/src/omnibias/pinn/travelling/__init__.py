# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Travelling-wave / soliton fields (theory 02-09, gated).

Tanh algebra, not a collapse. A multi-kink sum is not the n-soliton formula.
"""

from __future__ import annotations

from omnibias.core.tanh_method import (
    G1_NAMES,
    PDESpec,
    TravellingWaveAnsatz,
    evaluate_ansatz,
    solve_ansatz,
    substitute,
    verify_exact,
)

__all__ = [
    "G1_NAMES",
    "PDESpec",
    "SolitonField",
    "TravellingWaveAnsatz",
    "evaluate_ansatz",
    "solve_ansatz",
    "substitute",
    "verify_exact",
]


def __getattr__(name: str) -> object:
    if name == "SolitonField":
        from omnibias.pinn.travelling.torch import SolitonField

        return SolitonField
    raise AttributeError(f"module {__name__!r} has no attribute {name}")
