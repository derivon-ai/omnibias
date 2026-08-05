# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Measure the accuracy cliff: soft-solver factoring success vs bit-length.

For each factor bit-width we encode a handful of semiprimes as soft-gate Boolean
systems, run the beta-annealed solver, and record the fraction it recovers (and
verifies). The success rate falls off a cliff as the bit-width grows -- the search
space is *not* reduced by the relaxation. Pure heuristic; every "success" is an
exact verification, never an unchecked guess.
"""

from __future__ import annotations

from dataclasses import dataclass

from omnibias.boolean.torch.ops.solver import solve

from examples.boolean_limits.multiplier import (
    bits_to_factors,
    factor_system,
    semiprimes_for_width,
)


@dataclass(frozen=True)
class FactorAttempt:
    """One factoring attempt: the target, the recovered pair, and verification."""

    n_value: int
    width: int
    factors: tuple[int, int] | None
    verified: bool


def solve_one(
    n_value: int,
    width: int,
    *,
    steps: int = 200,
    restarts: int = 16,
    seed: int = 0,
) -> FactorAttempt:
    """Attempt to factor a single ``N`` with the annealed soft-gate solver."""
    system = factor_system(n_value, width, width)
    res = solve(system, steps=steps, restarts=restarts, seed=seed)
    if res.assignment is None or not res.verified:
        return FactorAttempt(n_value, width, None, False)
    p, q = bits_to_factors(res.assignment, width, width)
    nontrivial = p > 1 and q > 1 and p * q == n_value
    return FactorAttempt(n_value, width, (p, q), nontrivial)


def success_vs_bitlength(
    widths: tuple[int, ...] = (2, 3, 4),
    *,
    max_per_width: int = 6,
    steps: int = 200,
    restarts: int = 16,
    seed: int = 0,
) -> dict[int, float]:
    """Success rate (fraction verified non-trivially) per factor bit-width."""
    out: dict[int, float] = {}
    for width in widths:
        targets = semiprimes_for_width(width)[:max_per_width]
        if not targets:
            out[width] = 0.0
            continue
        wins = sum(
            solve_one(n, width, steps=steps, restarts=restarts, seed=seed).verified
            for n in targets
        )
        out[width] = wins / len(targets)
    return out


__all__ = ["FactorAttempt", "solve_one", "success_vs_bitlength"]
