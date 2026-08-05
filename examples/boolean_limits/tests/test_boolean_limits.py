# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Reproducible tiny case for the factoring limitation study."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from omnibias.boolean.torch.ops.solver import brute_force_solutions  # noqa: E402

from examples.boolean_limits.experiment import solve_one, success_vs_bitlength  # noqa: E402
from examples.boolean_limits.multiplier import (  # noqa: E402
    bits_to_factors,
    factor_system,
    semiprimes_for_width,
)


def test_semiprimes_for_width_excludes_trivial() -> None:
    # 2x2=4, 2x3=6, 3x3=9 (both factors in [2,3], product > 3 so no 1*N shortcut).
    assert semiprimes_for_width(2) == [4, 6, 9]


def test_factor_system_has_exact_nontrivial_solutions() -> None:
    system = factor_system(6, 2, 2)
    sols = brute_force_solutions(system)
    factors = {tuple(sorted(bits_to_factors(b, 2, 2))) for b in sols}
    assert (2, 3) in factors  # 6 = 2 * 3
    assert all(p * q == 6 for p, q in (bits_to_factors(b, 2, 2) for b in sols))


def test_solver_recovers_tiny_factorization() -> None:
    # n = 4 latent bits: the annealed solver reliably recovers and *verifies* it.
    attempt = solve_one(6, 2, steps=200, restarts=24, seed=0)
    assert attempt.verified
    assert attempt.factors is not None
    p, q = attempt.factors
    assert p * q == 6 and p > 1 and q > 1


def test_success_rate_smoke() -> None:
    rates = success_vs_bitlength(widths=(2,), max_per_width=2, steps=120, restarts=12)
    assert set(rates) == {2}
    assert 0.0 <= rates[2] <= 1.0
