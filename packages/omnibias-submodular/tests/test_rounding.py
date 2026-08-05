# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Pipage / swap rounding: feasibility and the value-preservation guarantees."""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.submodular import (
    Coverage,
    FacilityLocation,
    PartitionMatroid,
    SubmodularProblem,
    UniformMatroid,
    continuous_greedy,
    pipage_round,
    swap_round,
)


def _instances():
    rng = np.random.default_rng(0)
    out = []
    for seed in range(5):
        r = np.random.default_rng(seed)
        cov = Coverage((r.random((8, 6)) < 0.4).astype(float), r.random(8) + 0.3)
        out.append((cov, UniformMatroid(6, 3)))
        out.append((cov, PartitionMatroid([[0, 1, 2], [3, 4, 5]], [1, 2])))
        fac = FacilityLocation(r.random((5, 6)), r.random(5) + 0.2)
        out.append((fac, UniformMatroid(6, 2)))
    del rng
    return out


def test_pipage_is_feasible_and_preserves_value() -> None:
    for fn, matroid in _instances():
        p_star, _ = continuous_greedy(fn, matroid, steps=25)
        selection, value = pipage_round(fn, matroid, p_star)
        x = np.asarray(selection, dtype=float)
        assert matroid.is_independent(x), "pipage output must be independent"
        assert set(selection) <= {0, 1}
        f_frac = float(fn.multilinear(p_star))
        assert value >= f_frac - 1e-9, f"pipage lost value: {value} < F(p*)={f_frac}"


def test_swap_is_feasible_and_preserves_value_in_expectation() -> None:
    # Swap rounding is randomized: the guarantee E[f(S)] >= F(p*) is in expectation, so
    # we average over many seeds (a small slack absorbs residual Monte-Carlo variance).
    for fn, matroid in _instances():
        p_star, bases = continuous_greedy(fn, matroid, steps=25)
        f_frac = float(fn.multilinear(p_star))
        vals = []
        for seed in range(256):
            selection, value = swap_round(fn, matroid, bases, seed=seed)
            assert matroid.is_independent(np.asarray(selection, dtype=float)), "swap must be feasible"
            vals.append(value)
        # At large N the mean converges to E[f(S)] >= F(p*) (verified ~0 shortfall at
        # N=2e4); the 1.2e-2 slack absorbs the finite-sample noise at this N.
        assert float(np.mean(vals)) >= f_frac - 1.2e-2


def test_swap_is_deterministic_for_a_fixed_seed() -> None:
    fn = Coverage((np.random.default_rng(9).random((8, 6)) < 0.4).astype(float))
    matroid = UniformMatroid(6, 3)
    _, bases = continuous_greedy(fn, matroid, steps=20)
    a = swap_round(fn, matroid, bases, seed=3)
    b = swap_round(fn, matroid, bases, seed=3)
    assert a == b


def test_swap_requires_bases() -> None:
    fn = Coverage(np.eye(3))
    with pytest.raises(ValueError, match="non-empty"):
        swap_round(fn, UniformMatroid(3, 1), [])
