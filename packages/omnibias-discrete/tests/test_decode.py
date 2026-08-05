# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Generic decoder: rounding, 1-flip local search, oracle, and fast-path == fallback."""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.discrete import (
    brute_force_min,
    decode,
    flip_deltas,
    is_binary,
    one_flip_descent,
    round_relaxed,
)


def test_is_binary_and_round_relaxed() -> None:
    assert is_binary(np.array([0.0, 1.0, 0.0]))
    assert not is_binary(np.array([0.5, 1.0]))
    assert np.array_equal(round_relaxed(np.array([0.4, 0.5, 0.9])), np.array([0.0, 1.0, 1.0]))


def test_one_flip_descent_reaches_a_local_minimum(make_toy) -> None:  # type: ignore[no-untyped-def]
    prob = make_toy([[0.0, -2.0], [-2.0, 0.0]], c=[1.0, 1.0])
    x, e = one_flip_descent(prob, np.array([1.0, 0.0]))
    assert e == pytest.approx(float(prob.energy(x)))
    assert np.all(flip_deltas(prob, x) >= -1e-9)  # no single flip improves


def test_decode_is_an_upper_bound_and_finds_the_optimum(make_toy) -> None:  # type: ignore[no-untyped-def]
    # A frustrated 3-var instance; decode must not beat the exact optimum.
    rng = np.random.default_rng(3)
    m = rng.standard_normal((3, 3))
    prob = make_toy(m + m.T, c=rng.standard_normal(3), const=0.2)
    _, e_dec = decode(prob, n_starts=16)
    _, e_min = brute_force_min(prob)
    assert e_dec >= e_min - 1e-9
    assert e_dec == pytest.approx(e_min, abs=1e-9)  # enough starts -> exact here


def test_fast_path_matches_the_generic_fallback(make_toy) -> None:  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(7)
    m = rng.standard_normal((6, 6))
    Q, c = m + m.T, rng.standard_normal(6)
    fast = make_toy(Q, c=c)
    slow = make_toy(Q, c=c, fast=False)
    assert not hasattr(slow, "flip_deltas")  # forces the batched-energy fallback
    # Closed-form flip deltas and the batched-energy fallback agree bit-for-bit-ish.
    x = rng.integers(0, 2, size=6).astype(float)
    assert np.allclose(flip_deltas(fast, x), flip_deltas(slow, x), atol=1e-9)
    assert decode(fast, seed=1) == decode(slow, seed=1)


def test_brute_force_matches_exhaustive_scan_and_caps_n(make_toy) -> None:  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(1)
    m = rng.standard_normal((4, 4))
    prob = make_toy(m + m.T, c=rng.standard_normal(4))
    x_best, e_best = brute_force_min(prob)
    all_x = np.array([[int(b) for b in np.binary_repr(k, 4)] for k in range(16)], dtype=float)
    e_all = np.asarray(prob.energy(all_x))
    assert e_best == pytest.approx(float(np.min(e_all)))
    assert float(prob.energy(np.array(x_best, dtype=float))) == pytest.approx(e_best)
    with pytest.raises(ValueError, match="exponential"):
        brute_force_min(prob, max_n=2)
