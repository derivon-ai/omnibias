# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Sieve-streaming: one-pass (1/2 - eps) under a cardinality budget, incl. greedy-path fns."""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.submodular import (
    Coverage,
    FacilityLocation,
    LogDeterminant,
    Saturated,
    UniformMatroid,
    brute_force_max,
    sieve_streaming,
)


def _coverage(seed: int, n: int = 10, m: int = 12) -> Coverage:
    r = np.random.default_rng(seed)
    return Coverage((r.random((m, n)) < 0.35).astype(float), r.random(m) + 0.2)


def _facility(seed: int, n: int = 10, m: int = 7) -> FacilityLocation:
    r = np.random.default_rng(seed)
    return FacilityLocation(r.random((m, n)), r.random(m) + 0.2)


def _logdet(seed: int, n: int = 9) -> LogDeterminant:
    r = np.random.default_rng(seed)
    a = r.random((n, n))
    return LogDeterminant(a @ a.T + 0.5 * np.eye(n))  # SPD kernel


_MAKERS = [_coverage, _facility, _logdet]
_EPS = 0.1


@pytest.mark.parametrize("maker", _MAKERS)
@pytest.mark.parametrize("seed", range(5))
def test_sieve_meets_half_minus_eps(maker, seed: int) -> None:  # type: ignore[no-untyped-def]
    f = maker(seed)
    k = 4
    sel, val = sieve_streaming(f, k, epsilon=_EPS)
    _, opt = brute_force_max(f, UniformMatroid(f.n, k))
    assert val >= (0.5 - _EPS) * opt - 1e-9
    assert float(f.value(np.array(sel, dtype=float))) == pytest.approx(val, abs=1e-9)


@pytest.mark.parametrize("maker", _MAKERS)
def test_sieve_respects_cardinality(maker) -> None:  # type: ignore[no-untyped-def]
    f = maker(0)
    k = 3
    sel, _ = sieve_streaming(f, k, epsilon=_EPS)
    assert sum(sel) <= k


def test_sieve_guarantee_is_order_independent() -> None:
    f = _coverage(1)
    k = 4
    _, opt = brute_force_max(f, UniformMatroid(f.n, k))
    rng = np.random.default_rng(7)
    for _ in range(5):
        order = rng.permutation(f.n)
        _, val = sieve_streaming(f, k, epsilon=_EPS, order=order)
        assert val >= (0.5 - _EPS) * opt - 1e-9


def test_sieve_works_on_greedy_path_saturated() -> None:
    # Saturated has no multilinear extension; the sieve must still run via value/marginals.
    base = _coverage(2)
    cap = 0.5 * float(base.value(np.ones(base.n)))
    f = Saturated(base, cap)
    k = 5
    sel, val = sieve_streaming(f, k, epsilon=_EPS)
    _, opt = brute_force_max(f, UniformMatroid(f.n, k))
    assert val >= (0.5 - _EPS) * opt - 1e-9
    assert val <= cap + 1e-9  # never exceeds the cap


def test_sieve_rejects_bad_arguments() -> None:
    f = _coverage(0)
    with pytest.raises(ValueError):
        sieve_streaming(f, 0)
    with pytest.raises(ValueError):
        sieve_streaming(f, 3, epsilon=0.0)
    with pytest.raises(ValueError):
        sieve_streaming(f, 3, epsilon=1.0)
    with pytest.raises(ValueError, match="permutation"):
        sieve_streaming(f, 3, order=[0, 0, 1])
