# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""The submodular function family: monotone + submodular, exact F, gradient, polynomial."""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.submodular import (
    BudgetAdditive,
    Coverage,
    FacilityLocation,
    SubmodularFunction,
    is_monotone_submodular,
)


def _coverage(seed: int) -> Coverage:
    rng = np.random.default_rng(seed)
    c = (rng.random((7, 6)) < 0.45).astype(float)
    return Coverage(c, rng.random(7) + 0.3)


def _facility(seed: int) -> FacilityLocation:
    rng = np.random.default_rng(seed)
    return FacilityLocation(rng.random((5, 6)), rng.random(5) + 0.2)


def _budget(seed: int) -> BudgetAdditive:
    rng = np.random.default_rng(seed)
    return BudgetAdditive(rng.random(6) + 0.2, budget=1.4)


_FACTORIES = [_coverage, _facility, _budget]


@pytest.mark.parametrize("factory", _FACTORIES)
def test_monotone_and_submodular(factory) -> None:
    for seed in range(4):
        fn: SubmodularFunction = factory(seed)
        monotone, submodular = is_monotone_submodular(fn, samples=128, seed=seed)
        assert monotone, f"{type(fn).__name__} not monotone"
        assert submodular, f"{type(fn).__name__} not submodular"


@pytest.mark.parametrize("factory", _FACTORIES)
def test_multilinear_agrees_with_value_on_the_cube(factory) -> None:
    fn = factory(0)
    n = fn.n
    rng = np.random.default_rng(1)
    for _ in range(20):
        x = rng.integers(0, 2, size=n).astype(float)
        assert abs(float(fn.multilinear(x)) - float(fn.value(x))) < 1e-9


@pytest.mark.parametrize("factory", _FACTORIES)
def test_multilinear_is_the_monte_carlo_expectation(factory) -> None:
    fn = factory(2)
    n = fn.n
    rng = np.random.default_rng(3)
    p = rng.random(n)
    draws = (rng.random((40000, n)) < p[None, :]).astype(float)
    mc = float(np.mean(fn.value(draws)))
    assert abs(mc - float(fn.multilinear(p))) < 2e-2


@pytest.mark.parametrize("factory", _FACTORIES)
def test_gradient_matches_central_difference(factory) -> None:
    fn = factory(4)
    n = fn.n
    rng = np.random.default_rng(5)
    p = 0.2 + 0.6 * rng.random(n)  # interior point
    grad = fn.multilinear_grad(p)
    eps = 1e-6
    for i in range(n):
        hi = p.copy()
        hi[i] += eps
        lo = p.copy()
        lo[i] -= eps
        fd = (float(fn.multilinear(hi)) - float(fn.multilinear(lo))) / (2 * eps)
        assert abs(grad[i] - fd) < 1e-4


@pytest.mark.parametrize("factory", _FACTORIES)
def test_to_polynomial_matches_multilinear(factory) -> None:
    pytest.importorskip("omnibias.sos")
    fn = factory(6)
    n = fn.n
    poly = fn.to_polynomial()
    rng = np.random.default_rng(7)
    for _ in range(10):
        p = rng.random(n)
        assert abs(float(poly.evaluate(p)) - float(fn.multilinear(p))) < 1e-9


def test_coverage_rejects_bad_membership() -> None:
    with pytest.raises(ValueError, match="lie in"):
        Coverage(np.array([[1.5, 0.0]]))
    with pytest.raises(ValueError, match="nonnegative"):
        Coverage(np.array([[1.0, 0.0]]), weights=np.array([-1.0]))
