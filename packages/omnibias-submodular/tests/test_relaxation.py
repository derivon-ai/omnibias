# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Continuous greedy: torch<->jax parity, the (1-1/e) guarantee, and differentiability.

Covers all three closed-form multilinear families -- Coverage, FacilityLocation, and
BudgetAdditive -- so the "bit-identical twins" promise holds for every family that ships a
differentiable relaxation.
"""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.sos import Polynomial
from omnibias.submodular import (
    ONE_MINUS_INV_E,
    BudgetAdditive,
    ContinuousGreedySchedule,
    Coverage,
    FacilityLocation,
    PartitionMatroid,
    SubmodularProblem,
    UniformMatroid,
    brute_force_max,
    continuous_greedy,
)
from omnibias.submodular.functions import SubmodularFunction

# Parity tolerance: the worst torch<->jax gap measured across dozens of seeds / schedules
# (all three families, uniform + partition, betas up to 200) stayed ~1e-14; 1e-9 leaves
# five orders of margin.
_PARITY_TOL = 1e-9

_FAMILIES = ["coverage", "facility", "budget"]


def _problem(seed: int, family: str, matroid_kind: str = "uniform") -> SubmodularProblem:
    rng = np.random.default_rng(seed)
    n, m = 6, 9
    if family == "coverage":
        c = (rng.random((m, n)) < rng.uniform(0.25, 0.55)).astype(float)
        fn: SubmodularFunction = Coverage(c, rng.random(m) + 0.3)
    elif family == "facility":
        fn = FacilityLocation(rng.random((m, n)), rng.random(m) + 0.3)
    else:  # budget
        a = rng.random(n) * 1.5 + 0.2
        fn = BudgetAdditive(a, budget=float(a.sum() * 0.5))
    if matroid_kind == "uniform":
        matroid = UniformMatroid(n, int(rng.integers(2, 5)))
    else:
        matroid = PartitionMatroid([[0, 1, 2], [3, 4, 5]], [1, 2])
    return SubmodularProblem(fn, matroid)


def _coverage_problem(seed: int, matroid_kind: str = "uniform") -> SubmodularProblem:
    return _problem(seed, "coverage", matroid_kind)


@pytest.mark.parametrize("family", _FAMILIES)
@pytest.mark.parametrize("matroid_kind", ["uniform", "partition"])
def test_torch_jax_bit_identical(family: str, matroid_kind: str) -> None:
    pytest.importorskip("jax.numpy")
    pytest.importorskip("torch")
    from omnibias.submodular.jax import submodular_relaxation as jax_relax
    from omnibias.submodular.torch import submodular_relaxation as torch_relax

    worst = 0.0
    for seed in range(6):
        prob = _problem(seed, family, matroid_kind)
        pj = np.asarray(jax_relax(prob))
        pt = torch_relax(prob).detach().numpy()
        worst = max(worst, float(np.max(np.abs(pj - pt))))
    assert worst < _PARITY_TOL, f"{family}/{matroid_kind} torch<->jax parity {worst} >= {_PARITY_TOL}"


@pytest.mark.parametrize("family", _FAMILIES)
def test_continuous_greedy_meets_one_minus_inv_e(family: str) -> None:
    # The exact hard-oracle numpy path carries F(p*) >= (1 - 1/e) OPT for every family.
    for seed in range(6):
        prob = _problem(seed, family)
        p_star, _ = continuous_greedy(prob.function, prob.matroid, steps=40)
        f_frac = float(prob.function.multilinear(p_star))
        _, opt = brute_force_max(prob.function, prob.matroid)
        assert f_frac >= ONE_MINUS_INV_E * opt - 1e-9, f"{family} seed {seed}"


def test_numpy_continuous_greedy_is_in_the_matroid_polytope() -> None:
    # The exact hard-oracle path lands exactly in the matroid polytope.
    for family in _FAMILIES:
        for seed in range(5):
            prob = _problem(seed, family, "partition")
            p_star, _ = continuous_greedy(prob.function, prob.matroid, steps=25)
            assert np.all(p_star >= -1e-12) and np.all(p_star <= 1.0 + 1e-12)
            for group, cap in zip(prob.matroid.groups(), prob.matroid.caps(), strict=True):
                assert float(p_star[group].sum()) <= cap + 1e-9


def test_soft_relaxation_output_is_bounded_and_near_feasible() -> None:
    pytest.importorskip("jax")
    from omnibias.submodular.jax import submodular_relaxation as jax_relax

    for family in _FAMILIES:
        prob = _problem(2, family)
        # The soft (finite-beta) relaxation is only approximately feasible; exact
        # feasibility is delivered by rounding. Higher beta tightens it toward the polytope.
        p = np.asarray(jax_relax(prob, ContinuousGreedySchedule(steps=25, beta=300.0)))
        assert np.all(p >= -1e-9) and np.all(p <= 1.0 + 1e-9)
        assert np.all(np.isfinite(p))
        assert float(p.sum()) <= prob.matroid.rank() + 1e-2


def test_soft_relaxation_approaches_hard_path_at_high_beta() -> None:
    pytest.importorskip("jax")
    from omnibias.submodular.jax import submodular_relaxation as jax_relax

    for family in _FAMILIES:
        prob = _problem(1, family)
        p_hard, _ = continuous_greedy(prob.function, prob.matroid, steps=25)
        p_soft = np.asarray(jax_relax(prob, ContinuousGreedySchedule(steps=25, beta=400.0)))
        assert np.max(np.abs(p_soft - p_hard)) < 1e-2, family


def test_torch_coverage_relaxation_is_differentiable() -> None:
    torch = pytest.importorskip("torch")
    from omnibias.submodular.torch import coverage_multilinear, coverage_relaxation

    rng = np.random.default_rng(0)
    c = (rng.random((7, 6)) < 0.4).astype(float)
    w0 = rng.random(7) + 0.3
    matroid = UniformMatroid(6, 3)
    theta = torch.zeros(7, dtype=torch.float64, requires_grad=True)
    weights = torch.as_tensor(w0) * torch.sigmoid(theta)
    p = coverage_relaxation(torch.as_tensor(c), weights, matroid, ContinuousGreedySchedule.fast())
    obj = coverage_multilinear(p, torch.as_tensor(c), weights)
    obj.backward()
    assert theta.grad is not None and torch.all(torch.isfinite(theta.grad))
    assert float(theta.grad.norm()) > 0.0


def test_torch_facility_relaxation_is_differentiable() -> None:
    torch = pytest.importorskip("torch")
    from omnibias.submodular.torch import facility_multilinear, facility_relaxation

    rng = np.random.default_rng(0)
    gains = rng.random((7, 6))
    w0 = rng.random(7) + 0.3
    matroid = UniformMatroid(6, 3)
    theta = torch.zeros(7, dtype=torch.float64, requires_grad=True)
    weights = torch.as_tensor(w0) * torch.sigmoid(theta)
    p = facility_relaxation(torch.as_tensor(gains), weights, matroid, ContinuousGreedySchedule.fast())
    obj = facility_multilinear(p, torch.as_tensor(gains), weights)
    obj.backward()
    assert theta.grad is not None and torch.all(torch.isfinite(theta.grad))
    assert float(theta.grad.norm()) > 0.0


def test_torch_budget_relaxation_is_differentiable() -> None:
    torch = pytest.importorskip("torch")
    from omnibias.submodular.torch import budget_multilinear, budget_relaxation

    rng = np.random.default_rng(0)
    a = rng.random(6) * 1.5 + 0.2
    matroid = UniformMatroid(6, 3)
    # Differentiate F(p*) through the budget scalar (the constant-support twin is smooth in
    # p and in the budget via the min); dF/dbudget = P(T > budget) is finite.
    budget = torch.tensor(float(a.sum() * 0.5), dtype=torch.float64, requires_grad=True)
    p = budget_relaxation(a, budget, matroid, ContinuousGreedySchedule.fast())
    obj = budget_multilinear(p, a, budget)
    obj.backward()
    assert budget.grad is not None and bool(torch.isfinite(budget.grad))


def _graph(seed: int, n: int = 6) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.random((n, n))
    w = np.triu(a, 1) * (rng.random((n, n)) < 0.6)  # sparse upper triangle
    return w + w.T  # symmetric, zero diagonal


def test_graphcut_multilinear_torch_jax_bit_identical() -> None:
    pytest.importorskip("jax.numpy")
    torch = pytest.importorskip("torch")
    from omnibias.submodular.jax import graphcut_multilinear as jax_gc
    from omnibias.submodular.torch import graphcut_multilinear as torch_gc

    worst = 0.0
    for seed in range(6):
        w = _graph(seed)
        rng = np.random.default_rng(100 + seed)
        pt = rng.random((5, w.shape[0]))  # a batch of fractional points
        gj = np.asarray(jax_gc(pt, w))
        gt = torch_gc(torch.as_tensor(pt), torch.as_tensor(w)).detach().numpy()
        worst = max(worst, float(np.max(np.abs(gj - gt))))
    assert worst < _PARITY_TOL, f"graphcut twin parity {worst} >= {_PARITY_TOL}"


def test_graphcut_multilinear_matches_value_on_cube() -> None:
    pytest.importorskip("jax.numpy")
    from omnibias.submodular import GraphCut
    from omnibias.submodular.jax import graphcut_multilinear as jax_gc

    for seed in range(4):
        w = _graph(seed)
        fn = GraphCut(w)
        rng = np.random.default_rng(seed)
        for _ in range(8):
            x = (rng.random(fn.n) < 0.5).astype(float)
            assert abs(float(jax_gc(x, w)) - float(fn.value(x))) < 1e-9


def test_torch_graphcut_multilinear_is_differentiable() -> None:
    torch = pytest.importorskip("torch")
    from omnibias.submodular.torch import graphcut_multilinear

    w = torch.as_tensor(_graph(0))
    theta = torch.zeros(6, dtype=torch.float64, requires_grad=True)
    p = torch.sigmoid(theta)  # a differentiable fractional point in (0, 1)^n
    obj = graphcut_multilinear(p, w)
    obj.backward()
    assert theta.grad is not None and torch.all(torch.isfinite(theta.grad))
    assert float(theta.grad.norm()) > 0.0


class _GreedyPathOnly(SubmodularFunction):
    """A stand-in greedy-path function with no closed-form multilinear extension."""

    def __init__(self, n: int) -> None:
        self._n = n

    @property
    def n(self) -> int:
        return self._n

    def multilinear(self, p: object) -> float:
        raise NotImplementedError("greedy-path function: no closed-form multilinear extension")

    def to_polynomial(self) -> Polynomial:
        raise NotImplementedError("greedy-path function: no multilinear polynomial")


def test_unsupported_relaxation_raises_honestly() -> None:
    pytest.importorskip("jax")
    from omnibias.submodular.jax import submodular_relaxation as jax_relax

    prob = SubmodularProblem(_GreedyPathOnly(6), UniformMatroid(6, 2))
    with pytest.raises(NotImplementedError, match="maximize"):
        jax_relax(prob)
