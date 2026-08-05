# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Entropic relaxation layers: torch<->jax parity, feasibility, vertex limit, training."""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.combinatorics import (
    AnnealSchedule,
    AssignmentProblem,
    GraphicMatroid,
    MinCostFlowProblem,
    PartitionMatroid,
    UniformMatroid,
    classical_optimum,
    decode,
)

K = 8
PARITY_TOL = 1e-9

_ARCS = ((0, 1), (0, 2), (1, 3), (2, 3), (1, 2))
_CAP = np.array([3.0, 2.0, 2.0, 3.0, 1.0])


def _flow_problem(cost: np.ndarray) -> MinCostFlowProblem:
    return MinCostFlowProblem(4, _ARCS, cost, _CAP, source=0, sink=3, value=4.0)


def _need_backends() -> None:
    pytest.importorskip("jax")
    pytest.importorskip("torch")


@pytest.mark.parametrize("seed", range(K))
def test_assignment_parity(seed: int) -> None:
    _need_backends()
    from omnibias.combinatorics.jax import assignment_relaxation as ja
    from omnibias.combinatorics.torch import assignment_relaxation as ta

    cost = np.random.default_rng(seed).random((6, 6))
    xj = np.asarray(ja(cost))
    xt = ta(cost).detach().numpy()
    assert np.max(np.abs(xj - xt)) < PARITY_TOL


@pytest.mark.parametrize("seed", range(K))
def test_transport_parity(seed: int) -> None:
    _need_backends()
    from omnibias.combinatorics.jax import transport_relaxation as jt
    from omnibias.combinatorics.torch import transport_relaxation as tt

    rng = np.random.default_rng(seed)
    cost = rng.random((3, 4))
    supply = np.array([2.0, 3.0, 1.0])
    demand = np.array([1.0, 2.0, 2.0, 1.0])
    xj = np.asarray(jt(cost, supply, demand))
    xt = tt(cost, supply, demand).detach().numpy()
    assert np.max(np.abs(xj - xt)) < PARITY_TOL


@pytest.mark.parametrize("seed", range(K))
def test_flow_parity(seed: int) -> None:
    _need_backends()
    from omnibias.combinatorics.jax import min_cost_flow_relaxation as jf
    from omnibias.combinatorics.torch import min_cost_flow_relaxation as tf

    cost = np.random.default_rng(seed).random(len(_ARCS))
    prob = _flow_problem(cost)
    xj = np.asarray(jf(cost, prob))
    xt = tf(cost, prob).detach().numpy()
    assert np.max(np.abs(xj - xt)) < PARITY_TOL


@pytest.mark.parametrize("seed", range(K))
def test_matroid_parity(seed: int) -> None:
    _need_backends()
    from omnibias.combinatorics.jax import matroid_relaxation as jm
    from omnibias.combinatorics.torch import matroid_relaxation as tm

    rng = np.random.default_rng(seed)
    cases = (
        (rng.standard_normal(8), UniformMatroid(8, 3)),
        (rng.standard_normal(6), PartitionMatroid(((0, 1, 2), (3, 4, 5)), (1, 2))),
        (rng.random(4) + 0.1, GraphicMatroid(4, ((0, 1), (1, 2), (2, 0), (2, 3)))),
    )
    for w, mat in cases:
        xj = np.asarray(jm(w, mat))
        xt = tm(w, mat).detach().numpy()
        assert np.max(np.abs(xj - xt)) < PARITY_TOL, type(mat).__name__


def test_assignment_near_doubly_stochastic() -> None:
    _need_backends()
    from omnibias.combinatorics.jax import assignment_relaxation as ja

    P = np.asarray(ja(np.random.default_rng(0).random((7, 7))))
    assert np.all(P >= -1e-9)
    assert np.allclose(P.sum(axis=1), 1.0, atol=5e-2)
    assert np.allclose(P.sum(axis=0), 1.0, atol=5e-2)


def test_transport_near_marginals() -> None:
    _need_backends()
    from omnibias.combinatorics.jax import transport_relaxation as jt

    supply = np.array([2.0, 3.0, 1.0])
    demand = np.array([1.0, 2.0, 2.0, 1.0])
    P = np.asarray(jt(np.random.default_rng(1).random((3, 4)), supply, demand))
    assert np.allclose(P.sum(axis=1), supply, atol=1e-1)
    assert np.allclose(P.sum(axis=0), demand, atol=1e-1)


def test_flow_near_conservation() -> None:
    _need_backends()
    from omnibias.combinatorics.jax import min_cost_flow_relaxation as jf

    cost = np.random.default_rng(2).random(len(_ARCS))
    prob = _flow_problem(cost)
    f = np.asarray(jf(cost, prob))
    resid = prob.system().A_eq @ f - prob.system().b_eq
    assert np.max(np.abs(resid)) < 1e-1
    assert np.all(f >= -1e-9) and np.all(f <= _CAP + 1e-2)


def test_beta_to_infinity_gives_a_vertex() -> None:
    """A heavier schedule drives the assignment relaxation toward a permutation vertex.

    Uses a well-separated instance (unique optimum = the identity permutation); an
    entropic coupling legitimately stays fractional only across (near-)ties.
    """
    _need_backends()
    from omnibias.combinatorics.jax import assignment_relaxation as ja

    cost = 1.0 - np.eye(6)  # unique, strictly separated optimum: the identity matching
    light = np.asarray(ja(cost, AnnealSchedule(beta0=0.5, beta_growth=1.3, stages=4, steps=30)))
    heavy = np.asarray(ja(cost, AnnealSchedule(beta0=0.5, beta_growth=1.7, stages=18, steps=100)))
    # each row of the heavy relaxation is nearly one-hot (a permutation vertex)
    assert heavy.max(axis=1).min() > 0.98
    assert np.allclose(np.diag(heavy), 1.0, atol=2e-2)  # the identity vertex
    assert heavy.max(axis=1).min() >= light.max(axis=1).min() - 1e-9  # monotone sharpening


def test_matroid_beta_to_infinity_is_binary() -> None:
    _need_backends()
    from omnibias.combinatorics.jax import matroid_relaxation as jm

    w = np.random.default_rng(5).standard_normal(8)
    r = np.asarray(jm(w, UniformMatroid(8, 3), AnnealSchedule(beta0=1.0, beta_growth=1.7, stages=18, steps=1)))
    assert np.all(np.minimum(r, 1.0 - r) < 5e-2)  # every entry near 0 or 1


def test_differentiable_jax() -> None:
    pytest.importorskip("jax")
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    from omnibias.combinatorics.jax import assignment_relaxation as ja

    cost = jnp.asarray(np.random.default_rng(1).random((5, 5)))

    def scalar(c: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(ja(c) * c)

    g = np.asarray(jax.grad(scalar)(cost))
    assert g.shape == (5, 5)
    assert np.all(np.isfinite(g)) and np.any(g != 0.0)


def test_differentiable_torch() -> None:
    pytest.importorskip("torch")
    import torch
    from omnibias.combinatorics.torch import min_cost_flow_relaxation as tf

    cost = torch.tensor(
        np.random.default_rng(1).random(len(_ARCS)), dtype=torch.float64, requires_grad=True
    )
    prob = _flow_problem(cost.detach().numpy())
    loss = torch.sum(tf(cost, prob) * cost)
    loss.backward()
    assert cost.grad is not None
    assert bool(torch.all(torch.isfinite(cost.grad))) and bool(torch.any(cost.grad != 0.0))


def test_train_through_improves_decoded_decision() -> None:
    """Training a predicted cost *through* the Sinkhorn layer lowers the decoded true cost."""
    pytest.importorskip("torch")
    import torch
    from omnibias.combinatorics.torch import assignment_relaxation as ta

    rng = np.random.default_rng(0)
    true_cost = rng.random((5, 5))
    true_t = torch.tensor(true_cost, dtype=torch.float64)
    prob = AssignmentProblem(true_cost)
    _, opt = classical_optimum(prob)

    soft = AnnealSchedule(beta0=0.5, beta_growth=1.3, stages=6, steps=40)
    # start adversarially: predict the negated cost, so the layer prefers the worst matching
    pred = torch.nn.Parameter(-true_t.clone())

    with torch.no_grad():
        _, before = decode(prob, relaxed=ta(pred, soft).detach().numpy())

    opt_alg = torch.optim.Adam([pred], lr=0.1)
    for _ in range(200):
        opt_alg.zero_grad()
        loss = torch.sum(ta(pred, soft) * true_t)  # differentiable decision cost
        loss.backward()
        opt_alg.step()

    with torch.no_grad():
        _, after = decode(prob, relaxed=ta(pred, soft).detach().numpy())

    assert after < before - 1e-6  # training strictly improves the decoded decision
    assert after == pytest.approx(opt, abs=1e-9)  # and reaches the true optimum
