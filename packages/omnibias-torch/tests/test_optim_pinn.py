# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Torch twin of the JAX optim_pinn tests."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from omnibias.torch.optim_pinn import (
    LMConfig,
    cg_solve,
    engd_step,
    gauss_newton_solve,
    gn_row_budget,
    grid_line_search,
    init_lm_state,
    kron_damped_direction,
    lbfgs_with_counts,
    lm_step,
    lowrank_damped_solve,
    ravel_params,
    select_rows,
)

DT = torch.float64


@pytest.fixture(autouse=True)
def _f64():
    prev = torch.get_default_dtype()
    torch.set_default_dtype(DT)
    yield
    torch.set_default_dtype(prev)


def _problem(n_rows, n_par, seed=0):
    g = np.random.default_rng(seed)
    A = torch.tensor(g.standard_normal((n_rows, n_par)))
    y = torch.tensor(g.standard_normal(n_rows))

    def f(theta):
        return torch.tanh(A @ theta) - 0.3 * y

    return f, torch.tensor(0.2 * g.standard_normal(n_par))


def _jac(f, th):
    return torch.func.jacfwd(f)(th)


def test_row_budget_and_selection():
    assert gn_row_budget(100) == 200
    assert gn_row_budget(10_000, memory_bytes=2**30) < 20_000
    assert gn_row_budget(3) == 64 and gn_row_budget(3, minimum=8) == 8
    assert gn_row_budget(1000, maximum=500) == 500
    with pytest.raises(ValueError):
        gn_row_budget(0)
    g1, g2 = torch.Generator().manual_seed(0), torch.Generator().manual_seed(0)
    a, b = select_rows(100, 10, generator=g1), select_rows(100, 10, generator=g2)
    assert torch.equal(a, b) and bool((a[1:] > a[:-1]).all())
    assert torch.equal(select_rows(7, 10), torch.arange(7))


def test_ravel_round_trip_matches_the_leaf_order():
    p = {"b": torch.arange(3.0), "a": [torch.ones(2, 2), torch.zeros(1)]}
    flat, unravel = ravel_params(p)
    assert flat.shape == (8,)
    back = unravel(flat * 2)
    torch.testing.assert_close(back["a"][0], 2 * torch.ones(2, 2))
    torch.testing.assert_close(back["b"], 2 * torch.arange(3.0))


def test_cg_matches_a_direct_solve_with_and_without_host_checks():
    g = np.random.default_rng(1)
    M = g.standard_normal((30, 30))
    S = torch.tensor(M @ M.T + 0.5 * np.eye(30))
    b = torch.tensor(g.standard_normal(30))
    ref = np.linalg.solve(S.numpy(), b.numpy())
    for check in (10, 0):
        x, k = cg_solve(lambda v: S @ v, b, tol=1e-12, maxiter=100, check_every=check)
        np.testing.assert_allclose(x.numpy(), ref, rtol=1e-8, atol=1e-10)
        assert 0 < int(k) <= 100


@pytest.mark.parametrize("shape", [(20, 50), (60, 15)])
def test_dense_and_cg_agree_with_the_normal_equations(shape):
    f, th = _problem(*shape)
    lam = 1e-2
    r, dense = gauss_newton_solve(f, th, lam, solver="dense")
    _, cg = gauss_newton_solve(f, th, lam, solver="cg", cg_tol=1e-12, cg_maxiter=400)
    J = _jac(f, th).numpy()
    exact = np.linalg.solve(J.T @ J + lam * np.eye(J.shape[1]), J.T @ r.numpy())
    np.testing.assert_allclose(dense.delta.numpy(), exact, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(cg.delta.numpy(), exact, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(dense.jdelta.numpy(), J @ exact, rtol=1e-8, atol=1e-10)
    assert int(cg.iterations) > 0 and int(dense.iterations) == 0


def test_auto_solver_switches_on_the_row_budget():
    f, th = _problem(40, 10)
    _, small = gauss_newton_solve(f, th, 1e-2, dense_rows=64)
    _, big = gauss_newton_solve(f, th, 1e-2, dense_rows=8)
    assert int(small.method) in (0, 1) and int(big.method) in (2, 3)
    torch.testing.assert_close(small.delta, big.delta, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("solver", ["dense", "cg"])
def test_spring_momentum_minimises_the_regularised_least_squares(solver):
    f, th = _problem(25, 40)
    prev = torch.tensor(np.random.default_rng(3).standard_normal(40))
    mu, lam = 0.7, 5e-2
    r, sol = gauss_newton_solve(
        f, th, lam, solver=solver, momentum=mu, prev_delta=prev, cg_tol=1e-12, cg_maxiter=500
    )
    J = _jac(f, th).numpy()
    exact = np.linalg.solve(J.T @ J + lam * np.eye(40), J.T @ r.numpy() + lam * mu * prev.numpy())
    np.testing.assert_allclose(sol.delta.numpy(), exact, rtol=1e-6, atol=1e-8)
    with pytest.raises(ValueError, match="prev_delta"):
        gauss_newton_solve(f, th, lam, momentum=0.5)


def _regression():
    g = np.random.default_rng(5)
    X = torch.tensor(g.uniform(-1, 1, (80, 2)))
    Xh = torch.tensor(g.uniform(-1, 1, (40, 2)))
    target = lambda x: torch.sin(2.0 * x[:, 0]) * torch.cos(x[:, 1])  # noqa: E731
    p0 = {
        "W": torch.tensor(0.5 * g.standard_normal((2, 10))),
        "b": torch.zeros(10),
        "v": torch.tensor(0.3 * g.standard_normal(10)),
    }

    def model(p, x):
        return torch.tanh(x @ p["W"] + p["b"]) @ p["v"]

    return p0, (lambda p: model(p, X) - target(X)), (lambda p: model(p, Xh) - target(Xh))


@pytest.mark.parametrize("config", [LMConfig(), LMConfig(solver="cg"), LMConfig(momentum=0.3)])
def test_lm_step_reduces_heldout_error_and_adapts_damping(config):
    p, res, held = _regression()
    state = init_lm_state(p, 1e-1)
    first = last = None
    accepted = 0
    for _ in range(30):
        p, state, info = lm_step(res, p, state, heldout=held, config=config)
        first = first if first is not None else float(info["before"])
        if bool(info["accepted"]):
            last = float(info["after"])
        accepted += int(info["accepted"])
    assert accepted >= 5 and last is not None and last < 0.05 * first
    assert float(state.lam) < 1e-1


def test_lm_rejects_a_step_that_hurts_the_heldout_set_and_raises_damping():
    train = lambda p: p - 1.0  # noqa: E731
    held = lambda p: p + 1.0  # noqa: E731
    p0 = torch.zeros(3)
    new, state, info = lm_step(train, p0, init_lm_state(p0, 1e-2), heldout=held)
    assert not bool(info["accepted"])
    torch.testing.assert_close(new, p0)
    assert float(state.lam) > 1e-2 and float(state.prev_delta.abs().sum()) == 0.0
    new2, _, info2 = lm_step(train, p0, init_lm_state(p0, 1e-2))
    assert bool(info2["accepted"]) and float((new2 - 1.0).abs().max()) < 0.1


def test_grid_line_search_is_deterministic_and_loop_equals_vmap():
    loss = lambda th: torch.sum((th - 3.0) ** 2)  # noqa: E731
    th, delta = torch.zeros(4), -torch.full((4,), 3.0) * 2.0
    new, k, evals = grid_line_search(loss, th, delta, n_scales=8)
    assert int(k) == 1 and evals == 9
    torch.testing.assert_close(new, torch.full((4,), 3.0))
    new_l, k_l, _ = grid_line_search(loss, th, delta, n_scales=8, vectorize=False)
    assert int(k_l) == 1 and torch.equal(new, new_l)
    stay, k_u, _ = grid_line_search(loss, th, torch.full((4,), 3.0), n_scales=8)
    assert int(k_u) == 8 and torch.equal(stay, th)


def test_engd_step_decreases_the_loss():
    p, res, _ = _regression()
    l0 = float(0.5 * (res(p) ** 2).sum())
    for _ in range(8):
        p, info = engd_step(res, p, 1e-3, n_scales=12)
    assert float(0.5 * (res(p) ** 2).sum()) < 0.1 * l0
    assert info["evals"] == 13


def test_lowrank_and_kfac_first_layer_match_the_broadcast_formulation():
    g = np.random.default_rng(2)
    n, m = 40, 7
    U = torch.tensor(g.standard_normal((n, m)))
    c = torch.tensor(g.uniform(0.5, 3.0, m))
    rhs = torch.tensor(g.standard_normal((n, 5)))
    dense = np.linalg.solve(
        0.3 * np.eye(n) + U.numpy() @ np.diag(c.numpy()) @ U.numpy().T, rhs.numpy()
    )
    np.testing.assert_allclose(
        lowrank_damped_solve(U, c, 0.3, rhs).numpy(), dense, rtol=1e-9, atol=1e-11
    )

    g = np.random.default_rng(4)
    B, n_in, k, out = 12, 30, 3, 9
    X, Om = g.standard_normal((B, n_in)), g.standard_normal((k, n_in))
    Xa = np.concatenate([X, np.ones((B, 1))], axis=1)
    Oa = np.concatenate([Om, np.zeros((k, 1))], axis=1)
    rows = np.concatenate([Xa, np.broadcast_to(Oa, (B, k, n_in + 1)).reshape(-1, n_in + 1)], axis=0)
    A = rows.T @ rows
    G = (lambda Q: Q @ Q.T)(g.standard_normal((out, 25)))
    grad = g.standard_normal((out, n_in + 1))
    N, lam = B * (1 + k), 1e-2
    root = np.sqrt(lam * N)
    pi = np.sqrt(np.trace(A) / A.shape[0] / (np.trace(G) / G.shape[0]))
    left = np.linalg.solve(G + (root / pi) * np.eye(out), grad)
    ref = np.linalg.solve(A + pi * root * np.eye(n_in + 1), left.T).T * N
    Ut = torch.tensor(np.concatenate([Xa.T, Oa.T], axis=1))
    ct = torch.tensor(np.concatenate([np.ones(B), np.full(k, float(B))]))
    got = kron_damped_direction(Ut, ct, torch.tensor(G), N, lam, torch.tensor(grad))
    np.testing.assert_allclose(got.numpy(), ref, rtol=1e-8, atol=1e-10)


def test_lbfgs_reports_line_search_evaluations():
    g = np.random.default_rng(6)
    M = torch.tensor(g.standard_normal((6, 6)))
    S = M @ M.T + torch.eye(6)
    loss = lambda th: 0.5 * th @ S @ th - th.sum()  # noqa: E731
    th, hist = lbfgs_with_counts(loss, torch.zeros(6), steps=10)
    np.testing.assert_allclose(
        th.numpy(), np.linalg.solve(S.numpy(), np.ones(6)), rtol=1e-5, atol=1e-6
    )
    assert len(hist["evaluations"]) == 10 and hist["total"] >= 10 and min(hist["evaluations"]) >= 1
