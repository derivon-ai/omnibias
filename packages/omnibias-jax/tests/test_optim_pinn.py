# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Row budget, dense/matrix-free Gauss-Newton, SPRING, held-out Levenberg-Marquardt, ENGD, KFAC and L-BFGS counts."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from omnibias.jax.optim_pinn import (
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
    select_rows,
)

jax.config.update("jax_enable_x64", True)


def _problem(n_rows, n_par, seed=0):
    g = np.random.default_rng(seed)
    A = jnp.asarray(g.standard_normal((n_rows, n_par)))
    y = jnp.asarray(g.standard_normal(n_rows))

    def f(theta):  # a mildly nonlinear residual
        return jnp.tanh(A @ theta) - 0.3 * y

    return f, jnp.asarray(0.2 * g.standard_normal(n_par))


def test_row_budget_scales_with_parameters_and_memory():
    assert gn_row_budget(100) == 200
    assert gn_row_budget(10_000, memory_bytes=2**30) < 20_000  # memory caps the 2x rule
    big = gn_row_budget(10_000, memory_bytes=8 * 2**30)
    assert gn_row_budget(10_000, memory_bytes=2**30) < big <= 20_000
    assert gn_row_budget(3) == 64 and gn_row_budget(3, minimum=8) == 8
    assert gn_row_budget(1000, maximum=500) == 500
    assert gn_row_budget(10, dtype=jnp.float32) == gn_row_budget(10)
    for bad in ({"n_params": 0}, {"n_params": 5, "memory_bytes": 0}):
        with pytest.raises(ValueError):
            gn_row_budget(**bad)


def test_select_rows_is_reproducible_sorted_and_complete_when_it_fits():
    k = jax.random.PRNGKey(0)
    a, b = select_rows(k, 100, 10), select_rows(k, 100, 10)
    np.testing.assert_array_equal(a, b)
    assert np.all(np.diff(np.asarray(a)) > 0) and len(set(np.asarray(a).tolist())) == 10
    np.testing.assert_array_equal(select_rows(k, 7, 10), np.arange(7))


def test_cg_matches_a_direct_solve_and_reports_iterations():
    g = np.random.default_rng(1)
    M = g.standard_normal((30, 30))
    S = jnp.asarray(M @ M.T + 0.5 * np.eye(30))
    b = jnp.asarray(g.standard_normal(30))
    x, k = cg_solve(lambda v: S @ v, b, tol=1e-12, maxiter=100)
    np.testing.assert_allclose(
        x, np.linalg.solve(np.asarray(S), np.asarray(b)), rtol=1e-8, atol=1e-10
    )
    assert 0 < int(k) <= 100


@pytest.mark.parametrize("shape", [(20, 50), (60, 15)])  # dual (R<P) and primal (R>P)
def test_dense_and_cg_solutions_agree_with_the_normal_equations(shape):
    f, th = _problem(*shape)
    lam = 1e-2
    r, dense = gauss_newton_solve(f, th, lam, solver="dense")
    _, cg = gauss_newton_solve(f, th, lam, solver="cg", cg_tol=1e-12, cg_maxiter=400)
    J = np.asarray(jax.jacfwd(f)(th))
    exact = np.linalg.solve(J.T @ J + lam * np.eye(J.shape[1]), J.T @ np.asarray(r))
    np.testing.assert_allclose(dense.delta, exact, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(cg.delta, exact, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(dense.jdelta, J @ exact, rtol=1e-8, atol=1e-10)
    assert int(cg.iterations) > 0 and int(dense.iterations) == 0


def test_auto_solver_switches_on_the_row_budget():
    f, th = _problem(40, 10)
    _, small = gauss_newton_solve(f, th, 1e-2, dense_rows=64)
    _, big = gauss_newton_solve(f, th, 1e-2, dense_rows=8)
    assert int(small.method) in (0, 1) and int(big.method) in (2, 3)
    np.testing.assert_allclose(small.delta, big.delta, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("solver", ["dense", "cg"])
def test_spring_momentum_minimises_the_regularised_least_squares(solver):
    f, th = _problem(25, 40)
    g = np.random.default_rng(3)
    prev = jnp.asarray(g.standard_normal(40))
    mu, lam = 0.7, 5e-2
    r, sol = gauss_newton_solve(
        f, th, lam, solver=solver, momentum=mu, prev_delta=prev, cg_tol=1e-12, cg_maxiter=500
    )
    J = np.asarray(jax.jacfwd(f)(th))
    exact = np.linalg.solve(
        J.T @ J + lam * np.eye(40), J.T @ np.asarray(r) + lam * mu * np.asarray(prev)
    )
    np.testing.assert_allclose(sol.delta, exact, rtol=1e-6, atol=1e-8)
    with pytest.raises(ValueError, match="prev_delta"):
        gauss_newton_solve(f, th, lam, momentum=0.5)


def _regression():
    g = np.random.default_rng(5)
    X = jnp.asarray(g.uniform(-1, 1, (80, 2)))
    Xh = jnp.asarray(g.uniform(-1, 1, (40, 2)))
    target = lambda x: jnp.sin(2.0 * x[:, 0]) * jnp.cos(x[:, 1])  # noqa: E731
    p0 = {
        "W": jnp.asarray(0.5 * g.standard_normal((2, 10))),
        "b": jnp.zeros(10),
        "v": jnp.asarray(0.3 * g.standard_normal(10)),
    }

    def model(p, x):
        return jnp.tanh(x @ p["W"] + p["b"]) @ p["v"]

    res = lambda p, key: model(p, X) - target(X)  # noqa: E731
    held = lambda p, key: model(p, Xh) - target(Xh)  # noqa: E731
    return p0, res, held


@pytest.mark.parametrize("config", [LMConfig(), LMConfig(solver="cg"), LMConfig(momentum=0.3)])
def test_lm_step_reduces_heldout_error_and_adapts_damping(config):
    p, res, held = _regression()
    state = init_lm_state(p, 1e-1)
    step = jax.jit(lambda p_, s_, k_: lm_step(res, p_, s_, k_, heldout=held, config=config))
    first = last = None
    accepted = 0
    for i in range(30):
        p, state, info = step(p, state, jax.random.PRNGKey(i))
        first = first if first is not None else float(info["before"])
        last = float(info["after"]) if bool(info["accepted"]) else last
        accepted += int(info["accepted"])
    assert accepted >= 5 and last is not None and last < 0.05 * first
    assert float(state.lam) < 1e-1  # successful steps lower the damping


def test_lm_rejects_a_step_that_hurts_the_heldout_set_and_raises_damping():
    train = lambda p, k: p - 1.0  # minimum at 1  # noqa: E731
    held = lambda p, k: p + 1.0  # minimum at -1: moving towards 1 hurts  # noqa: E731
    p0 = jnp.zeros(3)
    new, state, info = lm_step(
        train, p0, init_lm_state(p0, 1e-2), jax.random.PRNGKey(0), heldout=held
    )
    assert not bool(info["accepted"])
    np.testing.assert_array_equal(new, p0)
    assert float(state.lam) > 1e-2 and float(jnp.sum(jnp.abs(state.prev_delta))) == 0.0
    # without a held-out set the classical training-batch rule accepts
    new2, _, info2 = lm_step(train, p0, init_lm_state(p0, 1e-2), jax.random.PRNGKey(0))
    assert bool(info2["accepted"]) and float(jnp.max(jnp.abs(new2 - 1.0))) < 0.1


def test_lm_step_is_traceable_in_a_scan():
    p, res, held = _regression()

    def body(carry, i):
        p_, s_ = carry
        p_, s_, info = lm_step(res, p_, s_, jax.random.PRNGKey(0) + i, heldout=held)
        return (p_, s_), info["after"]

    (_, _), afters = jax.lax.scan(
        body, (p, init_lm_state(p, 1e-1)), jnp.arange(10, dtype=jnp.uint32)
    )
    assert afters.shape == (10,) and float(afters[-1]) < float(afters[0])


def test_grid_line_search_is_deterministic_and_vmapped_equals_chunked():
    loss = lambda th: jnp.sum((th - 3.0) ** 2)  # noqa: E731
    th, delta = jnp.zeros(4), -jnp.full(4, 3.0) * 2.0  # th - s*delta = 6 s: best s = 0.5 -> k = 1
    new, k, evals = grid_line_search(loss, th, delta, n_scales=8)
    assert int(k) == 1 and int(evals) == 9
    np.testing.assert_allclose(new, 3.0)
    new_c, k_c, _ = grid_line_search(loss, th, delta, n_scales=8, chunk=3)
    assert int(k_c) == 1
    np.testing.assert_array_equal(new, new_c)
    uphill = jnp.full(4, 3.0)  # theta - s * delta moves away from the minimum
    stay, k_u, _ = grid_line_search(loss, th, uphill, n_scales=8)
    assert int(k_u) == 8
    np.testing.assert_array_equal(stay, th)
    again, k_a, _ = grid_line_search(loss, th, delta, n_scales=8)
    assert int(k_a) == int(k)
    np.testing.assert_array_equal(again, new)


def test_engd_step_decreases_the_loss():
    p, res, _ = _regression()
    key = jax.random.PRNGKey(0)
    l0 = float(0.5 * jnp.sum(res(p, key) ** 2))
    for _ in range(8):
        p, info = engd_step(res, p, key, 1e-3, n_scales=12)
    assert float(0.5 * jnp.sum(res(p, key) ** 2)) < 0.1 * l0
    assert int(info["evals"]) == 13


def test_lowrank_damped_solve_matches_the_dense_inverse():
    g = np.random.default_rng(2)
    n, m = 40, 7
    U = jnp.asarray(g.standard_normal((n, m)))
    c = jnp.asarray(g.uniform(0.5, 3.0, m))
    rhs = jnp.asarray(g.standard_normal((n, 5)))
    d = 0.3
    dense = np.linalg.solve(
        d * np.eye(n) + np.asarray(U) @ np.diag(np.asarray(c)) @ np.asarray(U).T, np.asarray(rhs)
    )
    np.testing.assert_allclose(lowrank_damped_solve(U, c, d, rhs), dense, rtol=1e-9, atol=1e-11)


def test_kfac_first_layer_direction_matches_the_broadcast_formulation():
    """A = [X 1]^T [X 1] + B * Omega^T Omega: the batch-repeated rows enter once, weighted by B."""
    g = np.random.default_rng(4)
    B, n_in, k, out = 12, 30, 3, 9
    X = g.standard_normal((B, n_in))
    Om = g.standard_normal((k, n_in))
    Xa = np.concatenate([X, np.ones((B, 1))], axis=1)
    Oa = np.concatenate([Om, np.zeros((k, 1))], axis=1)
    rows = np.concatenate(
        [Xa, np.broadcast_to(Oa, (B, k, n_in + 1)).reshape(-1, n_in + 1)], axis=0
    )  # B-fold broadcast
    A = rows.T @ rows
    G = (lambda Q: Q @ Q.T)(g.standard_normal((out, 25)))
    grad = g.standard_normal((out, n_in + 1))
    N, lam = B * (1 + k), 1e-2
    root = np.sqrt(lam * N)
    pi = np.sqrt(np.trace(A) / A.shape[0] / (np.trace(G) / G.shape[0]))
    left = np.linalg.solve(G + (root / pi) * np.eye(out), grad)
    ref = np.linalg.solve(A + pi * root * np.eye(n_in + 1), left.T).T * N

    U = jnp.asarray(np.concatenate([Xa.T, Oa.T], axis=1))  # (n_in+1, B+k): distinct rows only
    c = jnp.asarray(np.concatenate([np.ones(B), np.full(k, float(B))]))
    got = kron_damped_direction(U, c, jnp.asarray(G), N, lam, jnp.asarray(grad))
    np.testing.assert_allclose(got, ref, rtol=1e-8, atol=1e-10)
    assert U.shape == (n_in + 1, B + k)  # never an (n_in+1) x (n_in+1) factor, never B*k rows


def test_lbfgs_reports_line_search_counts():
    pytest.importorskip("optax")
    g = np.random.default_rng(6)
    M = jnp.asarray(g.standard_normal((6, 6)))
    S = M @ M.T + jnp.eye(6)
    loss = lambda th: 0.5 * th @ S @ th - jnp.sum(th)  # noqa: E731
    p, hist = lbfgs_with_counts(loss, jnp.zeros(6), steps=60)
    ref = np.linalg.solve(np.asarray(S), np.ones(6))
    np.testing.assert_allclose(p, ref, rtol=1e-4, atol=1e-5)
    assert hist["linesearch_steps"].shape == (60,) and int(hist["total"]) >= 60
    assert float(hist["loss"][-1]) <= float(hist["loss"][0])


# ----------------------------------------------------------------------- higher-order and structured residual rows
def _mlp(sizes, seed=0, act="tanh"):
    from jax.flatten_util import ravel_pytree

    g = np.random.default_rng(seed)
    params = [
        (
            jnp.asarray(g.standard_normal((o, i)) / np.sqrt(i)),
            jnp.asarray(0.1 * g.standard_normal(o)),
        )
        for i, o in zip(sizes[:-1], sizes[1:], strict=True)
    ]
    flat, unravel = ravel_pytree(params)

    def spec(theta):
        p = unravel(theta)
        return [(W, b, act if i < len(p) - 1 else None) for i, (W, b) in enumerate(p)]

    return flat, spec


def _check_gn(f, theta, lam=1e-2):
    r, dense = gauss_newton_solve(f, theta, lam, solver="dense")
    _, cg = gauss_newton_solve(f, theta, lam, solver="cg", cg_tol=1e-12, cg_maxiter=600)
    J = np.asarray(jax.jacfwd(f)(theta))
    exact = np.linalg.solve(J.T @ J + lam * np.eye(J.shape[1]), J.T @ np.asarray(r))
    np.testing.assert_allclose(dense.delta, exact, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(cg.delta, exact, rtol=1e-5, atol=1e-7)


def test_third_order_kdv_residual_rows_through_the_stream_engine():
    """``u_t + u u_x + u_xxx``: third derivatives from the order-3 streams, exact rows to every parameter."""
    from omnibias.jax.experimental.streams import partial_derivatives

    theta, spec = _mlp([2, 12, 12, 1], seed=4)
    x = jnp.asarray(np.random.default_rng(5).uniform(-1, 1, (14, 2)))

    def kdv(th):
        u, d = partial_derivatives(x, spec(th), [(0, 1), (1, 0), (3, 0)])
        return u[:, 0] * d[:, 1, 0] + d[:, 0, 0] + d[:, 2, 0]

    _check_gn(kdv, theta)


def test_composed_sixth_order_residual_rows_through_the_trilaplacian():
    from omnibias.jax.experimental.projected_polylaplacian import projected_deep_trilaplacian

    theta, spec = _mlp([3, 10, 10, 1], seed=6)
    x = jnp.asarray(np.random.default_rng(7).uniform(-1, 1, (9, 3)))

    def residual(th):
        _, _, _, _, tri, _ = projected_deep_trilaplacian(x, spec(th))
        return (
            tri[:, 0] / 1e3
        )  # |tri| ~ 1e3 here; the scaling keeps the normal equations well conditioned

    _check_gn(residual, theta)


def test_divergence_row_program_does_not_grow_with_the_row_count():
    """The rows come from ``lax.map`` over cotangent chunks: the lowered program is the same size for 32 and 128 points."""
    from omnibias.jax.experimental.projected_divergence import projected_deep_divergence
    from omnibias.jax.optim_pinn import _dense_jacobian

    theta, spec = _mlp([6, 16, 16, 6], seed=8)
    sizes = {}
    for B in (32, 128):
        x = jnp.asarray(np.random.default_rng(9).uniform(-1, 1, (B, 6)))

        def rows(
            th, pts, n=B
        ):  # the points are an argument: a closed-over array would be a literal in the program
            return _dense_jacobian(
                lambda t: projected_deep_divergence(pts, spec(t))[1].reshape(-1), th, n, 16
            )

        sizes[B] = len(jax.jit(rows).lower(theta, x).as_text())
    assert sizes[128] < 1.15 * sizes[32], sizes
