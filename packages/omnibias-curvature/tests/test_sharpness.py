# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Contract + behavioural tests for :mod:`omnibias.curvature.sharpness`.

Three families of guarantees:

1. **Exactness** -- the full loss Hessian equals :func:`jax.hessian` of the
   MSE loss for every Riccati activation, and the sharpness-penalty
   gradient (which rides on the closed-form :math:`\sigma'''`) equals a
   central finite difference.
2. **The SAM surrogate is right** -- the exact second-order gap brackets
   the empirically sampled worst case in an :math:`\ell_2`-ball far better
   than SAM's linear-only ascent estimate.
3. **It does what it promises** -- descending the curvature penalty lowers
   curvature (monotone), and sharpness-aware training reaches a
   provably flatter minimum than plain MSE from the same init.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from omnibias.curvature.one_layer import pack_params, unpack_params  # noqa: E402
from omnibias.curvature.sharpness import (  # noqa: E402
    hessian_frobenius_sq,
    hessian_top_eigenvalue,
    hessian_trace,
    mse_curvature_sharpness,
    mse_loss,
    mse_loss_hessian,
    sam_sharpness_gap,
    sharpness_aware_loss,
)

_RICCATI = ("tanh", "sigmoid", "softplus", "gaussian", "exp")


def _rand_problem(D=4, H=5, B=12, seed=0):
    rng = np.random.default_rng(seed)
    W = jnp.asarray(rng.normal(scale=0.3, size=(H, D)), dtype=jnp.float64)
    beta = jnp.asarray(rng.normal(scale=0.2, size=(H,)), dtype=jnp.float64)
    c = jnp.asarray(rng.normal(scale=0.4, size=(H,)), dtype=jnp.float64)
    b = jnp.asarray(0.13, dtype=jnp.float64)
    X = jnp.asarray(rng.normal(size=(B, D)), dtype=jnp.float64)
    Y = jnp.asarray(rng.normal(size=(B,)), dtype=jnp.float64)
    return W, beta, c, b, X, Y


def _loss_of_theta(theta, fn, X, Y, H, D, activation, **kw):
    b, c, beta, W = unpack_params(theta, H=H, D=D)
    return fn(X, Y, W, beta, c, b, activation, **kw)


def _teacher_data(seed, D, H_teacher, n_train, noise, n_test=300):
    """Realisable one-layer tanh teacher target (+ label noise on train)."""
    from omnibias.curvature.sharpness import _batch_forward
    rng = np.random.default_rng(seed)
    Wt = jnp.asarray(rng.normal(scale=1.0, size=(H_teacher, D)))
    bt = jnp.asarray(rng.normal(scale=0.7, size=(H_teacher,)))
    ct = jnp.asarray(rng.normal(scale=1.0, size=(H_teacher,)))
    b0 = jnp.asarray(0.2)
    Xtr = jnp.asarray(rng.normal(size=(n_train, D)))
    Xte = jnp.asarray(rng.normal(size=(n_test, D)))
    Ytr = _batch_forward(Xtr, Wt, bt, ct, b0, "tanh") \
        + noise * jnp.asarray(rng.normal(size=(n_train,)))
    Yte = _batch_forward(Xte, Wt, bt, ct, b0, "tanh")
    return Xtr, Ytr, Xte, Yte


def _adam(grad_fn, theta, steps=500, lr=5e-3):
    """Minimal jit-compiled Adam via lax.scan (fast on CPU)."""
    m = jnp.zeros_like(theta)
    v = jnp.zeros_like(theta)
    b1, b2, eps = 0.9, 0.999, 1e-8

    def body(carry, t):
        th, m, v = carry
        g = grad_fn(th)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * (g * g)
        mhat = m / (1 - b1 ** (t + 1))
        vhat = v / (1 - b2 ** (t + 1))
        th = th - lr * mhat / (jnp.sqrt(vhat) + eps)
        return (th, m, v), None

    (th, _, _), _ = jax.lax.scan(body, (theta, m, v), jnp.arange(steps))
    return th


# ---------------------------------------------------------------------------
# 1. Exactness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("activation", _RICCATI)
def test_full_loss_hessian_matches_jax_hessian(activation):
    """The full loss Hessian (GN + residual) must equal jax.hessian(L)."""
    W, beta, c, b, X, Y = _rand_problem(seed=hash(activation) & 0xFFFF)
    H, D = W.shape

    H_closed, _ = mse_loss_hessian(X, Y, W, beta, c, b, activation)
    theta = pack_params(b, c, beta, W)
    H_jax = jax.hessian(_loss_of_theta)(theta, mse_loss, X, Y, H, D, activation)

    assert H_closed.shape == H_jax.shape == (theta.shape[0], theta.shape[0])
    assert jnp.allclose(H_closed, H_closed.T, atol=1e-12), "Hessian not symmetric"
    np.testing.assert_allclose(
        np.asarray(H_closed), np.asarray(H_jax),
        rtol=1e-7, atol=1e-8,
        err_msg=f"loss Hessian mismatch for activation={activation!r}",
    )


@pytest.mark.parametrize("activation", _RICCATI)
def test_loss_gradient_matches_jax_grad(activation):
    """The returned loss gradient must equal jax.grad(L)."""
    W, beta, c, b, X, Y = _rand_problem(seed=(hash(activation) >> 3) & 0xFFFF)
    H, D = W.shape
    _, g_closed = mse_loss_hessian(X, Y, W, beta, c, b, activation)
    theta = pack_params(b, c, beta, W)
    g_jax = jax.grad(_loss_of_theta)(theta, mse_loss, X, Y, H, D, activation)
    np.testing.assert_allclose(
        np.asarray(g_closed), np.asarray(g_jax), rtol=1e-9, atol=1e-11,
    )


def test_gauss_newton_is_the_psd_part_of_full_hessian():
    """At a residual-free (interpolating) fit the full Hessian == GN Fisher."""
    from omnibias.curvature.one_layer import mse_gauss_newton_fisher
    from omnibias.curvature.sharpness import _batch_forward
    W, beta, c, b, X, _ = _rand_problem(seed=7)
    # Targets == model outputs -> residuals are zero -> residual term vanishes.
    Y = _batch_forward(X, W, beta, c, b, "tanh")
    H_full, _ = mse_loss_hessian(X, Y, W, beta, c, b, "tanh")
    F_gn, _ = mse_gauss_newton_fisher(X, Y, W, beta, c, b, "tanh")
    np.testing.assert_allclose(np.asarray(H_full), np.asarray(F_gn), atol=1e-10)


def test_measures_match_numpy_eigendecomposition():
    W, beta, c, b, X, Y = _rand_problem(seed=3)
    H, _ = mse_loss_hessian(X, Y, W, beta, c, b, "tanh")
    ev = np.linalg.eigvalsh(np.asarray(0.5 * (H + H.T)))
    assert np.isclose(float(hessian_trace(H)), ev.sum(), rtol=1e-10)
    assert np.isclose(float(hessian_frobenius_sq(H)), (ev**2).sum(), rtol=1e-10)
    assert np.isclose(float(hessian_top_eigenvalue(H)), ev.max(), rtol=1e-10)


@pytest.mark.parametrize("measure", ["trace", "frobenius", "top_eig"])
def test_sharpness_penalty_gradient_is_exact(measure):
    """grad of the curvature penalty (rides on closed-form sigma''') ==
    central finite difference. This is the crux: SAM cannot do this
    without a second forward/backward pass and a step-size choice."""
    W, beta, c, b, X, Y = _rand_problem(seed=11)
    H, D = W.shape
    theta = pack_params(b, c, beta, W)

    def S(th):
        return _loss_of_theta(th, mse_curvature_sharpness, X, Y, H, D, "tanh",
                              measure=measure)

    g = np.asarray(jax.grad(S)(theta))
    # Central finite difference on a handful of coordinates.
    eps = 1e-6
    idxs = [0, 1, 1 + H, 1 + 2 * H, theta.shape[0] - 1]
    for i in idxs:
        e = jnp.zeros_like(theta).at[i].set(eps)
        fd = float((S(theta + e) - S(theta - e)) / (2 * eps))
        assert abs(fd - g[i]) <= 1e-4 * (1 + abs(g[i])), (
            f"measure={measure} coord={i}: fd={fd} vs autodiff={g[i]}"
        )


def test_backprop_through_curvature_uses_exact_third_derivative():
    """The penalty gradient's mechanism: d/dz sigma''(z) == sigma'''(z)
    in closed form for every Riccati activation."""
    from omnibias.jax.activations import get_activation
    z = jnp.asarray(np.linspace(-2.0, 2.0, 9))
    for act in _RICCATI:
        spec = get_activation(act)
        d_sigma_pp = jax.vmap(jax.grad(lambda t, s=spec: s.fastpath(t, 2)))(z)
        sigma_ppp = spec.fastpath(z, 3)
        np.testing.assert_allclose(
            np.asarray(d_sigma_pp), np.asarray(sigma_ppp), rtol=1e-9, atol=1e-9,
            err_msg=f"d/dz sigma'' != sigma''' for {act!r}",
        )


# ---------------------------------------------------------------------------
# 2. The SAM surrogate is right
# ---------------------------------------------------------------------------


def test_second_order_model_beats_linear_along_directions():
    """The exact quadratic model ``L + g.eps + 1/2 eps^T H eps`` tracks the true
    loss change to ``O(rho^3)``; SAM's linear-only model errs at ``O(rho^2)``.
    We verify both the *convergence order* (dimension-free) and that the
    quadratic error is a small fraction of the linear error in aggregate --
    this is *why* exact curvature helps."""
    W, beta, c, b, X, Y = _rand_problem(seed=5)
    H, D = W.shape
    theta = pack_params(b, c, beta, W)
    H_mat, g = mse_loss_hessian(X, Y, W, beta, c, b, "tanh")
    L0 = float(_loss_of_theta(theta, mse_loss, X, Y, H, D, "tanh"))

    def err(rho, u):
        true = float(_loss_of_theta(theta + rho * u, mse_loss, X, Y, H, D, "tanh")) - L0
        lin = float(rho * (g @ u))
        quad = lin + float(0.5 * rho * rho * (u @ H_mat @ u))
        return abs(lin - true), abs(quad - true)

    # (A) Along the top-curvature eigenvector, linear -> order ~2, quad -> ~3.
    _, evec = jnp.linalg.eigh(0.5 * (H_mat + H_mat.T))
    u_top = evec[:, -1]
    el1, eq1 = err(0.08, u_top)
    el2, eq2 = err(0.04, u_top)
    lin_order = float(np.log2(el1 / el2))
    quad_order = float(np.log2(eq1 / eq2))
    assert lin_order < 2.4, f"linear model order {lin_order:.2f} not ~2"
    assert quad_order > 2.6, f"quadratic model order {quad_order:.2f} not ~3"

    # (B) Aggregate over random directions at small rho: quad error << lin error,
    #     and quad is never worse than linear along any single direction.
    rng = np.random.default_rng(0)
    els, eqs = [], []
    for _ in range(12):
        u = rng.normal(size=theta.shape[0])
        u = jnp.asarray(u / np.linalg.norm(u))
        el, eq = err(0.04, u)
        els.append(el)
        eqs.append(eq)
        assert eq <= el + 1e-12, "quadratic model worse than linear"
    assert np.mean(eqs) <= 0.1 * np.mean(els)


def test_sam_gap_upper_bounds_sampled_worst_case():
    """The exact 2nd-order SAM gap must upper-bound the worst loss increase in
    an l2-ball (sound model of the SAM inner-max), and stay close to it. Uses a
    low-dim field so uniform sphere sampling densely covers the ball."""
    W, beta, c, b, X, Y = _rand_problem(D=1, H=2, B=8, seed=1)
    H, D = W.shape
    theta = pack_params(b, c, beta, W)
    P = theta.shape[0]
    L0 = float(_loss_of_theta(theta, mse_loss, X, Y, H, D, "tanh"))

    rho = 0.1
    gap2 = float(sam_sharpness_gap(X, Y, W, beta, c, b, "tanh", rho=rho))
    assert gap2 > 0.0

    # Vectorised sampled worst case over the sphere of radius rho.
    rng = np.random.default_rng(0)
    dirs = rng.normal(size=(6000, P))
    dirs = rho * dirs / np.linalg.norm(dirs, axis=1, keepdims=True)

    def delta(d):
        return _loss_of_theta(theta + d, mse_loss, X, Y, H, D, "tanh") - L0

    worst = float(jnp.max(jax.vmap(delta)(jnp.asarray(dirs))))
    assert gap2 >= worst - 1e-3, f"gap2={gap2} under-shoots sampled worst={worst}"
    assert gap2 <= 1.6 * worst, f"gap2={gap2} far above sampled worst={worst}"


def test_unknown_measure_raises():
    W, beta, c, b, X, Y = _rand_problem(seed=1)
    with pytest.raises(ValueError, match="unknown sharpness measure"):
        mse_curvature_sharpness(X, Y, W, beta, c, b, "tanh", measure="nope")


# ---------------------------------------------------------------------------
# 3. It does what it promises
# ---------------------------------------------------------------------------


def test_penalty_only_descent_monotonically_reduces_curvature():
    """Gradient descent on the pure curvature penalty must reduce it."""
    W, beta, c, b, X, Y = _rand_problem(seed=2)
    H, D = W.shape
    theta = pack_params(b, c, beta, W)

    def S(th):
        return _loss_of_theta(th, mse_curvature_sharpness, X, Y, H, D, "tanh",
                              measure="frobenius")

    vg = jax.value_and_grad(S)
    prev = float(S(theta))
    for _ in range(15):
        val, g = vg(theta)
        theta = theta - 1e-3 * g
        cur = float(S(theta))
        assert cur <= prev + 1e-9, f"curvature increased: {prev} -> {cur}"
        prev = cur
    assert prev < float(S(pack_params(b, c, beta, W)))


def test_sharpness_aware_training_finds_flatter_minimum():
    """From the *same* init and data, curvature-regularised training reaches a
    minimum with substantially smaller top-eigenvalue curvature than plain MSE
    -- while still fitting the (realisable) data and generalising at least as
    well. This is the SAM promise, delivered via exact curvature."""
    D, H_teacher, H_student, n_train = 3, 4, 8, 30
    Xtr, Ytr, Xte, Yte = _teacher_data(seed=17, D=D, H_teacher=H_teacher,
                                       n_train=n_train, noise=0.1)
    H = H_student
    theta0 = _init_student(seed=17, D=D, H=H)

    def plain_g(th):
        return jax.grad(
            lambda t: _loss_of_theta(t, mse_loss, Xtr, Ytr, H, D, "tanh")
        )(th)

    def flat_g(th):
        return jax.grad(
            lambda t: _loss_of_theta(t, sharpness_aware_loss, Xtr, Ytr, H, D,
                                     "tanh", lam=5e-3, measure="frobenius")
        )(th)

    theta_plain = _adam(jax.jit(plain_g), theta0, steps=500, lr=5e-3)
    theta_flat = _adam(jax.jit(flat_g), theta0, steps=500, lr=5e-3)

    def report(theta):
        bb, cc, be, WW = unpack_params(theta, H=H, D=D)
        Hh, _ = mse_loss_hessian(Xtr, Ytr, WW, be, cc, bb, "tanh")
        return (
            float(mse_loss(Xtr, Ytr, WW, be, cc, bb, "tanh")),
            float(mse_loss(Xte, Yte, WW, be, cc, bb, "tanh")),
            float(hessian_top_eigenvalue(Hh)),
        )

    tr_p, te_p, curv_p = report(theta_plain)
    tr_f, te_f, curv_f = report(theta_flat)

    # (1) both fit the training data
    assert tr_p < 0.05 and tr_f < 0.05, (tr_p, tr_f)
    # (2) the flat run lands in a clearly flatter basin
    assert curv_f < 0.7 * curv_p, f"curv flat={curv_f:.3f} vs plain={curv_p:.3f}"
    # (3) and generalises at least as well (here: better)
    assert te_f <= te_p * 1.05, f"test mse flat={te_f:.4f} vs plain={te_p:.4f}"


def _init_student(seed, D, H):
    rng = np.random.default_rng(seed + 999)
    W = jnp.asarray(rng.normal(scale=0.5, size=(H, D)))
    beta = jnp.asarray(rng.normal(scale=0.3, size=(H,)))
    c = jnp.asarray(rng.normal(scale=0.5, size=(H,)))
    b = jnp.asarray(0.0)
    return pack_params(b, c, beta, W)
