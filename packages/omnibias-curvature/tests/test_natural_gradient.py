# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Natural-gradient / Fisher-scoring optimisation on the closed-form GLM Fisher.

* :func:`damped_solve` solves the regularised normal equations exactly.
* the closed-form :func:`glm_loss_gradient` equals the autodiff GLM NLL gradient.
* the Gaussian-family step equals Gauss-Newton ``mse_newton_step`` at zero damping.
* Fisher scoring converges (quadratically on a realizable zero-residual problem;
  monotone NLL descent for Bernoulli / Poisson).
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from omnibias.curvature.glm_fisher import glm_fisher  # noqa: E402
from omnibias.curvature.natural_gradient import (  # noqa: E402
    damped_solve,
    glm_loss_gradient,
    glm_natural_gradient_step,
    natural_gradient_step,
)
from omnibias.curvature.one_layer import (  # noqa: E402
    mse_newton_step,
    pack_params,
    unpack_params,
)
from omnibias.jax.activations import get_activation  # noqa: E402

_FAMILIES = ("bernoulli", "poisson", "gaussian")
_A_OF = {"bernoulli": jax.nn.softplus, "poisson": jnp.exp, "gaussian": lambda e: 0.5 * e**2}


def _problem(D=3, H=4, B=40, seed=0, activation="tanh"):
    rng = np.random.default_rng(seed)
    X = jnp.asarray(rng.normal(size=(B, D)))
    W = jnp.asarray(rng.normal(scale=0.5, size=(H, D)))
    beta = jnp.asarray(rng.normal(scale=0.3, size=(H,)))
    c = jnp.asarray(rng.normal(scale=0.5, size=(H,)))
    b = jnp.asarray(0.2)
    return X, W, beta, c, b, rng


def _field_eta(X, W, beta, c, b, activation="tanh"):
    spec = get_activation(activation)
    return b + spec.forward(X @ W.T + beta[None, :]) @ c


# ----- generic linear algebra -----------------------------------------------


def test_damped_solve_satisfies_normal_equations() -> None:
    rng = np.random.default_rng(0)
    p = 7
    a = rng.normal(size=(p, p))
    fisher = jnp.asarray(a @ a.T)  # SPD
    grad = jnp.asarray(rng.normal(size=p))
    lam = 1e-2
    delta = damped_solve(fisher, grad, damping=lam)
    resid = (fisher + lam * jnp.eye(p)) @ delta - grad
    assert float(jnp.max(jnp.abs(resid))) < 1e-9


def test_natural_gradient_step_formula() -> None:
    rng = np.random.default_rng(1)
    p = 5
    a = rng.normal(size=(p, p))
    fisher = jnp.asarray(a @ a.T)
    grad = jnp.asarray(rng.normal(size=p))
    theta = jnp.asarray(rng.normal(size=p))
    new = natural_gradient_step(theta, grad, fisher, learning_rate=0.5, damping=1e-3)
    expected = theta - 0.5 * damped_solve(fisher, grad, damping=1e-3)
    assert jnp.allclose(new, expected, rtol=1e-12, atol=1e-12)


def test_damped_solve_guards() -> None:
    with pytest.raises(ValueError, match="square"):
        damped_solve(jnp.zeros((2, 3)), jnp.zeros(2))
    with pytest.raises(ValueError, match="grad must be"):
        damped_solve(jnp.eye(3), jnp.zeros(2))
    with pytest.raises(ValueError, match="damping must be"):
        damped_solve(jnp.eye(3), jnp.zeros(3), damping=-1.0)


# ----- closed-form gradient == autodiff -------------------------------------


@pytest.mark.parametrize("family", _FAMILIES)
def test_glm_loss_gradient_matches_autodiff(family: str) -> None:
    X, W, beta, c, b, rng = _problem(seed=3)
    spec = get_activation("tanh")
    H, D = W.shape
    Y = jnp.asarray(rng.normal(size=X.shape[0]) ** 2 if family == "poisson" else rng.normal(size=X.shape[0]))
    A = _A_OF[family]

    def nll(theta):
        bb, cc, bb2, WW = unpack_params(theta, H=H, D=D)
        eta = bb + spec.forward(X @ WW.T + bb2[None, :]) @ cc
        return jnp.mean(A(eta) - Y * eta)

    theta = pack_params(b, c, beta, W)
    auto = jax.grad(nll)(theta)
    closed = glm_loss_gradient(X, Y, W, beta, c, b, activation="tanh", family=family)
    assert jnp.allclose(auto, closed, rtol=1e-9, atol=1e-9)


def test_glm_loss_gradient_rejects_unknown_family() -> None:
    X, W, beta, c, b, _ = _problem()
    Y = jnp.zeros(X.shape[0])
    with pytest.raises(ValueError, match="unknown GLM family"):
        glm_loss_gradient(X, Y, W, beta, c, b, family="nope")


# ----- equivalence with Gauss-Newton ----------------------------------------


def test_gaussian_step_equals_mse_newton_at_zero_damping() -> None:
    X, W, beta, c, b, rng = _problem(D=3, H=3, B=60, seed=5)
    Y = jnp.asarray(rng.normal(size=X.shape[0]))
    g_b, g_c, g_be, g_W = glm_natural_gradient_step(
        X, Y, W, beta, c, b, family="gaussian", learning_rate=1.0, damping=0.0
    )
    n_b, n_c, n_be, n_W = mse_newton_step(X, Y, W, beta, c, b, learning_rate=1.0, damping=0.0)
    assert jnp.allclose(g_b, n_b, atol=1e-8)
    assert jnp.allclose(g_c, n_c, atol=1e-8)
    assert jnp.allclose(g_be, n_be, atol=1e-8)
    assert jnp.allclose(g_W, n_W, atol=1e-8)


# ----- convergence ----------------------------------------------------------


def test_gaussian_realizable_quadratic_convergence() -> None:
    X, Wt, bt, ct, b0t, rng = _problem(D=2, H=3, B=80, seed=9)
    eta = _field_eta(X, Wt, bt, ct, b0t)  # teacher; zero-residual target
    W = Wt + 0.1 * jnp.asarray(rng.normal(size=Wt.shape))
    beta = bt + 0.1 * jnp.asarray(rng.normal(size=bt.shape))
    c = ct + 0.1 * jnp.asarray(rng.normal(size=ct.shape))
    b = b0t + 0.1
    g0 = float(jnp.linalg.norm(glm_loss_gradient(X, eta, W, beta, c, b, family="gaussian")))
    for _ in range(15):
        b, c, beta, W = glm_natural_gradient_step(
            X, eta, W, beta, c, b, family="gaussian", learning_rate=1.0, damping=1e-10
        )
    gN = float(jnp.linalg.norm(glm_loss_gradient(X, eta, W, beta, c, b, family="gaussian")))
    assert gN < g0 * 1e-4


@pytest.mark.parametrize("family", ["bernoulli", "poisson"])
def test_glm_nll_monotone_decrease(family: str) -> None:
    X, Wt, bt, ct, b0t, rng = _problem(D=2, H=3, B=80, seed=11)
    eta = _field_eta(X, Wt, bt, ct, b0t)
    A = _A_OF[family]
    if family == "bernoulli":
        prob = np.asarray(jax.nn.sigmoid(eta))
        Y = jnp.asarray((rng.random(X.shape[0]) < prob).astype(float))
    else:
        Y = jnp.asarray(rng.poisson(np.asarray(jnp.exp(eta))).astype(float))
    spec = get_activation("tanh")

    def nll(W, beta, c, b):
        e = b + spec.forward(X @ W.T + beta[None, :]) @ c
        return float(jnp.mean(A(e) - Y * e))

    W = jnp.zeros_like(Wt) + 0.1
    beta = jnp.zeros_like(bt)
    c = jnp.zeros_like(ct) + 0.1
    b = jnp.asarray(0.0)
    losses = [nll(W, beta, c, b)]
    for _ in range(20):
        b, c, beta, W = glm_natural_gradient_step(
            X, Y, W, beta, c, b, family=family, learning_rate=0.5, damping=1e-2
        )
        losses.append(nll(W, beta, c, b))
    assert losses[-1] < losses[0]
    assert all(losses[i + 1] <= losses[i] + 1e-9 for i in range(len(losses) - 1))
