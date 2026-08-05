# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Contract tests for :mod:`omnibias.curvature.glm_fisher`.

The multi-parameter GLM Fisher must equal the autodiff Fisher-scoring matrix
``F = (1/B) sum_n A''(eta_n) g_n g_n^T`` (with ``eta_n`` the field output and
``g_n`` its closed-form parameter gradient), reduce to the Gauss-Newton Fisher
for the Gaussian family, and be positive semidefinite. The natural-coordinate
Fisher-Rao metric must equal ``diag(A''(eta_k))`` and reduce to the scalar 1-D
Fisher information.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from omnibias.curvature.glm_fisher import (  # noqa: E402
    fisher_information_metric,
    glm_fisher,
)
from omnibias.curvature.one_layer import (  # noqa: E402
    mse_gauss_newton_fisher,
    pack_params,
    unpack_params,
)
from omnibias.jax.activations import get_activation  # noqa: E402
from omnibias.jax.information import fisher_information, glm_variance  # noqa: E402

_RICCATI = ("tanh", "sigmoid", "softplus", "gaussian", "exp")
_FAMILIES = ("bernoulli", "poisson", "gaussian")


def _rand_problem(D=4, H=3, B=10, seed=0):
    rng = np.random.default_rng(seed)
    W = jnp.asarray(rng.normal(scale=0.3, size=(H, D)))
    beta = jnp.asarray(rng.normal(scale=0.2, size=(H,)))
    c = jnp.asarray(rng.normal(scale=0.4, size=(H,)))
    b = jnp.asarray(0.13)
    X = jnp.asarray(rng.normal(size=(B, D)))
    return X, W, beta, c, b


def _field_from_flat(theta, x, H, D, activation):
    b, c, beta, W = unpack_params(theta, H=H, D=D)
    sigma = get_activation(activation).forward(W @ x + beta)
    return b + sigma @ c


def _variance_fn(eta, family):
    if family == "gaussian":
        return jnp.ones_like(eta)
    if family == "bernoulli":
        s = jax.nn.sigmoid(eta)
        return s * (1.0 - s)
    if family == "poisson":
        return jnp.exp(eta)
    raise AssertionError(family)


def _glm_fisher_autodiff_oracle(X, W, beta, c, b, activation, family):
    H, D = W.shape
    theta = pack_params(b, c, beta, W)

    def f(th, x):
        return _field_from_flat(th, x, H, D, activation)

    etas = jax.vmap(lambda x: f(theta, x))(X)
    gs = jax.vmap(lambda x: jax.grad(lambda th: f(th, x))(theta))(X)
    w = _variance_fn(etas, family)
    return (gs.T * w) @ gs / X.shape[0]


@pytest.mark.parametrize("activation", _RICCATI)
@pytest.mark.parametrize("family", _FAMILIES)
def test_glm_fisher_matches_autodiff_oracle(activation: str, family: str) -> None:
    X, W, beta, c, b = _rand_problem(seed=1)
    F = glm_fisher(X, W, beta, c, b, activation=activation, family=family)
    F_oracle = _glm_fisher_autodiff_oracle(X, W, beta, c, b, activation, family)
    assert F.shape == F_oracle.shape
    assert jnp.allclose(F, F_oracle, rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize("activation", _RICCATI)
def test_glm_fisher_is_symmetric_psd(activation: str) -> None:
    X, W, beta, c, b = _rand_problem(seed=2)
    F = glm_fisher(X, W, beta, c, b, activation=activation, family="bernoulli")
    assert jnp.allclose(F, F.T, atol=1e-12)
    assert float(jnp.linalg.eigvalsh(F).min()) >= -1e-9


def test_gaussian_family_is_half_the_gauss_newton_fisher() -> None:
    # GLM Fisher (A'' = 1) = (1/B) sum g g^T; MSE Gauss-Newton uses 2/B.
    X, W, beta, c, b = _rand_problem(seed=3)
    Y = jnp.zeros(X.shape[0])
    F_glm = glm_fisher(X, W, beta, c, b, activation="tanh", family="gaussian")
    F_mse, _ = mse_gauss_newton_fisher(X, Y, W, beta, c, b, "tanh")
    assert jnp.allclose(F_glm, 0.5 * F_mse, rtol=1e-10, atol=1e-12)


def test_glm_fisher_rejects_unknown_family() -> None:
    X, W, beta, c, b = _rand_problem()
    with pytest.raises(ValueError, match="unknown GLM family"):
        glm_fisher(X, W, beta, c, b, family="weibull")


# ----- natural-coordinate Fisher-Rao metric ---------------------------------


def test_fisher_information_metric_is_diag_of_variance() -> None:
    eta = jnp.asarray([0.3, -0.7, 1.2])
    g = fisher_information_metric(eta, base="softplus")
    assert g.shape == (3, 3)
    assert jnp.allclose(g, jnp.diag(glm_variance(eta, base="softplus")))


def test_fisher_information_metric_batches() -> None:
    eta = jnp.asarray([[0.1, 0.2], [-0.3, 0.5], [1.0, -1.0]])
    g = fisher_information_metric(eta, base="softplus")
    assert g.shape == (3, 2, 2)
    for k in range(3):
        assert jnp.allclose(g[k], jnp.diag(glm_variance(eta[k], base="softplus")))


def test_one_dim_metric_equals_scalar_fisher_information() -> None:
    eta = jnp.asarray([0.42])
    g = fisher_information_metric(eta, base="softplus")
    assert g.shape == (1, 1)
    assert float(g[0, 0]) == pytest.approx(float(fisher_information(eta, base="softplus")[0]))
