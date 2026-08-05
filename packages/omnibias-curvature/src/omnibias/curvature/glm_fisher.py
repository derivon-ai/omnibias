# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Multi-parameter Fisher information for one-layer GLM fields (JAX).

Two complementary objects lift the *scalar* Fisher information
``A''(theta)`` (``omnibias.jax.information.fisher_information``, the exact 1-D
Fisher-Rao metric) to many parameters:

* :func:`fisher_information_metric` -- the exponential-family **Fisher-Rao
  metric in natural coordinates**: the ``(d, d)`` diagonal matrix
  ``diag(A''(eta_1), ..., A''(eta_d))`` for ``d`` coordinates sharing the scalar
  log-partition ``base``. This is the direct multivariate generalisation of the
  1-D Fisher information; hand it to ``omnibias-geometry`` (see
  :func:`omnibias.geometry.jax.ops.exponential_family_fisher_metric`) to obtain
  its Christoffel symbols and curvature.

* :func:`glm_fisher` -- the ``(P, P)`` Fisher information of the one-layer field
  **parameters** ``theta = (b, c, beta, W)`` under a GLM observation model whose
  natural parameter is the field output ``eta_n = f_theta(x_n)``. It reuses the
  closed-form per-sample gradient :func:`omnibias.curvature.one_layer.one_layer_param_grad`
  and weights each outer product by the GLM variance ``A''(eta_n)``:

  .. math::

      F(\theta) = \frac{1}{B} \sum_n A''(\eta_n)\, g_n g_n^\top,
      \qquad g_n = \nabla_\theta \eta_n .

  This is the Fisher-scoring / IRLS information matrix and generalises
  :func:`omnibias.curvature.one_layer.mse_gauss_newton_fisher` (the Gaussian
  family, ``A'' = 1``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array
from omnibias.curvature.one_layer import one_layer_param_grad
from omnibias.jax.activations import get_activation
from omnibias.jax.information import glm_variance

#: GLM family -> log-partition activation whose 2nd derivative is the variance
#: function ``A''``. ``None`` marks the unit-variance Gaussian family.
_FAMILY_LOG_PARTITION: dict[str, str | None] = {
    "bernoulli": "softplus",
    "poisson": "exp",
    "gaussian": None,
}


def fisher_information_metric(
    eta: Array | float, *, base: str = "softplus"
) -> Array:
    r"""Exponential-family Fisher-Rao metric ``diag(A''(eta_k))`` in natural coords.

    ``eta`` carries the natural parameters on its last axis (shape ``(..., d)``);
    the returned metric has shape ``(..., d, d)`` and is diagonal with the GLM
    variance ``A''(eta_k)`` of the scalar log-partition ``base`` on the diagonal.
    For ``d == 1`` this is the 1-D Fisher information embedded as a ``(1, 1)``
    matrix.
    """
    v = glm_variance(eta, base=base)
    d = v.shape[-1]
    eye = jnp.eye(d, dtype=v.dtype)
    out: Array = v[..., :, None] * eye
    return out


def _glm_weights(eta: Array, family: str) -> Array:
    """Per-sample GLM variance ``A''(eta_n)`` (unit for the Gaussian family)."""
    if family not in _FAMILY_LOG_PARTITION:
        raise ValueError(
            f"unknown GLM family {family!r}; "
            f"choose from {sorted(_FAMILY_LOG_PARTITION)}"
        )
    base = _FAMILY_LOG_PARTITION[family]
    if base is None:
        return jnp.ones_like(eta)
    return glm_variance(eta, base=base)


def glm_fisher(
    X: Array,
    W: Array,
    beta: Array,
    c: Array,
    b: Array,
    *,
    activation: str = "tanh",
    family: str = "bernoulli",
) -> Array:
    r"""``(P, P)`` GLM Fisher information of the one-layer field parameters.

    The field output ``eta_n = b + sigma(W x_n + beta) . c`` is the natural
    parameter of the chosen exponential ``family`` (``"bernoulli"`` /
    ``"poisson"`` / ``"gaussian"``). Returns
    ``F = (1/B) sum_n A''(eta_n) g_n g_n^T`` with ``g_n`` the closed-form
    per-sample parameter gradient -- a positive-semidefinite Fisher matrix.
    """
    spec = get_activation(activation)
    n_batch = X.shape[0]

    def per_sample(x_n: Array) -> tuple[Array, Array]:
        eta = b + spec.forward(W @ x_n + beta) @ c
        g = one_layer_param_grad(x_n, W, beta, c, b, activation)
        return eta, g

    etas, gs = jax.vmap(per_sample)(X)  # (B,), (B, P)
    w = _glm_weights(etas, family)  # (B,)
    fisher: Array = (gs.T * w) @ gs / n_batch
    return fisher


__all__ = [
    "fisher_information_metric",
    "glm_fisher",
]
