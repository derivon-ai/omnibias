# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Natural-gradient (Fisher-scoring) optimisation on the closed-form GLM Fisher.

The natural gradient preconditions the loss gradient by the inverse Fisher
information, :math:`\tilde\nabla = F^{-1}\nabla L`, taking steps that are
invariant to reparametrisation and -- for an exponential family with the
canonical link -- coincide with Fisher scoring / IRLS and Newton's method.

Two generic helpers do the linear algebra on any ``(P, P)`` Fisher and ``(P,)``
gradient:

* :func:`damped_solve` -- the regularised solve ``(F + lambda I)^{-1} grad``.
* :func:`natural_gradient_step` -- the parameter update ``theta - lr (F+lambda I)^{-1} grad``.

and one driver wires them to the **closed-form one-layer GLM Fisher**
(:func:`omnibias.curvature.glm_fisher.glm_fisher`):

* :func:`glm_natural_gradient_step` -- one Fisher-scoring step on a one-layer
  field ``eta = b + sigma(W x + beta) . c`` under a Bernoulli / Poisson /
  Gaussian observation model. The Fisher reuses the exact per-sample gradient
  and the GLM variance ``A''(eta)``; the loss gradient reuses the GLM mean
  ``A'(eta)``, so no autodiff backward pass is needed. For the Gaussian family
  this reduces (at zero damping) to the Gauss-Newton
  :func:`omnibias.curvature.one_layer.mse_newton_step`.
"""

from __future__ import annotations

import jax
from jax import Array
from omnibias.curvature.glm_fisher import _FAMILY_LOG_PARTITION, glm_fisher
from omnibias.curvature.one_layer import (
    one_layer_param_grad,
    pack_params,
    unpack_params,
)
from omnibias.curvature.regularize import regularized_solve
from omnibias.jax.activations import get_activation
from omnibias.jax.information import glm_mean


def damped_solve(fisher: Array, grad: Array, *, damping: float = 1e-3) -> Array:
    r"""Natural-gradient direction ``delta = (F + damping I)^{-1} grad``.

    Solves the (Tikhonov-)regularised linear system for the preconditioned
    gradient. ``fisher`` is a ``(P, P)`` positive-semidefinite matrix and ``grad``
    a ``(P,)`` vector; ``damping >= 0`` keeps the system well-posed when ``F`` is
    singular or ill-conditioned.
    """
    if fisher.ndim != 2 or fisher.shape[0] != fisher.shape[1]:
        raise ValueError(
            f"fisher must be a square (P, P) matrix, got {tuple(fisher.shape)}"
        )
    if grad.ndim != 1 or grad.shape[0] != fisher.shape[0]:
        raise ValueError(
            f"grad must be (P,) with P = {fisher.shape[0]}, got {tuple(grad.shape)}"
        )
    if damping < 0.0:
        raise ValueError(f"damping must be >= 0, got {damping}")
    # Delegates to the shared eps -> 0 collapse kernel; bit-identical at this damping.
    delta: Array = regularized_solve(fisher, grad, eps=damping)
    return delta


def natural_gradient_step(
    theta: Array,
    grad: Array,
    fisher: Array,
    *,
    learning_rate: float = 1.0,
    damping: float = 1e-3,
) -> Array:
    r"""Preconditioned parameter update ``theta - lr (F + damping I)^{-1} grad``.

    The metric-aware (natural-gradient) counterpart of vanilla gradient descent:
    descending along ``F^{-1} grad`` instead of ``grad`` makes the step invariant
    to smooth reparametrisations. ``theta`` and ``grad`` are flat ``(P,)`` vectors
    and ``fisher`` the ``(P, P)`` Fisher information.
    """
    delta = damped_solve(fisher, grad, damping=damping)
    out: Array = theta - learning_rate * delta
    return out


def glm_loss_gradient(
    X: Array,
    Y: Array,
    W: Array,
    beta: Array,
    c: Array,
    b: Array,
    *,
    activation: str = "tanh",
    family: str = "bernoulli",
) -> Array:
    r"""Flat ``(P,)`` gradient of the GLM negative log-likelihood.

    For the one-layer natural parameter ``eta_n = b + sigma(W x_n + beta) . c`` the
    exponential-family NLL gradient is ``(1/B) sum_n (A'(eta_n) - y_n) g_n`` with
    ``A'`` the GLM mean (link inverse) and ``g_n`` the closed-form per-sample
    parameter gradient. The residual ``A'(eta_n) - y_n`` is the canonical GLM
    working residual; pair this with :func:`omnibias.curvature.glm_fisher.glm_fisher`
    for a Fisher-scoring step.
    """
    if family not in _FAMILY_LOG_PARTITION:
        raise ValueError(
            f"unknown GLM family {family!r}; "
            f"choose from {sorted(_FAMILY_LOG_PARTITION)}"
        )
    spec = get_activation(activation)
    base = _FAMILY_LOG_PARTITION[family]
    n_batch = X.shape[0]

    def per_sample(x_n: Array) -> tuple[Array, Array]:
        eta = b + spec.forward(W @ x_n + beta) @ c
        g = one_layer_param_grad(x_n, W, beta, c, b, activation)
        return eta, g

    etas, gs = jax.vmap(per_sample)(X)  # (B,), (B, P)
    mu = etas if base is None else glm_mean(etas, base=base)
    g_loss: Array = (gs.T @ (mu - Y)) / n_batch
    return g_loss


def glm_natural_gradient_step(
    X: Array,
    Y: Array,
    W: Array,
    beta: Array,
    c: Array,
    b: Array,
    *,
    activation: str = "tanh",
    family: str = "bernoulli",
    learning_rate: float = 1.0,
    damping: float = 1e-3,
) -> tuple[Array, Array, Array, Array]:
    r"""One natural-gradient (Fisher-scoring / IRLS) step on a one-layer GLM field.

    Updates ``(b, c, beta, W)`` via
    ``theta <- theta - lr (F + damping I)^{-1} g`` with the closed-form GLM Fisher
    ``F`` (:func:`omnibias.curvature.glm_fisher.glm_fisher`) and the GLM NLL
    gradient ``g`` (:func:`glm_loss_gradient`). For ``family="gaussian"`` this is
    Gauss-Newton and, at ``damping = 0``, equals
    :func:`omnibias.curvature.one_layer.mse_newton_step`.

    Returns ``(b_new, c_new, beta_new, W_new)`` with the input shapes.
    """
    g_loss = glm_loss_gradient(
        X, Y, W, beta, c, b, activation=activation, family=family
    )
    fisher = glm_fisher(X, W, beta, c, b, activation=activation, family=family)
    theta = pack_params(b, c, beta, W)
    theta_new = natural_gradient_step(
        theta, g_loss, fisher, learning_rate=learning_rate, damping=damping
    )
    return unpack_params(theta_new, H=W.shape[0], D=W.shape[1])


__all__ = [
    "damped_solve",
    "glm_loss_gradient",
    "glm_natural_gradient_step",
    "natural_gradient_step",
]
