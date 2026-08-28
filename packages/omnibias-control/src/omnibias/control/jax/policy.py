# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Policy-gradient trainer steps on the ``DifferentiableEnvironment`` seam (JAX,
theory 10-02).

Five comparable single-step trainers, all differentiating through the same
:class:`omnibias.control.ocp.DifferentiableEnvironment` rollout:

* :func:`bptt_step` -- full-horizon backprop-through-time, ordinary
  ``jax.grad`` through the unrolled rollout (no truncation, no bootstrap).
  Gate G1 checks this against :func:`omnibias.control.jax.adjoint.
  actor_adjoint_gradient` to float64 round-off: two independent
  computational paths (plain reverse-mode AD through ``env.step`` vs. the
  closed-form-``dpi/dy`` adjoint recursion) must agree exactly on an exact
  problem.
* :func:`truncated_bptt_step` -- backprop through only the last ``window``
  steps (``jax.lax.stop_gradient`` at the boundary); a biased baseline,
  standing in for the paper's tBPTT arm.
* :func:`zero_order_step` -- antithetic random-shooting (evolution-strategies)
  gradient estimate; the **model-free stand-in** used here for the
  PPO / TD3 comparison point. This is *not* a reimplementation of PPO or
  TD3 (no replay buffer, no critic, no clipped surrogate objective) -- it
  is the simplest model-free zero-order estimator, included so every arm
  in the comparison set differs only in *how much exact structure it uses*,
  from none (zero-order) to all of it (the exact adjoint).
* :func:`actor_adjoint_step` -- **ours**: the full-horizon exact adjoint
  gradient via :func:`omnibias.control.jax.adjoint.actor_adjoint_gradient`
  (closed-form ``dpi/dy`` plus exact environment/parameter Jacobians, no
  truncation, no learned bootstrap).
* :func:`actor_adjoint_jet_step` -- **ours, PEARL-style**: exact adjoint
  recursion over a short ``window``, terminated by a *learned* PSD-quadratic
  terminal adjoint head (:class:`PSDTerminalHead`) instead of the true
  terminal cost gradient -- the value-gradient bootstrap PEARL uses, with a
  closed-form quadratic head rather than a generic value network.

None of these five finds a global optimum of ``J(theta)``; each finds (an
estimate of) a stationary point. ``actor_adjoint_jet_step``'s gradient is
biased whenever the learned head has not converged -- that bias is exactly
what :mod:`omnibias.control.certified.gradient_bias` bounds, conditional on
a supplied error model, not asserted unconditionally here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from omnibias.control.jax.adjoint import (
    ActorAdjointResult,
    Layer,
    actor_adjoint_gradient,
    cost_gradients,
    env_step_jacobians,
    flatten_layers,
    policy_forward,
    policy_jacobian_dtheta,
    policy_jacobian_dy,
    terminal_cost_gradient,
)
from omnibias.control.ocp import DifferentiableEnvironment
from omnibias.core.adjoint import (
    FloatArray,
    adjoint_recursion,
    closed_loop_jacobian,
    policy_gradient,
    total_state_cost_gradient,
)


@dataclass(frozen=True)
class StepResult:
    """Common return type for every trainer step below."""

    layers: list[Layer]
    grad_norm: float
    total_cost: float
    diagnostics: dict[str, Any]


def rollout_cost(env: DifferentiableEnvironment[Array], layers: Sequence[Layer], y0: Array, horizon: int) -> Array:
    """Pure differentiable rollout cost ``sum_k L(y_k, u_k) + Phi(y_H)``."""
    y = y0
    total = jnp.zeros((), dtype=y0.dtype)
    for _ in range(horizon):
        u = policy_forward(layers, y)
        total = total + env.cost(y, u)
        y = env.step(y, u)
    return total + env.terminal_cost(y)


def _apply_grad(theta: Array, grad: Array, lr: float) -> Array:
    return theta - lr * grad


def bptt_step(
    env: DifferentiableEnvironment[Array],
    layers: Sequence[Layer],
    y0: Array,
    horizon: int,
    lr: float,
) -> StepResult:
    """Full-horizon ``jax.grad`` through the unrolled rollout (gate G1 reference)."""
    theta, unravel = flatten_layers(layers)

    def loss(th: Array) -> Array:
        return rollout_cost(env, unravel(th), y0, horizon)

    total_cost, grad = jax.value_and_grad(loss)(theta)
    new_theta = _apply_grad(theta, grad, lr)
    return StepResult(
        layers=unravel(new_theta),
        grad_norm=float(jnp.linalg.norm(grad)),
        total_cost=float(total_cost),
        diagnostics={"grad_theta": np.asarray(grad, dtype=np.float64)},
    )


def truncated_bptt_step(
    env: DifferentiableEnvironment[Array],
    layers: Sequence[Layer],
    y0: Array,
    horizon: int,
    window: int,
    lr: float,
) -> StepResult:
    """Backprop through only the last ``window`` steps; stop-gradient elsewhere."""
    if window < 1 or window > horizon:
        raise ValueError(f"window must be in [1, horizon={horizon}], got {window}")
    theta, unravel = flatten_layers(layers)

    def loss(th: Array) -> Array:
        ll = unravel(th)
        y = y0
        total = jnp.zeros((), dtype=y0.dtype)
        boundary = horizon - window
        for t in range(horizon):
            if t == boundary and boundary > 0:
                y = jax.lax.stop_gradient(y)
            u = policy_forward(ll, y)
            total = total + env.cost(y, u)
            y = env.step(y, u)
        return total + env.terminal_cost(y)

    total_cost, grad = jax.value_and_grad(loss)(theta)
    new_theta = _apply_grad(theta, grad, lr)
    return StepResult(
        layers=unravel(new_theta),
        grad_norm=float(jnp.linalg.norm(grad)),
        total_cost=float(total_cost),
        diagnostics={"window": window, "grad_theta": np.asarray(grad, dtype=np.float64)},
    )


def zero_order_step(
    env: DifferentiableEnvironment[Array],
    layers: Sequence[Layer],
    y0: Array,
    horizon: int,
    lr: float,
    *,
    sigma: float = 0.05,
    n_samples: int = 32,
    key: Array,
) -> StepResult:
    r"""Antithetic zero-order (evolution-strategies) gradient estimate.

    ``grad_hat = mean_i [(cost(theta + sigma eps_i) - cost(theta - sigma eps_i))
    / (2 sigma)] eps_i``. The model-free comparison point in this module's
    docstring; not PPO, not TD3.
    """
    if sigma <= 0.0:
        raise ValueError(f"sigma must be > 0, got {sigma}")
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")
    theta, unravel = flatten_layers(layers)

    def cost_fn(th: Array) -> Array:
        return rollout_cost(env, unravel(th), y0, horizon)

    eps = jax.random.normal(key, (n_samples, *theta.shape), dtype=theta.dtype)

    def sample(e: Array) -> Array:
        c_plus = cost_fn(theta + sigma * e)
        c_minus = cost_fn(theta - sigma * e)
        return (c_plus - c_minus) / (2.0 * sigma) * e

    grads = jax.vmap(sample)(eps)
    grad = jnp.mean(grads, axis=0)
    base_cost = cost_fn(theta)
    new_theta = _apply_grad(theta, grad, lr)
    return StepResult(
        layers=unravel(new_theta),
        grad_norm=float(jnp.linalg.norm(grad)),
        total_cost=float(base_cost),
        diagnostics={"n_samples": n_samples, "sigma": sigma, "grad_theta": np.asarray(grad, dtype=np.float64)},
    )


def actor_adjoint_step(
    env: DifferentiableEnvironment[Array],
    layers: Sequence[Layer],
    y0: Array,
    horizon: int,
    lr: float,
) -> StepResult:
    """Full-horizon exact adjoint gradient step (ours; gates G3/G4)."""
    theta, unravel = flatten_layers(layers)
    result: ActorAdjointResult = actor_adjoint_gradient(env, list(layers), y0, horizon)
    grad = jnp.asarray(result.grad_theta, dtype=theta.dtype)
    new_theta = _apply_grad(theta, grad, lr)
    return StepResult(
        layers=unravel(new_theta),
        grad_norm=float(np.linalg.norm(result.grad_theta)),
        total_cost=result.total_cost,
        diagnostics={"lambda_norms": result.lambda_norms, "grad_theta": result.grad_theta},
    )


@dataclass(frozen=True)
class PSDTerminalHead:
    r"""Learned terminal adjoint ``lambda_head(y) = (L L^T)(y - target)``.

    ``V_head(y) = 0.5 (y - target)^T (L L^T) (y - target)`` is a PSD
    quadratic (by construction, for any real ``L``), so ``lambda_head`` is
    its gradient -- a closed-form Riccati-style terminal value, not a
    generic MLP critic. This is the specific parameterization named in
    ``theory/10-control/02-jet-adjoint-policy-optimization.md``.
    """

    l_matrix: Array
    target: Array

    def value_gradient(self, y: Array) -> Array:
        err = y - self.target
        return (self.l_matrix @ self.l_matrix.T) @ err

    def value(self, y: Array) -> Array:
        err = y - self.target
        return 0.5 * jnp.dot(err, (self.l_matrix @ self.l_matrix.T) @ err)


def fit_terminal_head(
    head: PSDTerminalHead,
    states: Sequence[Array],
    true_terminal_grads: Sequence[Array],
    lr: float,
    steps: int,
) -> PSDTerminalHead:
    """Gradient-descent regression of ``L`` toward the observed terminal cost gradients."""
    if len(states) != len(true_terminal_grads):
        raise ValueError("states and true_terminal_grads must have equal length")
    states_arr = jnp.stack(list(states))
    targets_arr = jnp.stack(list(true_terminal_grads))
    l_matrix = head.l_matrix

    def loss(l_mat: Array) -> Array:
        h = PSDTerminalHead(l_matrix=l_mat, target=head.target)
        preds = jax.vmap(h.value_gradient)(states_arr)
        return jnp.mean(jnp.sum((preds - targets_arr) ** 2, axis=-1))

    for _ in range(steps):
        grad = jax.grad(loss)(l_matrix)
        l_matrix = l_matrix - lr * grad
    return PSDTerminalHead(l_matrix=l_matrix, target=head.target)


def actor_adjoint_jet_step(
    env: DifferentiableEnvironment[Array],
    layers: Sequence[Layer],
    head: PSDTerminalHead,
    y0: Array,
    window: int,
    lr: float,
) -> StepResult:
    r"""Short-window exact adjoint bootstrapped by a learned terminal head (ours, PEARL-style).

    Rolls out exactly ``window`` steps (all Jacobians exact, as in
    :func:`actor_adjoint_step`), then substitutes ``head.value_gradient(y_window)``
    for the true continuation adjoint instead of stopping at the true
    horizon. The resulting bias is exactly the quantity
    :mod:`omnibias.control.certified.gradient_bias` bounds.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    theta, unravel = flatten_layers(layers)
    y = jnp.asarray(y0)
    a_seq: list[FloatArray] = []
    b_seq: list[FloatArray] = []
    dpi_dy_seq: list[FloatArray] = []
    dpi_dtheta_seq: list[FloatArray] = []
    dl_dy_seq: list[FloatArray] = []
    dl_du_seq: list[FloatArray] = []
    total_cost = 0.0
    for _ in range(window):
        u = policy_forward(layers, y)
        a, b = env_step_jacobians(env, y, u)
        dpi_dy = policy_jacobian_dy(layers, y)
        dpi_dtheta = policy_jacobian_dtheta(layers, y)
        dl_dy, dl_du = cost_gradients(env, y, u)
        total_cost += float(env.cost(y, u))
        a_seq.append(np.asarray(a, dtype=np.float64))
        b_seq.append(np.asarray(b, dtype=np.float64))
        dpi_dy_seq.append(np.asarray(dpi_dy, dtype=np.float64))
        dpi_dtheta_seq.append(np.asarray(dpi_dtheta, dtype=np.float64))
        dl_dy_seq.append(np.asarray(dl_dy, dtype=np.float64))
        dl_du_seq.append(np.asarray(dl_du, dtype=np.float64))
        y = env.step(y, u)

    lambda_terminal_pred = np.asarray(head.value_gradient(y), dtype=np.float64)
    m_seq = [closed_loop_jacobian(a_seq[k], b_seq[k], dpi_dy_seq[k]) for k in range(window)]
    c_seq = [
        total_state_cost_gradient(dl_dy_seq[k], dl_du_seq[k], dpi_dy_seq[k]) for k in range(window)
    ]
    lambda_seq = adjoint_recursion(m_seq, c_seq, lambda_terminal_pred)
    grad_theta = policy_gradient(dl_du_seq, b_seq, dpi_dtheta_seq, lambda_seq)

    grad = jnp.asarray(grad_theta, dtype=theta.dtype)
    new_theta = _apply_grad(theta, grad, lr)
    true_terminal_grad = np.asarray(terminal_cost_gradient(env, y), dtype=np.float64)
    return StepResult(
        layers=unravel(new_theta),
        grad_norm=float(np.linalg.norm(grad_theta)),
        total_cost=total_cost + float(env.terminal_cost(y)),
        diagnostics={
            "window": window,
            "lambda_terminal_pred": lambda_terminal_pred,
            "true_terminal_grad": true_terminal_grad,
            "head_error": float(np.max(np.abs(lambda_terminal_pred - true_terminal_grad))),
            "grad_theta": grad_theta,
            "final_state": y,
        },
    )


__all__ = [
    "PSDTerminalHead",
    "StepResult",
    "actor_adjoint_jet_step",
    "actor_adjoint_step",
    "bptt_step",
    "fit_terminal_head",
    "rollout_cost",
    "truncated_bptt_step",
    "zero_order_step",
]
