# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Policy-gradient trainer steps on the ``DifferentiableEnvironment`` seam (torch,
theory 10-02).

Bit-identical-in-spirit twin of :mod:`omnibias.control.jax.policy`; see that
module's docstring for the five-arm comparison (``bptt_step``,
``truncated_bptt_step``, ``zero_order_step``, ``actor_adjoint_step``,
``actor_adjoint_jet_step``) and the honesty notes on what each one is (and
is not) a claim about.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from omnibias.control.ocp import DifferentiableEnvironment
from omnibias.control.torch.adjoint import (
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
from omnibias.core.adjoint import (
    FloatArray,
    adjoint_recursion,
    closed_loop_jacobian,
    policy_gradient,
    total_state_cost_gradient,
)
from torch import Tensor


@dataclass(frozen=True)
class StepResult:
    """Common return type for every trainer step below."""

    layers: list[Layer]
    grad_norm: float
    total_cost: float
    diagnostics: dict[str, Any]


def rollout_cost(
    env: DifferentiableEnvironment[Tensor], layers: Sequence[Layer], y0: Tensor, horizon: int
) -> Tensor:
    """Pure differentiable rollout cost ``sum_k L(y_k, u_k) + Phi(y_H)``."""
    y = y0
    total = torch.zeros((), dtype=y0.dtype, device=y0.device)
    for _ in range(horizon):
        u = policy_forward(layers, y)
        total = total + env.cost(y, u)
        y = env.step(y, u)
    return total + env.terminal_cost(y)


def _apply_grad(theta: Tensor, grad: Tensor, lr: float) -> Tensor:
    return theta - lr * grad


def bptt_step(
    env: DifferentiableEnvironment[Tensor],
    layers: Sequence[Layer],
    y0: Tensor,
    horizon: int,
    lr: float,
) -> StepResult:
    """Full-horizon autograd through the unrolled rollout (gate G1 reference)."""
    theta, structure = flatten_layers(layers)
    theta = theta.detach().clone().requires_grad_(True)
    total_cost = rollout_cost(env, structure.unflatten(theta), y0, horizon)
    (grad,) = torch.autograd.grad(total_cost, theta)
    new_theta = _apply_grad(theta.detach(), grad, lr)
    return StepResult(
        layers=structure.unflatten(new_theta),
        grad_norm=float(torch.linalg.norm(grad)),
        total_cost=float(total_cost.detach()),
        diagnostics={"grad_theta": grad.detach().cpu().numpy().astype(np.float64)},
    )


def truncated_bptt_step(
    env: DifferentiableEnvironment[Tensor],
    layers: Sequence[Layer],
    y0: Tensor,
    horizon: int,
    window: int,
    lr: float,
) -> StepResult:
    """Backprop through only the last ``window`` steps; ``.detach()`` elsewhere."""
    if window < 1 or window > horizon:
        raise ValueError(f"window must be in [1, horizon={horizon}], got {window}")
    theta, structure = flatten_layers(layers)
    theta = theta.detach().clone().requires_grad_(True)
    ll = structure.unflatten(theta)
    y = y0
    total = torch.zeros((), dtype=y0.dtype, device=y0.device)
    boundary = horizon - window
    for t in range(horizon):
        if t == boundary and boundary > 0:
            y = y.detach()
        u = policy_forward(ll, y)
        total = total + env.cost(y, u)
        y = env.step(y, u)
    total_cost = total + env.terminal_cost(y)
    (grad,) = torch.autograd.grad(total_cost, theta)
    new_theta = _apply_grad(theta.detach(), grad, lr)
    return StepResult(
        layers=structure.unflatten(new_theta),
        grad_norm=float(torch.linalg.norm(grad)),
        total_cost=float(total_cost.detach()),
        diagnostics={"window": window, "grad_theta": grad.detach().cpu().numpy().astype(np.float64)},
    )


def zero_order_step(
    env: DifferentiableEnvironment[Tensor],
    layers: Sequence[Layer],
    y0: Tensor,
    horizon: int,
    lr: float,
    *,
    sigma: float = 0.05,
    n_samples: int = 32,
    generator: torch.Generator | None = None,
) -> StepResult:
    r"""Antithetic zero-order (evolution-strategies) gradient estimate.

    The model-free comparison point (see the JAX twin's docstring); not
    PPO, not TD3.
    """
    if sigma <= 0.0:
        raise ValueError(f"sigma must be > 0, got {sigma}")
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")
    theta, structure = flatten_layers(layers)
    theta = theta.detach()

    def cost_fn(th: Tensor) -> Tensor:
        with torch.no_grad():
            return rollout_cost(env, structure.unflatten(th), y0, horizon)

    grads = []
    for _ in range(n_samples):
        eps = torch.randn(theta.shape, dtype=theta.dtype, device=theta.device, generator=generator)
        c_plus = cost_fn(theta + sigma * eps)
        c_minus = cost_fn(theta - sigma * eps)
        grads.append(((c_plus - c_minus) / (2.0 * sigma)) * eps)
    grad = torch.stack(grads, dim=0).mean(dim=0)
    base_cost = cost_fn(theta)
    new_theta = _apply_grad(theta, grad, lr)
    return StepResult(
        layers=structure.unflatten(new_theta),
        grad_norm=float(torch.linalg.norm(grad)),
        total_cost=float(base_cost),
        diagnostics={
            "n_samples": n_samples,
            "sigma": sigma,
            "grad_theta": grad.detach().cpu().numpy().astype(np.float64),
        },
    )


def actor_adjoint_step(
    env: DifferentiableEnvironment[Tensor],
    layers: Sequence[Layer],
    y0: Tensor,
    horizon: int,
    lr: float,
) -> StepResult:
    """Full-horizon exact adjoint gradient step (ours; gates G3/G4)."""
    theta, structure = flatten_layers(layers)
    result: ActorAdjointResult = actor_adjoint_gradient(env, list(layers), y0, horizon)
    grad = torch.as_tensor(result.grad_theta, dtype=theta.dtype, device=theta.device)
    new_theta = _apply_grad(theta.detach(), grad, lr)
    return StepResult(
        layers=structure.unflatten(new_theta),
        grad_norm=float(np.linalg.norm(result.grad_theta)),
        total_cost=result.total_cost,
        diagnostics={"lambda_norms": result.lambda_norms, "grad_theta": result.grad_theta},
    )


@dataclass(frozen=True)
class PSDTerminalHead:
    r"""Learned terminal adjoint ``lambda_head(y) = (L L^T)(y - target)``.

    See the JAX twin :class:`omnibias.control.jax.policy.PSDTerminalHead`.
    """

    l_matrix: Tensor
    target: Tensor

    def value_gradient(self, y: Tensor) -> Tensor:
        err = y - self.target
        return (self.l_matrix @ self.l_matrix.T) @ err

    def value(self, y: Tensor) -> Tensor:
        err = y - self.target
        return 0.5 * torch.dot(err, (self.l_matrix @ self.l_matrix.T) @ err)


def fit_terminal_head(
    head: PSDTerminalHead,
    states: Sequence[Tensor],
    true_terminal_grads: Sequence[Tensor],
    lr: float,
    steps: int,
) -> PSDTerminalHead:
    """Gradient-descent regression of ``L`` toward the observed terminal cost gradients."""
    if len(states) != len(true_terminal_grads):
        raise ValueError("states and true_terminal_grads must have equal length")
    states_arr = torch.stack(list(states))
    targets_arr = torch.stack(list(true_terminal_grads))
    l_matrix = head.l_matrix.detach().clone()

    def loss(l_mat: Tensor) -> Tensor:
        lt = l_mat @ l_mat.T
        errs = states_arr - head.target
        preds = errs @ lt.T
        return torch.mean(torch.sum((preds - targets_arr) ** 2, dim=-1))

    for _ in range(steps):
        l_matrix = l_matrix.detach().clone().requires_grad_(True)
        loss_val = loss(l_matrix)
        (g,) = torch.autograd.grad(loss_val, l_matrix)
        l_matrix = (l_matrix.detach() - lr * g).clone()
    return PSDTerminalHead(l_matrix=l_matrix.detach(), target=head.target)


def actor_adjoint_jet_step(
    env: DifferentiableEnvironment[Tensor],
    layers: Sequence[Layer],
    head: PSDTerminalHead,
    y0: Tensor,
    window: int,
    lr: float,
) -> StepResult:
    r"""Short-window exact adjoint bootstrapped by a learned terminal head (ours, PEARL-style).

    See the JAX twin :func:`omnibias.control.jax.policy.actor_adjoint_jet_step`.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    theta, structure = flatten_layers(layers)
    y = torch.as_tensor(y0)
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
        a_seq.append(a.detach().cpu().numpy().astype(np.float64))
        b_seq.append(b.detach().cpu().numpy().astype(np.float64))
        dpi_dy_seq.append(dpi_dy.detach().cpu().numpy().astype(np.float64))
        dpi_dtheta_seq.append(dpi_dtheta.detach().cpu().numpy().astype(np.float64))
        dl_dy_seq.append(dl_dy.detach().cpu().numpy().astype(np.float64))
        dl_du_seq.append(dl_du.detach().cpu().numpy().astype(np.float64))
        y = env.step(y, u)

    lambda_terminal_pred = head.value_gradient(y).detach().cpu().numpy().astype(np.float64)
    m_seq = [closed_loop_jacobian(a_seq[k], b_seq[k], dpi_dy_seq[k]) for k in range(window)]
    c_seq = [
        total_state_cost_gradient(dl_dy_seq[k], dl_du_seq[k], dpi_dy_seq[k]) for k in range(window)
    ]
    lambda_seq = adjoint_recursion(m_seq, c_seq, lambda_terminal_pred)
    grad_theta = policy_gradient(dl_du_seq, b_seq, dpi_dtheta_seq, lambda_seq)

    grad = torch.as_tensor(grad_theta, dtype=theta.dtype, device=theta.device)
    new_theta = _apply_grad(theta.detach(), grad, lr)
    true_terminal_grad = terminal_cost_gradient(env, y).detach().cpu().numpy().astype(np.float64)
    return StepResult(
        layers=structure.unflatten(new_theta),
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
