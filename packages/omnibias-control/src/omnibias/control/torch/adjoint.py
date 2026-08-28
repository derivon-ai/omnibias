# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Closed-loop Jacobian assembly for the jet-adjoint policy gradient (torch twin).

Bit-identical partner of :mod:`omnibias.control.jax.adjoint` on the same
directional jets. See that module's docstring for the exactness claims:
``dpi/dy_k`` comes from the **founding bias collapse** (``delta -> 0``)
tower via :func:`omnibias.torch.jet.mlp_jet`; the environment Jacobians
and ``dpi/dtheta_k`` come from ordinary reverse-mode ``torch.func``
automatic differentiation, not the closed-form tower. Temperature
collapse (``beta -> inf``, feasibility) does not appear in this module.
do not conflate the two.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from omnibias.control.ocp import DifferentiableEnvironment
from omnibias.core.adjoint import (
    FloatArray,
    adjoint_recursion,
    closed_loop_jacobian,
    policy_gradient,
    total_state_cost_gradient,
)
from omnibias.core.spec import ActivationSpec
from omnibias.torch.jet import jet_to_tower, mlp_jet
from torch import Tensor
from torch.func import grad, jacrev, vmap

Layer = tuple[Tensor, Tensor, ActivationSpec[Tensor] | str | None]


class _LayerStructure:
    """Flatten / unflatten a list of ``(W, b)`` pairs to and from one flat vector."""

    def __init__(self, layers: Sequence[Layer]) -> None:
        self.specs = [spec for (_w, _b, spec) in layers]
        self.shapes = [(w.shape, b.shape) for (w, b, _spec) in layers]
        self.numels = [(w.numel(), b.numel()) for (w, b, _spec) in layers]
        self.dtype = layers[0][0].dtype
        self.device = layers[0][0].device

    def flatten(self, layers: Sequence[Layer]) -> Tensor:
        parts = []
        for w, b, _spec in layers:
            parts.append(w.reshape(-1))
            parts.append(b.reshape(-1))
        return torch.cat(parts)

    def unflatten(self, theta: Tensor) -> list[Layer]:
        out: list[Layer] = []
        offset = 0
        for (w_shape, b_shape), (w_n, b_n), spec in zip(
            self.shapes, self.numels, self.specs, strict=True
        ):
            w = theta[offset : offset + w_n].reshape(w_shape)
            offset += w_n
            b = theta[offset : offset + b_n].reshape(b_shape)
            offset += b_n
            out.append((w, b, spec))
        return out


def flatten_layers(layers: Sequence[Layer]) -> tuple[Tensor, _LayerStructure]:
    """Ravel ``(W, b)`` pairs into one flat parameter vector plus an unravel object."""
    structure = _LayerStructure(layers)
    return structure.flatten(layers), structure


def policy_forward(layers: Sequence[Layer], y: Tensor) -> Tensor:
    """Evaluate the policy ``u = pi(y)`` at one state vector (no batch dim)."""
    y = torch.as_tensor(y)
    zero_dir = torch.zeros_like(y)
    jet = mlp_jet(y, zero_dir, list(layers), order=0)
    return jet_to_tower(jet)[0]


def policy_jacobian_dy(layers: Sequence[Layer], y: Tensor) -> Tensor:
    r"""Exact ``dpi/dy`` at one state via one directional jet per basis vector.

    Returns shape ``(action_dim, state_dim)``, matching the JAX twin.
    """
    y = torch.as_tensor(y)
    state_dim = int(y.shape[-1])
    basis = torch.eye(state_dim, dtype=y.dtype, device=y.device)

    def column(e_i: Tensor) -> Tensor:
        jet = mlp_jet(y, e_i, list(layers), order=1)
        return jet_to_tower(jet)[1]

    cols = vmap(column)(basis)  # (state_dim, action_dim)
    return cols.T


def policy_jacobian_dtheta(layers: Sequence[Layer], y: Tensor) -> Tensor:
    r"""Exact ``dpi/dtheta`` at one state via reverse-mode ``torch.func.jacrev``.

    Ordinary automatic differentiation (not the closed-form tower).
    Returns shape ``(action_dim, n_params)``.
    """
    theta, structure = flatten_layers(layers)

    def forward(theta_flat: Tensor) -> Tensor:
        return policy_forward(structure.unflatten(theta_flat), y)

    return jacrev(forward)(theta)


def env_step_jacobians(
    env: DifferentiableEnvironment[Tensor], y: Tensor, u: Tensor
) -> tuple[Tensor, Tensor]:
    """``(A, B) = (d step/dy, d step/du)`` at one ``(y, u)`` via ``torch.func.jacrev``."""
    a = jacrev(lambda yy: env.step(yy, u))(y)
    b = jacrev(lambda uu: env.step(y, uu))(u)
    return a, b


def cost_gradients(
    env: DifferentiableEnvironment[Tensor], y: Tensor, u: Tensor
) -> tuple[Tensor, Tensor]:
    """``(dL/dy, dL/du)`` at one ``(y, u)`` via ``torch.func.grad``."""
    dl_dy = grad(lambda yy: env.cost(yy, u))(y)
    dl_du = grad(lambda uu: env.cost(y, uu))(u)
    return dl_dy, dl_du


def terminal_cost_gradient(env: DifferentiableEnvironment[Tensor], y: Tensor) -> Tensor:
    """``dPhi/dy`` at the terminal state via ``torch.func.grad``."""
    return grad(env.terminal_cost)(y)


@dataclass(frozen=True)
class ActorAdjointResult:
    """Exact adjoint policy gradient plus rollout diagnostics."""

    grad_theta: FloatArray
    lambda_norms: list[float]
    total_cost: float
    states: list[Tensor]


def actor_adjoint_gradient(
    env: DifferentiableEnvironment[Tensor],
    layers: Sequence[Layer],
    y0: Tensor,
    horizon: int,
) -> ActorAdjointResult:
    r"""Exact policy gradient via the adjoint recursion (theory 10-02, gate G3/G4).

    See the JAX twin :func:`omnibias.control.jax.adjoint.actor_adjoint_gradient`
    for the full description; the two must agree to float64 round-off (gate G9).
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    y = torch.as_tensor(y0)
    states = [y]
    a_seq: list[FloatArray] = []
    b_seq: list[FloatArray] = []
    dpi_dy_seq: list[FloatArray] = []
    dpi_dtheta_seq: list[FloatArray] = []
    dl_dy_seq: list[FloatArray] = []
    dl_du_seq: list[FloatArray] = []
    total_cost = 0.0
    for _ in range(horizon):
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
        states.append(y)
    total_cost += float(env.terminal_cost(y))
    lambda_terminal = terminal_cost_gradient(env, y).detach().cpu().numpy().astype(np.float64)

    m_seq = [closed_loop_jacobian(a_seq[k], b_seq[k], dpi_dy_seq[k]) for k in range(horizon)]
    c_seq = [
        total_state_cost_gradient(dl_dy_seq[k], dl_du_seq[k], dpi_dy_seq[k])
        for k in range(horizon)
    ]
    lambda_seq = adjoint_recursion(m_seq, c_seq, lambda_terminal)
    grad_theta = policy_gradient(dl_du_seq, b_seq, dpi_dtheta_seq, lambda_seq)
    return ActorAdjointResult(
        grad_theta=grad_theta,
        lambda_norms=[float(np.linalg.norm(lam)) for lam in lambda_seq],
        total_cost=total_cost,
        states=states,
    )


__all__ = [
    "ActorAdjointResult",
    "Layer",
    "actor_adjoint_gradient",
    "cost_gradients",
    "env_step_jacobians",
    "flatten_layers",
    "policy_forward",
    "policy_jacobian_dtheta",
    "policy_jacobian_dy",
    "terminal_cost_gradient",
]
