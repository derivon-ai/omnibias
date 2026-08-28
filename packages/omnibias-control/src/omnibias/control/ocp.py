# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Backend-agnostic optimal-control-problem containers (theory 10-02).

``DifferentiableEnvironment`` is the one seam every trainer and benchmark
in this program is written against: a differentiable step function plus a
declared state/action dimension. Backends supply concrete environments
(:mod:`omnibias.control.torch.envs`, :mod:`omnibias.control.jax.envs`,
:mod:`omnibias.control.robotics`) behind the same protocol so a trainer
written once (:mod:`omnibias.control.torch.policy`,
:mod:`omnibias.control.jax.policy`) runs unchanged on a PDE double-gyre, a
native cartpole, or a wrapped physics engine.

This module holds no arrays and no tensor operations; it is pure Python
containers plus a ``Protocol``, matching :mod:`omnibias.control.problem`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar, runtime_checkable

ArrayT = TypeVar("ArrayT")


@runtime_checkable
class DifferentiableEnvironment(Protocol[ArrayT]):
    r"""The shared seam: a differentiable discrete-time transition.

    Attributes
    ----------
    state_dim, action_dim:
        Declared dimensions (used to shape policy networks and Jacobians
        without tracing the step once for that purpose alone).

    Every implementation must be autodiff-transparent in its own backend
    (a plain ``torch.Tensor -> torch.Tensor`` or ``jax.Array -> jax.Array``
    function) so a policy can be trained *through* ``step`` by
    backpropagation-through-time or the adjoint recursion in
    :mod:`omnibias.control.torch.adjoint` / :mod:`omnibias.control.jax.adjoint`.
    """

    state_dim: int
    action_dim: int

    def step(self, y: ArrayT, u: ArrayT) -> ArrayT:
        """One differentiable transition ``y_{k+1} = step(y_k, u_k)``."""
        ...

    def cost(self, y: ArrayT, u: ArrayT) -> ArrayT:
        """Per-step running cost ``L(y, u)`` (scalar per batch row)."""
        ...

    def terminal_cost(self, y: ArrayT) -> ArrayT:
        """Terminal cost ``Phi(y_H)`` (scalar per batch row)."""
        ...


@dataclass(frozen=True)
class OCPSpec:
    r"""Static shape / horizon description of a finite-horizon OCP.

    Attributes
    ----------
    state_dim, action_dim:
        Dimensions of ``y`` and ``u``.
    horizon:
        Number of discrete stages ``H >= 1``.
    dt:
        Nominal time step (informational; the environment owns integration).
    """

    state_dim: int
    action_dim: int
    horizon: int
    dt: float = 1.0

    def __post_init__(self) -> None:
        if self.state_dim < 1 or self.action_dim < 1:
            raise ValueError("state_dim and action_dim must be >= 1")
        if self.horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {self.horizon}")
        if self.dt <= 0.0:
            raise ValueError(f"dt must be > 0, got {self.dt}")


@dataclass(frozen=True)
class Rollout(Generic[ArrayT]):
    r"""A recorded closed-loop trajectory.

    Attributes
    ----------
    states:
        ``y_0, ..., y_H`` (length ``H+1``), each shape ``(..., state_dim)``.
    actions:
        ``u_0, ..., u_{H-1}`` (length ``H``), each shape ``(..., action_dim)``.
    costs:
        Per-step running costs ``L_0, ..., L_{H-1}``.
    terminal_cost:
        ``Phi(y_H)``.
    """

    states: list[ArrayT]
    actions: list[ArrayT]
    costs: list[ArrayT]
    terminal_cost: ArrayT


def rollout_generic(
    env: DifferentiableEnvironment[ArrayT],
    policy: Callable[[ArrayT], ArrayT],
    y0: ArrayT,
    horizon: int,
) -> Rollout[ArrayT]:
    r"""Backend-agnostic closed-loop rollout using only ``env`` / ``policy`` calls.

    Works for any backend whose arrays support the operations ``env.step``
    /  ``env.cost`` use internally; this function itself performs no
    arithmetic on the arrays, only Python list bookkeeping, so it is safe
    to reuse verbatim from the torch and jax twins.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    states = [y0]
    actions: list[ArrayT] = []
    costs: list[ArrayT] = []
    y = y0
    for _ in range(horizon):
        u = policy(y)
        costs.append(env.cost(y, u))
        y = env.step(y, u)
        states.append(y)
        actions.append(u)
    terminal = env.terminal_cost(y)
    return Rollout(states=states, actions=actions, costs=costs, terminal_cost=terminal)


__all__ = [
    "DifferentiableEnvironment",
    "OCPSpec",
    "Rollout",
    "rollout_generic",
]
