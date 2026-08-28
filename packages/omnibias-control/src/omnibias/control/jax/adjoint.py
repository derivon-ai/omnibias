# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Closed-loop Jacobian assembly for the jet-adjoint policy gradient (JAX twin).

Bit-identical partner of :mod:`omnibias.control.torch.adjoint` on the same
directional jets. This module assembles the per-step arrays
:mod:`omnibias.core.adjoint` needs (``A_k, B_k, dpi/dy_k, dpi/dtheta_k,
dL/dy_k, dL/du_k``) and then hands the *exact* backward recursion to that
pure-Python module -- the recursion itself is not reimplemented here.

Two different exactness claims are combined and must not be conflated:

* ``dpi/dy_k`` -- the policy's state Jacobian -- is assembled from the
  founding **bias collapse** (``delta -> 0``) tower via
  :func:`omnibias.jax.jet.mlp_jet`: one directional jet per input basis
  vector, each of which is exact to float64 round-off, not a finite
  difference.
* ``A_k, B_k`` (environment Jacobians) and ``dpi/dtheta_k`` (the policy's
  *parameter* Jacobian) come from ordinary ``jax.jacfwd`` / ``jax.jacrev``
  reverse/forward-mode automatic differentiation -- exact, but **not** the
  closed-form tower (there is no activation dictionary to look up; the
  environment step and the parameter dependence are generic functions).

Temperature collapse (``beta -> inf``, feasibility) does not appear in
this module; it appears only in the CBF-QP filter elsewhere in this
package. do not conflate the two.

The driver is host-side eager (Python loops over the horizon, one
directional jet per state dimension per step); do not wrap it in ``jax.jit``
and expect it to trace through cleanly -- see the risk note in
``theory/10-control/02-jet-adjoint-policy-optimization.md`` on the cost of
Jacobian assembly at scale.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.flatten_util import ravel_pytree
from omnibias.control.ocp import DifferentiableEnvironment
from omnibias.core.adjoint import (
    FloatArray,
    adjoint_recursion,
    closed_loop_jacobian,
    policy_gradient,
    total_state_cost_gradient,
)
from omnibias.core.spec import ActivationSpec
from omnibias.jax.jet import jet_to_tower, mlp_jet

Layer = tuple[Array, Array, ActivationSpec[Array] | str | None]


def flatten_layers(layers: Sequence[Layer]) -> tuple[Array, Any]:
    """Ravel ``(W, b)`` pairs into one flat parameter vector plus an unravel fn.

    The activation spec of each layer is *structure*, not a parameter, and
    is closed over by the returned unravel function.
    """
    packed = [(w, b) for (w, b, _spec) in layers]
    flat, unravel_pair = ravel_pytree(packed)
    specs = [spec for (_w, _b, spec) in layers]

    def unravel(theta: Array) -> list[Layer]:
        pairs = unravel_pair(theta)
        return [(w, b, spec) for (w, b), spec in zip(pairs, specs, strict=True)]

    return flat, unravel


def policy_forward(layers: Sequence[Layer], y: Array) -> Array:
    """Evaluate the policy ``u = pi(y)`` at one state vector (no batch dim)."""
    y = jnp.asarray(y)
    zero_dir = jnp.zeros_like(y)
    jet = mlp_jet(y, zero_dir, list(layers), order=0)
    return jet_to_tower(jet)[0]


def policy_jacobian_dy(layers: Sequence[Layer], y: Array) -> Array:
    r"""Exact ``dpi/dy`` at one state via one directional jet per basis vector.

    Returns shape ``(action_dim, state_dim)``. Each column is
    ``jet_to_tower(mlp_jet(y, e_i, layers, order=1))[1]``, the order-1 jet
    coefficient along the ``i``-th standard basis direction ``e_i`` -- the
    founding bias collapse tower, contracted along a direction rather than
    formed as a dense multivariate jet (see the risk note on scale).
    """
    y = jnp.asarray(y)
    state_dim = int(y.shape[-1])
    basis = jnp.eye(state_dim, dtype=y.dtype)

    def column(e_i: Array) -> Array:
        jet = mlp_jet(y, e_i, list(layers), order=1)
        return jet_to_tower(jet)[1]

    cols = jax.vmap(column)(basis)  # (state_dim, action_dim)
    return cols.T


def policy_jacobian_dtheta(layers: Sequence[Layer], y: Array) -> Array:
    r"""Exact ``dpi/dtheta`` at one state via reverse-mode ``jax.jacrev``.

    Ordinary automatic differentiation (not the closed-form tower): the
    dependence on flattened parameters ``theta`` has no activation
    dictionary to look up. Returns shape ``(action_dim, n_params)``.
    """
    theta, unravel = flatten_layers(layers)

    def forward(theta_flat: Array) -> Array:
        return policy_forward(unravel(theta_flat), y)

    return jax.jacrev(forward)(theta)


def env_step_jacobians(
    env: DifferentiableEnvironment[Array], y: Array, u: Array
) -> tuple[Array, Array]:
    """``(A, B) = (d step/dy, d step/du)`` at one ``(y, u)`` via ``jax.jacfwd``."""
    a = jax.jacfwd(lambda yy: env.step(yy, u))(y)
    b = jax.jacfwd(lambda uu: env.step(y, uu))(u)
    return a, b


def cost_gradients(
    env: DifferentiableEnvironment[Array], y: Array, u: Array
) -> tuple[Array, Array]:
    """``(dL/dy, dL/du)`` at one ``(y, u)`` via ``jax.grad``."""
    dl_dy = jax.grad(lambda yy: env.cost(yy, u))(y)
    dl_du = jax.grad(lambda uu: env.cost(y, uu))(u)
    return dl_dy, dl_du


def terminal_cost_gradient(env: DifferentiableEnvironment[Array], y: Array) -> Array:
    """``dPhi/dy`` at the terminal state via ``jax.grad``."""
    return jax.grad(env.terminal_cost)(y)


@dataclass(frozen=True)
class ActorAdjointResult:
    """Exact adjoint policy gradient plus rollout diagnostics."""

    grad_theta: FloatArray
    lambda_norms: list[float]
    total_cost: float
    states: list[Array]


def actor_adjoint_gradient(
    env: DifferentiableEnvironment[Array],
    layers: Sequence[Layer],
    y0: Array,
    horizon: int,
) -> ActorAdjointResult:
    r"""Exact policy gradient via the adjoint recursion (theory 10-02, gate G3/G4).

    Rolls the closed loop out ``horizon`` steps, assembles ``(A_k, B_k,
    dpi/dy_k, dpi/dtheta_k, dL/dy_k, dL/du_k)`` at each step, and passes
    them to the backend-free recursion in :mod:`omnibias.core.adjoint`.
    The returned gradient is a plain ``numpy`` array (the adjoint sweep
    itself is not re-differentiated; applying it as a plain gradient step
    is standard practice for adjoint-method training).
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    y = jnp.asarray(y0)
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
        a_seq.append(np.asarray(a, dtype=np.float64))
        b_seq.append(np.asarray(b, dtype=np.float64))
        dpi_dy_seq.append(np.asarray(dpi_dy, dtype=np.float64))
        dpi_dtheta_seq.append(np.asarray(dpi_dtheta, dtype=np.float64))
        dl_dy_seq.append(np.asarray(dl_dy, dtype=np.float64))
        dl_du_seq.append(np.asarray(dl_du, dtype=np.float64))
        y = env.step(y, u)
        states.append(y)
    total_cost += float(env.terminal_cost(y))
    lambda_terminal = np.asarray(terminal_cost_gradient(env, y), dtype=np.float64)

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
