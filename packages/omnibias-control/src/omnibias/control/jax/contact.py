# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""A 1-D bouncing point mass on the certified contact smoothing (JAX,
theory 10-04).

:class:`ContactPointMass1D` is a :class:`omnibias.control.ocp.
DifferentiableEnvironment` whose only nonlinearity is the smoothed normal
contact force ``f_max * sigmoid(-beta * z)`` -- the tensor twin of
:func:`omnibias.core.contact_smoothing.contact_force_smooth`, evaluated
with ``jax.nn.sigmoid`` rather than the pure-Python core function (so it
composes with ``jax.grad`` / ``jax.jacfwd`` for the adjoint trainers), but
identical in value. :func:`contact_certificate` calls the core module's
sound one-sided bias bound directly (not reimplemented) so a training run
can report whether its chosen ``(beta, z_min)`` is actually certified for
the state range it operates in.

Validated only on this 1-D point mass; a 2-D block with Coulomb friction
is leftover-recorded (see :mod:`omnibias.core.contact_smoothing`).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array
from omnibias.core.contact_smoothing import ContactBiasReport, contact_smoothing_bias_bound


def contact_force_smooth(z: Array, beta: float, f_max: float = 1.0) -> Array:
    """Tensor twin of :func:`omnibias.core.contact_smoothing.contact_force_smooth`."""
    return f_max * jax.nn.sigmoid(-beta * z)


@dataclass(frozen=True)
class ContactPointMass1D:
    r"""A point mass of unit-ish mass bouncing off a smoothed floor at ``z = 0``.

    State ``y = (z, v)``: height and vertical velocity. Dynamics (semi-implicit
    Euler):

    .. math::
        v_{k+1} = v_k + dt\,(-g + f_{\text{smooth}}(z_k)/m + u_k/m), \qquad
        z_{k+1} = z_k + dt\, v_{k+1}

    Cost tracks a target height with a control-effort penalty; terminal
    cost is the same tracking error, weighted.
    """

    mass: float = 1.0
    gravity: float = 9.8
    beta: float = 40.0
    f_max: float = 40.0
    dt: float = 0.02
    target_height: float = 1.0
    state_cost: float = 1.0
    control_cost: float = 0.02
    terminal_cost_weight: float = 3.0
    state_dim: int = 2
    action_dim: int = 1

    def step(self, y: Array, u: Array) -> Array:
        z, v = y[0], y[1]
        force = contact_force_smooth(z, self.beta, self.f_max)
        v_next = v + self.dt * (-self.gravity + force / self.mass + u[0] / self.mass)
        z_next = z + self.dt * v_next
        return jnp.stack([z_next, v_next])

    def cost(self, y: Array, u: Array) -> Array:
        err = y[0] - self.target_height
        return self.state_cost * err * err + self.control_cost * jnp.sum(u * u)

    def terminal_cost(self, y: Array) -> Array:
        err = y[0] - self.target_height
        return self.terminal_cost_weight * err * err


def contact_certificate(z_min: float, beta: float, f_max: float = 1.0) -> ContactBiasReport:
    """Sound one-sided bias bound for the environment's ``(beta, f_max)`` choice.

    Reuses :func:`omnibias.core.contact_smoothing.contact_smoothing_bias_bound`
    verbatim; see that function for the ``|z| >= z_min`` scope and the
    ``Inconclusive`` case at ``z_min <= 0``.
    """
    return contact_smoothing_bias_bound(z_min=z_min, beta=beta, f_max=f_max)


__all__ = [
    "ContactPointMass1D",
    "contact_certificate",
    "contact_force_smooth",
]
