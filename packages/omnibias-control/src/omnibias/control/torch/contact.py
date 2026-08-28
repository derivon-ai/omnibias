# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""A 1-D bouncing point mass on the certified contact smoothing (torch,
theory 10-04).

Bit-identical-in-spirit twin of :mod:`omnibias.control.jax.contact`; see
that module's docstring.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from omnibias.core.contact_smoothing import ContactBiasReport, contact_smoothing_bias_bound
from torch import Tensor


def contact_force_smooth(z: Tensor, beta: float, f_max: float = 1.0) -> Tensor:
    """Tensor twin of :func:`omnibias.core.contact_smoothing.contact_force_smooth`."""
    return f_max * torch.sigmoid(-beta * z)


@dataclass(frozen=True)
class ContactPointMass1D:
    r"""A point mass bouncing off a smoothed floor at ``z = 0``.

    See :class:`omnibias.control.jax.contact.ContactPointMass1D` for the
    dynamics; numerically identical up to backend round-off.
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

    def step(self, y: Tensor, u: Tensor) -> Tensor:
        z, v = y[0], y[1]
        force = contact_force_smooth(z, self.beta, self.f_max)
        v_next = v + self.dt * (-self.gravity + force / self.mass + u[0] / self.mass)
        z_next = z + self.dt * v_next
        return torch.stack([z_next, v_next])

    def cost(self, y: Tensor, u: Tensor) -> Tensor:
        err = y[0] - self.target_height
        return self.state_cost * err * err + self.control_cost * torch.sum(u * u)

    def terminal_cost(self, y: Tensor) -> Tensor:
        err = y[0] - self.target_height
        return self.terminal_cost_weight * err * err


def contact_certificate(z_min: float, beta: float, f_max: float = 1.0) -> ContactBiasReport:
    """Sound one-sided bias bound; reuses :mod:`omnibias.core.contact_smoothing` verbatim."""
    return contact_smoothing_bias_bound(z_min=z_min, beta=beta, f_max=f_max)


__all__ = [
    "ContactPointMass1D",
    "contact_certificate",
    "contact_force_smooth",
]
