# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Klein-Gordon equation residual (torch backend).

The Klein-Gordon equation for a *real* scalar field
:math:`\phi(x^\mu)` reads

.. math::

    \Box\phi - m^2\phi - \lambda\phi^3 = 0,
    \qquad \Box = -\partial_t^2 + \nabla^2,

with the mostly-plus metric convention :math:`\eta_{\mu\nu} =
\text{diag}(-1, +1, +1, +1)`. The :math:`\lambda\phi^3` term comes
from a :math:`\phi^4 / 4` self-interaction; setting :math:`\lambda = 0`
recovers the free Klein-Gordon equation.

For v0.0.1 we ship the real-scalar form (the most common in
field-theoretic applications); a future v0.0.2 will add the
complex-scalar variant with a non-zero conserved current.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from omnibias.pinn._core.state import FieldState
from torch import Tensor


@dataclass
class KleinGordonOutput:
    """Output of :class:`KleinGordon`.

    Attributes
    ----------
    residual
        Pointwise scalar residual ``Box phi - m^2 phi - lambda phi^3 -
        source``. Shape ``(B,)``.
    diag
        Diagnostic dict.
    """

    residual: Tensor
    diag: dict[str, float]


@dataclass
class KleinGordon:
    r"""Configurable Klein-Gordon residual.

    Parameters
    ----------
    mass
        :math:`m`. Default 1.0.
    lambda_phi4
        Coefficient :math:`\lambda` of the cubic self-interaction
        :math:`\lambda\phi^3`. Default 0.0 (free Klein-Gordon).
    component
        Field component name. Default ``"phi"``. The component is a
        *real* scalar in v0.0.1.
    source
        Optional callable ``s(state) -> Tensor of shape (B,)`` added
        to the residual (manufactured solutions / coupling to an
        external source).
    """

    mass: float = 1.0
    lambda_phi4: float = 0.0
    component: str = "phi"
    source: Callable[[FieldState], Tensor] | None = None

    def __call__(self, state: FieldState) -> KleinGordonOutput:
        time = state.coordinate_spec.time_axis
        if time is None:
            raise ValueError(
                "Klein-Gordon equation requires a time axis (Box = -d_tt + lap)"
            )
        phi = state.ops.value(state, self.component)
        # Box = -d_tt + lap = d'Alembertian with c=1 in the mostly-plus signature.
        box = state.ops.dalembertian(state, self.component, c=1.0)
        m_sq = self.mass * self.mass
        residual = box - m_sq * phi - self.lambda_phi4 * phi * phi * phi
        if self.source is not None:
            residual = residual - self.source(state)
        return KleinGordonOutput(
            residual=residual,
            diag={"mean_sq_residual": float((residual.detach() ** 2).mean())},
        )


def klein_gordon(
    state: FieldState,
    *,
    mass: float = 1.0,
    lambda_phi4: float = 0.0,
    component: str = "phi",
    source: Callable[[FieldState], Tensor] | None = None,
) -> KleinGordonOutput:
    """Stateless one-shot wrapper around :class:`KleinGordon`."""
    return KleinGordon(
        mass=mass, lambda_phi4=lambda_phi4, component=component, source=source,
    )(state)


__all__ = ["KleinGordon", "KleinGordonOutput", "klein_gordon"]
