# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Heat equation residual.

The linear heat equation in any spatial dimension :math:`D` is

.. math::

    u_t \\;-\\; \\alpha\\, \\Delta u \\;=\\; s(x, t),

where :math:`\\alpha > 0` is the diffusivity and :math:`s` is an
optional source term. The PINN residual is just the LHS minus the
source.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from omnibias.pinn._core.state import FieldState
from omnibias.pinn.torch.equations._types import HeatOutput
from torch import Tensor


@dataclass
class Heat:
    """Configurable linear-heat-equation residual.

    Parameters
    ----------
    alpha
        Diffusivity. Default 1.0.
    component
        Name of the field component to use. Default ``"u"``.
    source
        Optional callable ``s(state) -> Tensor`` of shape ``(B,)``.
    """

    alpha: float = 1.0
    component: str = "u"
    source: Callable[[FieldState], Tensor] | None = None

    def __call__(self, state: FieldState) -> HeatOutput:
        time = state.coordinate_spec.time_axis
        if time is None:
            raise ValueError(
                "Heat equation requires a time axis in the coordinate spec"
            )
        u_t = state.ops.derivative(state, self.component, axis=time, order=1)
        lap_u = state.ops.laplacian(state, self.component)
        residual = u_t - self.alpha * lap_u
        if self.source is not None:
            residual = residual - self.source(state)
        diag = {
            "mean_sq_residual": float((residual.detach() ** 2).mean()),
        }
        return HeatOutput(residual=residual, diag=diag)


def heat(
    state: FieldState,
    *,
    alpha: float = 1.0,
    component: str = "u",
    source: Callable[[FieldState], Tensor] | None = None,
) -> HeatOutput:
    """Stateless one-shot wrapper around :class:`Heat`."""
    return Heat(alpha=alpha, component=component, source=source)(state)


__all__ = ["Heat", "heat"]
