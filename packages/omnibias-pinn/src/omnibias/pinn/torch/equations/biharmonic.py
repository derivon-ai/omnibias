# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Biharmonic equation residual.

The biharmonic equation in any spatial dimension :math:`D` is

.. math::

    \\Delta^2 u \\;=\\; s(x)

with :math:`s` an optional source term. This is the canonical
4th-order operator that appears in plate theory, slow Stokes flow and
the 2D Navier-Stokes vorticity-streamfunction viscous term.

Steady (no time dependence) by default. For the time-dependent variant
``\\partial_t u + \\Delta^2 u = s`` set ``include_time=True``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from omnibias.pinn._core.state import FieldState
from omnibias.pinn.torch.equations._types import BiharmonicOutput
from torch import Tensor


@dataclass
class Biharmonic:
    """Biharmonic residual.

    Parameters
    ----------
    component
        Field component name. Default ``"u"``.
    include_time
        If ``True``, residual is :math:`u_t + \\Delta^2 u - s`.
        Default ``False`` (steady form :math:`\\Delta^2 u - s`).
    source
        Optional ``f(state) -> Tensor`` of shape ``(B,)``.
    """

    component: str = "u"
    include_time: bool = False
    source: Callable[[FieldState], Tensor] | None = None

    def __call__(self, state: FieldState) -> BiharmonicOutput:
        bih = state.ops.biharmonic(state, self.component)
        if self.include_time:
            time = state.coordinate_spec.time_axis
            if time is None:
                raise ValueError(
                    "Biharmonic(include_time=True) requires a time axis"
                )
            u_t = state.ops.derivative(state, self.component, axis=time, order=1)
            residual = u_t + bih
        else:
            residual = bih
        if self.source is not None:
            residual = residual - self.source(state)
        return BiharmonicOutput(
            residual=residual,
            diag={"mean_sq_residual": float((residual.detach() ** 2).mean())},
        )


def biharmonic(
    state: FieldState,
    *,
    component: str = "u",
    include_time: bool = False,
    source: Callable[[FieldState], Tensor] | None = None,
) -> BiharmonicOutput:
    """Stateless one-shot wrapper around :class:`Biharmonic`."""
    return Biharmonic(
        component=component, include_time=include_time, source=source,
    )(state)


__all__ = ["Biharmonic", "biharmonic"]
