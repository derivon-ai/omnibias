# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Burgers' equation residual (scalar 1D / vector form).

The viscous Burgers equation is

.. math::

    u_t \\;+\\; (u \\cdot \\nabla) u \\;-\\; \\nu\\, \\Delta u \\;=\\; 0.

In 1D this collapses to :math:`u_t + u\\, u_x - \\nu\\, u_{xx} = 0`. For
:math:`D > 1` it is the vector form (Burgers fluid). We support both:

* ``form="scalar"`` -- 1D scalar; component is :math:`u`. Residual ``(B,)``.
* ``form="vector"`` -- vector form on any :math:`D`. Residual ``(B, D)``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from omnibias.pinn._core.state import FieldState
from omnibias.pinn.torch.equations._types import BurgersOutput
from torch import Tensor


@dataclass
class Burgers:
    """Configurable Burgers residual.

    Parameters
    ----------
    nu
        Kinematic viscosity. Default 0.01.
    form
        Either ``"scalar"`` (1D Burgers, single component) or ``"vector"``
        (vector form on any D, requires ``velocity`` tuple).
    component
        Component name when ``form="scalar"``. Default ``"u"``.
    velocity
        Tuple of component names when ``form="vector"``. Default
        ``("u", "v", "w")``; only the first ``D`` are used (where ``D``
        is the number of spatial axes).
    forcing
        Optional ``f(state) -> Tensor`` of shape matching the residual.
    """

    nu: float = 0.01
    form: str = "scalar"
    component: str = "u"
    velocity: tuple[str, ...] = ("u", "v", "w")
    forcing: Callable[[FieldState], Tensor] | None = None

    def __call__(self, state: FieldState) -> BurgersOutput:
        if self.form == "scalar":
            return self._scalar(state)
        if self.form == "vector":
            return self._vector(state)
        raise ValueError(
            f"Burgers form must be 'scalar' or 'vector', got {self.form!r}"
        )

    def _scalar(self, state: FieldState) -> BurgersOutput:
        time = state.coordinate_spec.time_axis
        if time is None:
            raise ValueError("Burgers equation requires a time axis")
        spatial = state.coordinate_spec.spatial_axes
        u = state.ops.value(state, self.component)
        u_t = state.ops.derivative(state, self.component, axis=time, order=1)
        lap = state.ops.laplacian(state, self.component)
        adv = None
        for ax in spatial:
            u_a = state.ops.derivative(state, self.component, axis=ax, order=1)
            term = u * u_a
            adv = term if adv is None else adv + term
        assert adv is not None
        residual = u_t + adv - self.nu * lap
        if self.forcing is not None:
            residual = residual - self.forcing(state)
        return BurgersOutput(
            residual=residual,
            diag={"mean_sq_residual": float((residual.detach() ** 2).mean())},
        )

    def _vector(self, state: FieldState) -> BurgersOutput:
        time = state.coordinate_spec.time_axis
        if time is None:
            raise ValueError("Burgers equation requires a time axis")
        D = state.coordinate_spec.n_spatial
        comps = self.velocity[:D]
        if len(comps) != D:
            raise ValueError(
                f"Burgers vector form needs {D} velocity components, got "
                f"{len(comps)} from velocity={self.velocity!r}"
            )
        u_t = state.ops.vector_derivative(state, comps, axis=time, order=1)
        adv = state.ops.advection(state, velocity=comps)
        lap_vec = state.ops.vector_laplacian(state, comps)
        residual = u_t + adv - self.nu * lap_vec
        if self.forcing is not None:
            residual = residual - self.forcing(state)
        return BurgersOutput(
            residual=residual,
            diag={"mean_sq_residual": float((residual.detach() ** 2).mean())},
        )


def burgers(
    state: FieldState,
    *,
    nu: float = 0.01,
    form: str = "scalar",
    component: str = "u",
    velocity: tuple[str, ...] = ("u", "v", "w"),
    forcing: Callable[[FieldState], Tensor] | None = None,
) -> BurgersOutput:
    """Stateless one-shot wrapper around :class:`Burgers`."""
    return Burgers(
        nu=nu, form=form, component=component, velocity=velocity,
        forcing=forcing,
    )(state)


__all__ = ["Burgers", "burgers"]
