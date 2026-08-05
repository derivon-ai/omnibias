# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Kuramoto-Sivashinsky equation residual.

The 1D Kuramoto-Sivashinsky equation on a periodic domain is

.. math::

    u_t \\;+\\; c_{\\mathrm{conv}}\\, u\\, u_x
        \\;+\\; c_{\\mathrm{lap}}\\, u_{xx}
        \\;+\\; c_{\\mathrm{bih}}\\, u_{xxxx}
        \\;=\\; 0.

Canonical coefficients are :math:`c_{\\mathrm{conv}} = c_{\\mathrm{lap}}
= c_{\\mathrm{bih}} = 1`. We support a 2D generalisation
(``form="2d"``) which uses :math:`u\\, |\\nabla u|` for the
nonlinearity and :math:`\\Delta`, :math:`\\Delta^2` for the linear
parts.
"""

from __future__ import annotations

from dataclasses import dataclass

from omnibias.pinn._core.state import FieldState
from omnibias.pinn.torch.equations._types import KSOutput


@dataclass
class KuramotoSivashinsky:
    """KS residual (1D scalar).

    Parameters
    ----------
    c_conv, c_lap, c_bih
        Coefficients on :math:`u\\,u_x`, :math:`u_{xx}`, :math:`u_{xxxx}`.
        Defaults are 1 each (canonical form).
    component
        Component name. Default ``"u"``.
    form
        ``"1d"`` (default) or ``"2d"`` for the 2D generalisation
        :math:`u_t + c_{\\mathrm{conv}}\\, u\\, |\\nabla u| + c_{\\mathrm{lap}}\\, \\Delta u + c_{\\mathrm{bih}}\\, \\Delta^2 u`.
    """

    c_conv: float = 1.0
    c_lap: float = 1.0
    c_bih: float = 1.0
    component: str = "u"
    form: str = "1d"

    def __call__(self, state: FieldState) -> KSOutput:
        if self.form == "1d":
            return self._1d(state)
        if self.form == "2d":
            return self._2d(state)
        raise ValueError(
            f"KuramotoSivashinsky form must be '1d' or '2d', got {self.form!r}"
        )

    def _1d(self, state: FieldState) -> KSOutput:
        time = state.coordinate_spec.time_axis
        if time is None:
            raise ValueError("KS equation requires a time axis")
        spatial = state.coordinate_spec.spatial_axes
        if len(spatial) != 1:
            raise ValueError(
                f"KS 1D form requires exactly 1 spatial axis, got "
                f"{len(spatial)} ({spatial!r})"
            )
        ax = spatial[0]
        u = state.ops.value(state, self.component)
        u_x = state.ops.derivative(state, self.component, axis=ax, order=1)
        u_xx = state.ops.derivative(state, self.component, axis=ax, order=2)
        u_xxxx = state.ops.derivative(state, self.component, axis=ax, order=4)
        u_t = state.ops.derivative(state, self.component, axis=time, order=1)
        residual = (
            u_t
            + self.c_conv * u * u_x
            + self.c_lap * u_xx
            + self.c_bih * u_xxxx
        )
        return KSOutput(
            residual=residual,
            diag={"mean_sq_residual": float((residual.detach() ** 2).mean())},
        )

    def _2d(self, state: FieldState) -> KSOutput:
        time = state.coordinate_spec.time_axis
        if time is None:
            raise ValueError("KS equation requires a time axis")
        spatial = state.coordinate_spec.spatial_axes
        if len(spatial) != 2:
            raise ValueError(
                f"KS 2D form requires exactly 2 spatial axes, got "
                f"{len(spatial)} ({spatial!r})"
            )
        u = state.ops.value(state, self.component)
        grad_u = state.ops.gradient(state, self.component)             # (B, 2)
        u_t = state.ops.derivative(state, self.component, axis=time, order=1)
        lap = state.ops.laplacian(state, self.component)
        bih = state.ops.biharmonic(state, self.component)
        # Nonlinearity: u * |grad u|^2 / 2 = u (u_x^2 + u_y^2) / 2 is the
        # natural conservative-form nonlinearity in 2D KS literature.
        # The dispatch keeps ``c_conv=1`` matching the 1D residual at
        # any-D for ``u(x) = sin(kx)`` (which lives in only one axis).
        grad_sq = (grad_u * grad_u).sum(dim=-1)
        residual = (
            u_t
            + self.c_conv * 0.5 * u * grad_sq  # conservative form (u, |grad u|^2/2)
            + self.c_lap * lap
            + self.c_bih * bih
        )
        return KSOutput(
            residual=residual,
            diag={"mean_sq_residual": float((residual.detach() ** 2).mean())},
        )


def kuramoto_sivashinsky(
    state: FieldState,
    *,
    c_conv: float = 1.0,
    c_lap: float = 1.0,
    c_bih: float = 1.0,
    component: str = "u",
    form: str = "1d",
) -> KSOutput:
    """Stateless one-shot wrapper around :class:`KuramotoSivashinsky`."""
    return KuramotoSivashinsky(
        c_conv=c_conv, c_lap=c_lap, c_bih=c_bih,
        component=component, form=form,
    )(state)


__all__ = ["KuramotoSivashinsky", "kuramoto_sivashinsky"]
