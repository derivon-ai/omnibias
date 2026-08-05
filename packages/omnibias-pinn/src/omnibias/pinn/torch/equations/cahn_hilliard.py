# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Cahn-Hilliard equation residual.

The Cahn-Hilliard equation in conservative form is

.. math::

    c_t \\;=\\; M\\, \\Delta \\mu, \\qquad
    \\mu \\;=\\; f'(c) \\;-\\; \\kappa\\, \\Delta c.

Eliminating :math:`\\mu` and using the chain rule
:math:`\\Delta f'(c) = f''(c)\\, \\Delta c + f'''(c)\\, |\\nabla c|^2`
gives the *omnibias-friendly* residual

.. math::

    R(c) = c_t \\;-\\; M\\big[ f''(c)\\, \\Delta c + f'''(c)\\, |\\nabla c|^2 \\big]
        \\;+\\; M\\, \\kappa\\, \\Delta^2 c.

Each term is a closed-form derivative of the field; we just need the
potential :math:`f`'s second and third derivatives.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from omnibias.pinn._core.state import FieldState
from omnibias.pinn.torch.equations._types import CHOutput
from torch import Tensor


class Potential(Protocol):
    """Free-energy density exposing the three derivatives the CH residual needs."""

    def f(self, c: Tensor) -> Tensor: ...

    def df(self, c: Tensor) -> Tensor: ...

    def d2f(self, c: Tensor) -> Tensor: ...

    def d3f(self, c: Tensor) -> Tensor: ...


@dataclass(frozen=True)
class GinzburgLandauPotential:
    """``f(c) = W * (c^2 - 1)^2 / 4`` -- canonical double-well potential."""

    W: float = 1.0

    def f(self, c: Tensor) -> Tensor:
        x2 = c * c - 1.0
        return self.W * 0.25 * x2 * x2

    def df(self, c: Tensor) -> Tensor:
        return self.W * (c * c * c - c)

    def d2f(self, c: Tensor) -> Tensor:
        return self.W * (3.0 * c * c - 1.0)

    def d3f(self, c: Tensor) -> Tensor:
        return self.W * 6.0 * c


@dataclass
class CahnHilliard:
    """Configurable Cahn-Hilliard residual.

    Parameters
    ----------
    M
        Mobility coefficient. Default 1.0.
    kappa
        Gradient-energy coefficient. Default 1e-3.
    component
        Field component name. Default ``"c"``.
    potential
        Object exposing ``f, df, d2f, d3f``. Default
        :class:`GinzburgLandauPotential`.
    forcing
        Optional ``f(state) -> Tensor`` of shape ``(B,)`` (sourced
        Cahn-Hilliard, e.g. for non-conservative variants).
    """

    M: float = 1.0
    kappa: float = 1e-3
    component: str = "c"
    potential: Potential = None  # type: ignore[assignment]  # default below
    forcing: Callable[[FieldState], Tensor] | None = None

    def __post_init__(self) -> None:
        if self.potential is None:
            object.__setattr__(self, "potential", GinzburgLandauPotential())

    def __call__(self, state: FieldState) -> CHOutput:
        time = state.coordinate_spec.time_axis
        if time is None:
            raise ValueError("Cahn-Hilliard equation requires a time axis")
        c = state.ops.value(state, self.component)
        grad_c = state.ops.gradient(state, self.component)             # (B, D)
        lap_c = state.ops.laplacian(state, self.component)
        bih_c = state.ops.biharmonic(state, self.component)
        c_t = state.ops.derivative(state, self.component, axis=time, order=1)

        f2 = self.potential.d2f(c)
        f3 = self.potential.d3f(c)
        grad_sq = (grad_c * grad_c).sum(dim=-1)

        Delta_fprime = f2 * lap_c + f3 * grad_sq
        residual = c_t - self.M * Delta_fprime + self.M * self.kappa * bih_c
        if self.forcing is not None:
            residual = residual - self.forcing(state)
        return CHOutput(
            residual=residual,
            diag={"mean_sq_residual": float((residual.detach() ** 2).mean())},
        )


def cahn_hilliard(
    state: FieldState,
    *,
    M: float = 1.0,
    kappa: float = 1e-3,
    component: str = "c",
    potential: Potential | None = None,
    forcing: Callable[[FieldState], Tensor] | None = None,
) -> CHOutput:
    """Stateless one-shot wrapper around :class:`CahnHilliard`."""
    return CahnHilliard(
        M=M, kappa=kappa, component=component,
        potential=potential if potential is not None else GinzburgLandauPotential(),
        forcing=forcing,
    )(state)


__all__ = [
    "CahnHilliard",
    "GinzburgLandauPotential",
    "Potential",
    "cahn_hilliard",
]
