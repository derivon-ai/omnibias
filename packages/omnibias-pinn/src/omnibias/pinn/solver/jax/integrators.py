# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Time integrators for the jax method-of-lines (twins of the torch ones).

The high-order jet-Taylor integrator uses
:func:`omnibias.jax.jet_mv.jet_multiply` for the nonlinear (Burgers) product,
exactly as the torch twin uses ``omnibias.torch.jet_mv.jet_multiply``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
from omnibias.jax.jet_mv import jet_multiply
from omnibias.pinn.solver.jax.spectral import SpectralGrid1D


def euler_step(rhs: Callable[[Any], Any], u: Any, dt: float) -> Any:
    return u + dt * rhs(u)


def rk4_step(rhs: Callable[[Any], Any], u: Any, dt: float) -> Any:
    k1 = rhs(u)
    k2 = rhs(u + 0.5 * dt * k1)
    k3 = rhs(u + 0.5 * dt * k2)
    k4 = rhs(u + dt * k3)
    return u + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def implicit_linear_step(
    grid: SpectralGrid1D, symbol: Any, u: Any, dt: float, *, scheme: str = "implicit_euler"
) -> Any:
    uh = jnp.fft.fft(u)
    lam = symbol.astype(uh.dtype)
    if scheme == "implicit_euler":
        uh = uh / (1.0 - dt * lam)
    elif scheme == "crank_nicolson":
        uh = uh * (1.0 + 0.5 * dt * lam) / (1.0 - 0.5 * dt * lam)
    else:
        raise ValueError(f"unknown implicit scheme {scheme!r}")
    return jnp.real(jnp.fft.ifft(uh))


def linear_jet_step(apply_L: Callable[[Any], Any], u0: Any, dt: float, order: int) -> Any:
    a = u0
    result = u0
    for k in range(1, order + 1):
        a = apply_L(a) / k
        result = result + a * (dt ** k)
    return result


def burgers_jet_step(
    grid: SpectralGrid1D, u0: Any, dt: float, order: int, viscosity: float
) -> Any:
    a = jnp.zeros((order + 1, *u0.shape), dtype=u0.dtype)
    a = a.at[0].set(u0)
    for k in range(order):
        dx_a = grid.dx(a)
        dxx_a = grid.dxx(a)
        product = jet_multiply(a, dx_a, dim=1, order=order)
        r_k = viscosity * dxx_a[k] - product[k]
        a = a.at[k + 1].set(r_k / (k + 1))
    u = a[0]
    power = 1.0
    for k in range(1, order + 1):
        power *= dt
        u = u + a[k] * power
    return u


__all__ = [
    "burgers_jet_step",
    "euler_step",
    "implicit_linear_step",
    "linear_jet_step",
    "rk4_step",
]
