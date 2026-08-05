# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Time integrators for the method-of-lines drivers (torch).

* :func:`rk4_step` / :func:`euler_step` -- standard **numerical** explicit steps.
* :func:`implicit_linear_step` -- implicit Euler / Crank-Nicolson for a *linear*
  problem with a diagonal Fourier symbol (**numerical**, unconditionally stable).
* :func:`linear_jet_step` / :func:`burgers_jet_step` -- the **high-order**
  jet-Taylor integrator. It builds the local time-Taylor expansion from the
  exact high-order time derivatives; the nonlinear (Burgers) term uses
  :func:`omnibias.torch.jet_mv.jet_multiply` (the jet-level Leibniz rule) for the
  ``u * u_x`` product. Local truncation is ``O(dt^{order+1})`` -- high-order, not
  exact-in-time.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from omnibias.pinn.solver.torch.spectral import SpectralGrid1D
from omnibias.torch.jet_mv import jet_multiply
from torch import Tensor


def euler_step(rhs: Callable[[Tensor], Tensor], u: Tensor, dt: float) -> Tensor:
    return u + dt * rhs(u)


def rk4_step(rhs: Callable[[Tensor], Tensor], u: Tensor, dt: float) -> Tensor:
    """Classical explicit fourth-order Runge-Kutta step."""
    k1 = rhs(u)
    k2 = rhs(u + 0.5 * dt * k1)
    k3 = rhs(u + 0.5 * dt * k2)
    k4 = rhs(u + dt * k3)
    return u + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def implicit_linear_step(
    grid: SpectralGrid1D,
    symbol: Tensor,
    u: Tensor,
    dt: float,
    *,
    scheme: str = "implicit_euler",
) -> Tensor:
    """One implicit step for ``u_t = L u`` with diagonal Fourier symbol ``L``."""
    uh = torch.fft.fft(u)
    lam = symbol.to(uh.dtype)
    if scheme == "implicit_euler":
        uh = uh / (1.0 - dt * lam)
    elif scheme == "crank_nicolson":
        uh = uh * (1.0 + 0.5 * dt * lam) / (1.0 - 0.5 * dt * lam)
    else:
        raise ValueError(f"unknown implicit scheme {scheme!r}")
    return torch.fft.ifft(uh).real


def linear_jet_step(
    apply_L: Callable[[Tensor], Tensor], u0: Tensor, dt: float, order: int
) -> Tensor:
    """Jet-Taylor step for a *linear* RHS ``u_t = L u``.

    Uses ``a_k = L a_{k-1} / k`` (so ``a_k = L^k u0 / k!``) and sums
    ``sum_k a_k dt^k`` -- the truncated action of ``exp(dt L)``.
    """
    a = u0
    result = u0
    for k in range(1, order + 1):
        a = apply_L(a) / k
        result = result + a * (dt ** k)
    return result


def burgers_jet_step(
    grid: SpectralGrid1D, u0: Tensor, dt: float, order: int, viscosity: float
) -> Tensor:
    """Jet-Taylor step for viscous Burgers ``u_t = nu u_xx - u u_x``.

    Bootstraps the time-Taylor coefficients ``a_k`` (``a_k = u^{(k)}(t)/k!``);
    the nonlinear ``u u_x`` term's time-jet is the Cauchy product computed by
    :func:`omnibias.torch.jet_mv.jet_multiply` (``dim=1`` = the time variable).
    """
    a = torch.zeros((order + 1, *u0.shape), dtype=u0.dtype, device=u0.device)
    a[0] = u0
    for k in range(order):
        dx_a = grid.dx(a)
        dxx_a = grid.dxx(a)
        product = jet_multiply(a, dx_a, dim=1, order=order)  # time-jet of u * u_x
        r_k = viscosity * dxx_a[k] - product[k]
        a[k + 1] = r_k / (k + 1)
    u = a[0].clone()
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
