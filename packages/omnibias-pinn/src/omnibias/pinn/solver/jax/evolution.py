# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Time-dependent drivers (jax): space-time collocation + method-of-lines."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
from omnibias.pinn.solver._core.system import System
from omnibias.pinn.solver._core.taxonomy import Linearity
from omnibias.pinn.solver.jax import integrators as _int
from omnibias.pinn.solver.jax import stiff as _stiff
from omnibias.pinn.solver.jax._solution import FieldSolution, GridSolution
from omnibias.pinn.solver.jax.spectral import SpectralGrid1D
from omnibias.pinn.solver.jax.steady import solve_least_squares


def solve_evolution(system: System, *, method: str = "collocation", **kwargs: Any) -> FieldSolution:
    """Solve a time-dependent ``System`` by space-time collocation (linear->lstsq)."""
    if method != "collocation":
        raise ValueError(
            "solve_evolution supports method='collocation'; use method_of_lines "
            "for the spectral method-of-lines route"
        )
    if not system.is_time_dependent():
        raise ValueError("solve_evolution requires a time-dependent system")
    if system.linearity is not Linearity.LINEAR:
        raise ValueError(
            "the jax backend's space-time collocation supports linear systems in "
            "v1; use the torch backend for nonlinear residual minimisation"
        )
    return solve_least_squares(system, **kwargs)


@dataclass
class SemiDiscrete:
    """A semi-discrete RHS ``u_t = rhs(u)`` on a spectral grid (jax twin).

    ``symbol`` is the diagonal Fourier multiplier of the stiff *linear* part and
    ``nonlinear`` is the rest, so ``rhs(u) = L u + nonlinear(u)`` -- the split
    the exponential and IMEX integrators need. A problem with a ``symbol`` and
    no ``nonlinear`` is linear, and only then may an implicit linear step be
    used, since otherwise it would silently drop the nonlinear term.
    """

    rhs: Callable[[Any], Any]
    grid: SpectralGrid1D
    n_fields: int = 1
    symbol: Any | None = None
    jet_step: Callable[[Any, float, int], Any] | None = None
    nonlinear: Callable[[Any], Any] | None = None

    @property
    def is_linear(self) -> bool:
        """Whether the whole right-hand side is the linear symbol."""
        return self.symbol is not None and self.nonlinear is None


def heat_semidiscrete(grid: SpectralGrid1D, diffusivity: float) -> SemiDiscrete:
    d = float(diffusivity)

    def rhs(u: Any) -> Any:
        return d * grid.dxx(u)

    def jet_step(u: Any, dt: float, order: int) -> Any:
        return _int.linear_jet_step(lambda w: d * grid.dxx(w), u, dt, order)

    return SemiDiscrete(rhs=rhs, grid=grid, symbol=-d * grid.k ** 2, jet_step=jet_step)


def advection_diffusion_semidiscrete(
    grid: SpectralGrid1D, velocity: float, diffusivity: float
) -> SemiDiscrete:
    a = float(velocity)
    d = float(diffusivity)

    def rhs(u: Any) -> Any:
        return -a * grid.dx(u) + d * grid.dxx(u)

    def jet_step(u: Any, dt: float, order: int) -> Any:
        return _int.linear_jet_step(lambda w: -a * grid.dx(w) + d * grid.dxx(w), u, dt, order)

    symbol = -1j * a * grid.k - d * grid.k ** 2
    return SemiDiscrete(rhs=rhs, grid=grid, symbol=symbol, jet_step=jet_step)


def burgers_semidiscrete(grid: SpectralGrid1D, viscosity: float) -> SemiDiscrete:
    nu = float(viscosity)

    def rhs(u: Any) -> Any:
        return nu * grid.dxx(u) - u * grid.dx(u)

    def nonlinear(u: Any) -> Any:
        return -u * grid.dx(u)

    def jet_step(u: Any, dt: float, order: int) -> Any:
        return _int.burgers_jet_step(grid, u, dt, order, nu)

    # The viscous term is the stiff, diagonal half; the advective term is not.
    return SemiDiscrete(
        rhs=rhs,
        grid=grid,
        symbol=-nu * grid.k**2,
        jet_step=jet_step,
        nonlinear=nonlinear,
    )


def kuramoto_sivashinsky_semidiscrete(grid: SpectralGrid1D) -> SemiDiscrete:
    r"""``u_t = -u u_x - u_xx - u_xxxx`` -- the canonical stiff spectral test.

    The fourth-derivative term makes the symbol ``k^2 - k^4`` scale like the
    fourth power of the wavenumber, so an explicit step is stable only for
    ``dt ~ dx^4`` while the solution itself evolves on ``O(1)`` timescales. That
    gap is exactly what an exponential integrator closes.
    """
    symbol = grid.k**2 - grid.k**4

    def nonlinear(u: Any) -> Any:
        return -0.5 * grid.dx(u * u)

    def rhs(u: Any) -> Any:
        uh = jnp.fft.fft(u)
        return jnp.real(jnp.fft.ifft(uh * symbol.astype(uh.dtype))) + nonlinear(u)

    return SemiDiscrete(rhs=rhs, grid=grid, symbol=symbol, nonlinear=nonlinear)


def reaction_diffusion_semidiscrete(
    grid: SpectralGrid1D,
    diffusivities: tuple[float, float],
    reaction: Callable[[Any, Any], tuple[Any, Any]],
) -> SemiDiscrete:
    du, dv = float(diffusivities[0]), float(diffusivities[1])

    def rhs(state: Any) -> Any:
        u, v = state[0], state[1]
        ru, rv = reaction(u, v)
        return jnp.stack([du * grid.dxx(u) + ru, dv * grid.dxx(v) + rv], axis=0)

    return SemiDiscrete(rhs=rhs, grid=grid, n_fields=2)


def method_of_lines(
    semi: SemiDiscrete,
    u0: Any,
    times: Sequence[float],
    *,
    integrator: str = "rk4",
    order: int = 6,
) -> tuple[Any, Any]:
    """March ``u_t = semi.rhs(u)`` over ``times``; return ``(snapshots, times)``.

    ``integrator`` is ``"rk4"`` / ``"euler"`` (explicit), ``"jet_taylor"`` (the
    high-order Taylor step), ``"implicit_euler"`` / ``"crank_nicolson"``
    (implicit baselines, **linear** problems only), or the stiff splittings
    ``"etdrk4"`` / ``"imex_euler"`` / ``"imex_cnab2"`` for ``u_t = L u + N(u)``.
    """
    ts = [float(t) for t in times]
    if len(ts) < 2:
        raise ValueError("times must contain at least two entries")
    stiff_schemes = ("etdrk4", "imex_euler", "imex_cnab2")
    if integrator in stiff_schemes and semi.symbol is None:
        raise ValueError(
            f"{integrator!r} splits off a linear Fourier symbol; this problem "
            "has none"
        )
    nonlinear = semi.nonlinear if semi.nonlinear is not None else jnp.zeros_like
    u = u0
    snaps = [u0]
    previous_n: Any | None = None
    for i in range(len(ts) - 1):
        dt = ts[i + 1] - ts[i]
        if integrator == "rk4":
            u = _int.rk4_step(semi.rhs, u, dt)
        elif integrator == "euler":
            u = _int.euler_step(semi.rhs, u, dt)
        elif integrator == "jet_taylor":
            if semi.jet_step is None:
                raise ValueError("this problem has no jet-Taylor step")
            u = semi.jet_step(u, dt, order)
        elif integrator in ("implicit_euler", "crank_nicolson"):
            if not semi.is_linear:
                raise ValueError(
                    "implicit schemes require a linear Fourier symbol and no "
                    "nonlinear part; use 'etdrk4' or 'imex_cnab2' for a split "
                    "problem, which integrate the nonlinear term rather than "
                    "dropping it"
                )
            u = _int.implicit_linear_step(semi.grid, semi.symbol, u, dt, scheme=integrator)
        elif integrator in stiff_schemes:
            if integrator == "etdrk4":
                u = _stiff.etdrk4_step(semi.symbol, nonlinear, u, dt)
            elif integrator == "imex_euler":
                u = _stiff.imex_euler_step(semi.symbol, nonlinear, u, dt)
            else:
                u, previous_n = _stiff.imex_cnab2_step(
                    semi.symbol, nonlinear, u, dt, previous_nonlinear=previous_n
                )
        else:
            raise ValueError(f"unknown integrator {integrator!r}")
        snaps.append(u)
    return jnp.stack(snaps, axis=0), jnp.asarray(ts, dtype=u0.dtype)


def grid_solution(
    snapshots: Any, times: Any, grid: SpectralGrid1D, names: Sequence[str]
) -> GridSolution:
    if len(names) == 1:
        values = {names[0]: snapshots}
    else:
        values = {name: snapshots[:, i] for i, name in enumerate(names)}
    return GridSolution(times=times, x=grid.x, values=values, method="method_of_lines")


__all__ = [
    "SemiDiscrete",
    "advection_diffusion_semidiscrete",
    "burgers_semidiscrete",
    "grid_solution",
    "heat_semidiscrete",
    "kuramoto_sivashinsky_semidiscrete",
    "method_of_lines",
    "reaction_diffusion_semidiscrete",
    "solve_evolution",
]
