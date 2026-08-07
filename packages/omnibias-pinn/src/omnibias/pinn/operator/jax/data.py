# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Operator-learning datasets from the existing spectral MOL reference (JAX).

Thin twin of :mod:`omnibias.pinn.operator.torch.data`: the reference march is
the JAX ``method_of_lines``.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array
from omnibias.pinn.operator._core.sensors import SensorGrid, sample_fourier_ics
from omnibias.pinn.solver.jax.evolution import (
    burgers_semidiscrete,
    heat_semidiscrete,
    kuramoto_sivashinsky_semidiscrete,
    method_of_lines,
)
from omnibias.pinn.solver.jax.spectral import SpectralGrid1D


@dataclass(frozen=True)
class OperatorSlab:
    sensors: Array
    coords: Array
    values: Array
    grid: SpectralGrid1D


def _space_time_coords(grid: SpectralGrid1D, times: Array) -> Array:
    x = grid.points()
    xs = jnp.tile(x, times.size)
    ts = jnp.repeat(times, x.size)
    return jnp.stack([xs, ts], axis=-1)


def make_heat_slab(
    *,
    n_samples: int = 32,
    n_grid: int = 64,
    n_sensors: int = 32,
    n_modes: int = 4,
    amplitude: float = 1.0,
    diffusivity: float = 0.1,
    t_final: float = 0.5,
    n_times: int = 11,
    seed: int = 0,
) -> OperatorSlab:
    length = 2.0 * jnp.pi
    grid = SpectralGrid1D(n_grid, float(length))
    fine = SensorGrid(points=grid.points(), length=float(length))
    u0 = jnp.asarray(
        sample_fourier_ics(
            n_samples, fine, n_modes=n_modes, amplitude=amplitude, seed=seed
        )
    )
    idx = jnp.linspace(0, n_grid, num=n_sensors, endpoint=False).astype(jnp.int32)
    sensors = u0[:, idx]
    times = jnp.linspace(0.0, t_final, n_times)
    times_seq = [float(t) for t in times]
    semi = heat_semidiscrete(grid, diffusivity)
    snaps_list = []
    for i in range(n_samples):
        snaps, _ = method_of_lines(semi, u0[i], times_seq, integrator="rk4")
        snaps_list.append(snaps)
    snaps_b = jnp.stack(snaps_list, axis=0)
    if not bool(jnp.isfinite(snaps_b).all()):
        raise RuntimeError(
            "heat MOL reference produced non-finite values; reduce amplitude "
            "or n_modes, or increase n_grid"
        )
    coords = _space_time_coords(grid, times)
    values = snaps_b.reshape(n_samples, -1, 1)
    return OperatorSlab(sensors=sensors, coords=coords, values=values, grid=grid)


def make_burgers_slab(
    *,
    n_samples: int = 16,
    n_grid: int = 64,
    n_sensors: int = 32,
    n_modes: int = 2,
    amplitude: float = 0.5,
    viscosity: float = 0.05,
    t_final: float = 0.5,
    n_times: int = 11,
    seed: int = 0,
) -> OperatorSlab:
    length = 2.0 * jnp.pi
    grid = SpectralGrid1D(n_grid, float(length))
    fine = SensorGrid(points=grid.points(), length=float(length))
    u0 = jnp.asarray(
        sample_fourier_ics(
            n_samples, fine, n_modes=n_modes, amplitude=amplitude, seed=seed
        )
    )
    idx = jnp.linspace(0, n_grid, num=n_sensors, endpoint=False).astype(jnp.int32)
    sensors = u0[:, idx]
    times = jnp.linspace(0.0, t_final, n_times)
    times_seq = [float(t) for t in times]
    semi = burgers_semidiscrete(grid, viscosity)
    snaps_list = []
    for i in range(n_samples):
        snaps, _ = method_of_lines(semi, u0[i], times_seq, integrator="etdrk4")
        snaps_list.append(snaps)
    snaps_b = jnp.stack(snaps_list, axis=0)
    if not bool(jnp.isfinite(snaps_b).all()):
        raise RuntimeError(
            "Burgers MOL reference produced non-finite values; reduce amplitude "
            "or n_modes, or increase n_grid / viscosity"
        )
    coords = _space_time_coords(grid, times)
    values = snaps_b.reshape(n_samples, -1, 1)
    return OperatorSlab(sensors=sensors, coords=coords, values=values, grid=grid)


def make_ks_slab(
    *,
    n_samples: int = 8,
    n_grid: int = 128,
    n_sensors: int = 32,
    n_modes: int = 2,
    amplitude: float = 0.5,
    t_final: float = 1.0,
    n_times: int = 11,
    seed: int = 0,
) -> OperatorSlab:
    """Periodic Kuramoto-Sivashinsky slabs from spectral MOL (ETDRK4)."""
    import math

    length = 32.0 * math.pi
    grid = SpectralGrid1D(n_grid, float(length))
    fine = SensorGrid(points=grid.points(), length=float(length))
    u0 = jnp.asarray(
        sample_fourier_ics(
            n_samples, fine, n_modes=n_modes, amplitude=amplitude, seed=seed
        )
    )
    idx = jnp.linspace(0, n_grid, num=n_sensors, endpoint=False).astype(jnp.int32)
    sensors = u0[:, idx]
    times = jnp.linspace(0.0, t_final, n_times)
    times_seq = [float(t) for t in times]
    semi = kuramoto_sivashinsky_semidiscrete(grid)
    snaps_list = []
    for i in range(n_samples):
        snaps, _ = method_of_lines(semi, u0[i], times_seq, integrator="etdrk4")
        snaps_list.append(snaps)
    snaps_b = jnp.stack(snaps_list, axis=0)
    if not bool(jnp.isfinite(snaps_b).all()):
        raise RuntimeError(
            "KS MOL reference produced non-finite values; reduce amplitude "
            "or n_modes, or shorten t_final"
        )
    coords = _space_time_coords(grid, times)
    values = snaps_b.reshape(n_samples, -1, 1)
    return OperatorSlab(sensors=sensors, coords=coords, values=values, grid=grid)


__all__ = [
    "OperatorSlab",
    "make_burgers_slab",
    "make_heat_slab",
    "make_ks_slab",
]
