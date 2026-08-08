# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Heat MOL reference must be exact and obey the maximum principle.

The four-gap operator benchmark previously marched heat with explicit RK4,
which crosses its stability boundary around diffusivity ~0.14 on a 64-point
grid and silently produced ``max|u| ~ 1e9`` snapshots that still passed an
``isfinite`` check. This module locks the ETDRK4 fix and the physical guard.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from omnibias.pinn.operator.torch.data import make_parametric_heat_slab
from omnibias.pinn.solver.torch.evolution import heat_semidiscrete, method_of_lines
from omnibias.pinn.solver.torch.spectral import SpectralGrid1D

DTYPE = torch.float64


def _analytic_fourier_heat(
    grid: SpectralGrid1D,
    u0: torch.Tensor,
    times: list[float],
    diffusivity: float,
) -> torch.Tensor:
    """Exact Fourier solution of periodic heat from IC ``u0``."""
    uh0 = torch.fft.fft(u0)
    k2 = grid.k**2
    snaps = []
    for t in times:
        uh = uh0 * torch.exp(-float(diffusivity) * k2 * float(t))
        snaps.append(torch.real(torch.fft.ifft(uh)))
    return torch.stack(snaps, dim=0)


@pytest.mark.parametrize("nu", [0.04, 0.12, 0.16, 0.20, 0.24, 1.0, 10.0])
def test_etdrk4_heat_matches_analytic(nu: float) -> None:
    grid = SpectralGrid1D(64, 2.0 * np.pi, dtype=DTYPE)
    x = grid.points()
    u0 = torch.sin(x) + 0.5 * torch.sin(2.0 * x)
    times = [float(t) for t in torch.linspace(0.0, 0.5, 9, dtype=DTYPE)]
    semi = heat_semidiscrete(grid, nu)
    snaps, _ = method_of_lines(semi, u0, times, integrator="etdrk4")
    exact = _analytic_fourier_heat(grid, u0, times, nu)
    err = float((snaps - exact).abs().max())
    assert err < 1e-12, (nu, err)
    assert float(snaps.abs().max()) <= float(u0.abs().max()) * (1.0 + 1e-6) + 1e-12


def test_rk4_heat_violates_maximum_principle_at_nu_0_24() -> None:
    """Document the failure mode the maximum-principle guard catches."""
    grid = SpectralGrid1D(64, 2.0 * np.pi, dtype=DTYPE)
    x = grid.points()
    u0 = torch.sin(x) + 0.5 * torch.sin(2.0 * x)
    times = [float(t) for t in torch.linspace(0.0, 0.5, 9, dtype=DTYPE)]
    semi = heat_semidiscrete(grid, 0.24)
    snaps, _ = method_of_lines(semi, u0, times, integrator="rk4")
    assert float(snaps.abs().max()) > float(u0.abs().max()) * 10.0


def test_parametric_heat_slab_obeys_maximum_principle_ood() -> None:
    """OOD diffusivities that previously blew up under RK4 stay bounded."""
    slab = make_parametric_heat_slab(
        n_samples=6,
        n_grid=64,
        n_sensors=16,
        n_modes=2,
        n_times=9,
        diffusivities=tuple(0.04 + 0.04 * i for i in range(6)),
        seed=1000,
        dtype=DTYPE,
    )
    u0_max = float(slab.sensors.abs().max())  # sensors are IC subsamples
    # Reconstruct per-sample IC max from the full-grid IC via the first snapshot
    # at t=0 (first n_grid values of each sample).
    n_grid = slab.grid.n
    for i in range(slab.values.shape[0]):
        u0 = slab.values[i, :n_grid, 0]
        snaps = slab.values[i, :, 0]
        assert float(snaps.abs().max()) <= float(u0.abs().max()) * (1.0 + 1e-6) + 1e-12
    assert u0_max < 10.0  # sanity: IC amplitude is O(1)
