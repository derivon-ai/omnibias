# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Fast deterministic tests for lattice observables and the heat-bath sampler.

These avoid full Monte-Carlo runs: Wilson-loop geometry identities, the
Polyakov loop, gauge invariance / orbit distance, the exact Kennedy-Pendleton
``q0`` marginal (checked against numerical quadrature), and the GEVP plateau
scan on a synthetic two-mode correlator.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from omnibias.geometry.gauge.lattice._core import kernels
from omnibias.geometry.gauge.lattice.observables import (
    average_plaquette,
    average_polyakov_loop,
    average_wilson_loop,
    gauge_orbit_distance,
    gauge_transform_links,
    gevp_plateau,
    plaquette_trace,
    polyakov_loop,
    wilson_loop_trace,
)
from omnibias.geometry.gauge.lattice.su2 import (
    _sample_q0_kp,
    identity_links,
    normalize_quaternion,
    random_links,
)


def _random_site_gauge(shape: tuple[int, ...], *, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    g = torch.randn((*shape, 4), dtype=torch.float64, generator=gen)
    return normalize_quaternion(g)


# ---------------------------------------------------------------------------
# Wilson-loop geometry
# ---------------------------------------------------------------------------
def test_wilson_1x1_equals_plaquette() -> None:
    """The unit R=T=1 Wilson loop in the (mu, nu) plane is the plaquette."""
    shape = (4, 4, 4, 4)
    links = random_links(shape, device="cpu")
    for mu in range(4):
        for nu in range(4):
            if nu == mu:
                continue
            w11 = wilson_loop_trace(links, mu, 1, 1, t_dir=nu)
            plaq = plaquette_trace(links, mu, nu)
            assert torch.allclose(w11, plaq, atol=1e-12)


def test_wilson_cold_start_is_one() -> None:
    links = identity_links((4, 4, 4, 6))
    for r, t in ((1, 1), (2, 1), (1, 2), (2, 3)):
        w = wilson_loop_trace(links, 0, r, t, t_dir=3)
        assert torch.allclose(w, torch.ones_like(w), atol=1e-12)


def test_wilson_trace_bounded() -> None:
    links = random_links((4, 4, 4, 6), device="cpu")
    for r, t in ((1, 1), (2, 2), (2, 3)):
        w = wilson_loop_trace(links, 0, r, t, t_dir=3)
        assert torch.all(w <= 1.0 + 1e-9)
        assert torch.all(w >= -1.0 - 1e-9)


# ---------------------------------------------------------------------------
# Polyakov loop + gauge invariance / orbit distance
# ---------------------------------------------------------------------------
def test_polyakov_cold_start_is_one() -> None:
    links = identity_links((4, 4, 4, 4))
    p = polyakov_loop(links)
    assert torch.allclose(p, torch.ones_like(p), atol=1e-12)
    assert average_polyakov_loop(links) == pytest.approx(1.0, abs=1e-12)


def test_gauge_invariance_of_loops_and_orbit_distance() -> None:
    shape = (4, 4, 4, 4)
    links = random_links(shape, device="cpu")
    g = _random_site_gauge(shape, seed=7)
    transformed = gauge_transform_links(links, g)

    assert average_plaquette(transformed) == pytest.approx(average_plaquette(links), abs=1e-10)
    assert average_wilson_loop(transformed, 2, 2) == pytest.approx(
        average_wilson_loop(links, 2, 2), abs=1e-10
    )
    assert average_polyakov_loop(transformed) == pytest.approx(
        average_polyakov_loop(links), abs=1e-10
    )
    # Gauge-orbit distance proxy is invariant: a gauge copy is distance ~0.
    assert gauge_orbit_distance(links, transformed) < 1e-9
    # A genuinely different config is at positive distance.
    other = random_links(shape, device="cpu")
    assert gauge_orbit_distance(links, other) > 1e-3


# ---------------------------------------------------------------------------
# Exact Kennedy-Pendleton q0 marginal
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("w", [0.5, 2.0, 5.0])
def test_kennedy_pendleton_q0_marginal(w: float) -> None:
    """Sampled q0 mean matches the Haar-weighted heat-bath marginal mean.

    Target density p(q0) ~ sqrt(1-q0^2) exp(w q0); the mean is computed by
    dependency-free quadrature and compared to the sampler average.
    """
    n = 200_000
    a = torch.full((n,), float(w), dtype=torch.float64)  # beta=1 so w = a
    gen = torch.Generator().manual_seed(2024)
    q0 = _sample_q0_kp(a, beta=1.0, generator=gen)
    assert torch.all(q0 <= 1.0 + 1e-9)
    assert torch.all(q0 >= -1.0 - 1e-9)

    grid = np.linspace(-1.0, 1.0, 400_001)
    weight = np.sqrt(np.clip(1.0 - grid**2, 0.0, None)) * np.exp(w * grid)
    mean_theory = np.trapezoid(grid * weight, grid) / np.trapezoid(weight, grid)
    var_theory = (
        np.trapezoid(grid**2 * weight, grid) / np.trapezoid(weight, grid) - mean_theory**2
    )
    sem = float(np.sqrt(var_theory / n))
    assert abs(float(q0.mean()) - mean_theory) < 6.0 * sem + 1e-3


# ---------------------------------------------------------------------------
# GEVP plateau / generalized eigenvalue
# ---------------------------------------------------------------------------
def test_gevp_ground_lambda_exact_scaling() -> None:
    """C(t1) = exp(-m dt) C(t0) implies every gen-eigenvalue is exp(-m dt)."""
    gen = torch.Generator().manual_seed(0)
    b = torch.randn((3, 3), dtype=torch.float64, generator=gen)
    c0 = b @ b.T + 3.0 * torch.eye(3, dtype=torch.float64)
    m, dt = 0.37, 2
    lam = np.exp(-m * dt)
    c1 = lam * c0
    rec = kernels.gevp_ground_lambda(torch, c0, c1)
    assert rec == pytest.approx(lam, rel=1e-9)
    assert -np.log(rec) / dt == pytest.approx(m, rel=1e-9)


def test_gevp_plateau_recovers_two_mode_ground_mass() -> None:
    """Synthetic two-mode correlator: the plateau recovers the ground mass."""
    t_len, m0, m1, n_meas = 16, 0.25, 0.85, 800
    t = torch.arange(t_len, dtype=torch.float64)
    sep = (t[:, None] - t[None, :]).abs()
    dist = torch.minimum(sep, t_len - sep)
    eye = torch.eye(t_len, dtype=torch.float64)

    def _draw(mass: float, seed: int) -> torch.Tensor:
        cov = torch.exp(-mass * dist) + 1e-9 * eye
        chol = torch.linalg.cholesky(cov)
        z = torch.randn((n_meas, t_len), dtype=torch.float64, generator=torch.Generator().manual_seed(seed))
        return z @ chol.T

    base0 = _draw(m0, 1)
    base1 = _draw(m1, 2)
    mix = torch.tensor([[1.0, 0.4], [0.3, 1.0]], dtype=torch.float64)
    o_samples = torch.stack(
        [mix[a, 0] * base0 + mix[a, 1] * base1 for a in range(2)], dim=1
    )

    scan = gevp_plateau(
        o_samples,
        smear_levels=(1, 2),
        t0_values=(0, 1, 2, 3),
        dt_values=(1, 2),
    )
    assert scan["points"]
    assert scan["plateau"]["value"] == pytest.approx(m0, rel=0.3)
