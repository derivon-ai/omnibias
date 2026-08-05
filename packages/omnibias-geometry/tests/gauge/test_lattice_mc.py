# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Fast CPU tests for SU(2) lattice Monte Carlo correctness."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from omnibias.geometry.gauge.lattice.montecarlo import run_lattice_mc
from omnibias.geometry.gauge.lattice.observables import (
    average_plaquette,
    connected_correlator_ensemble,
    effective_mass,
    is_unitary_det_one,
)
from omnibias.geometry.gauge.lattice.su2 import (
    checkerboard_mask,
    heatbath_update_links,
    identity_links,
    normalize_quaternion,
    overrelax_update_links,
    quat_conj,
    quat_to_matrix,
    random_links,
    sweep,
)


def _assert_links_su2(links: torch.Tensor) -> None:
    norm = torch.linalg.vector_norm(links, dim=-1)
    assert torch.allclose(norm, torch.ones_like(norm), atol=1e-10)
    mats = quat_to_matrix(links)
    unitary, det_one = is_unitary_det_one(mats, atol=1e-8)
    assert bool(unitary)
    assert bool(det_one)


def test_updates_preserve_su2_constraints() -> None:
    shape = (4, 4, 4, 4)
    links = random_links(shape, device="cpu")
    _assert_links_su2(links)
    mask = checkerboard_mask(shape, 0, device=torch.device("cpu"), dtype=torch.float64)
    gen = torch.Generator().manual_seed(0)
    heatbath_update_links(links, mu=0, mask=mask, beta=2.0, generator=gen)
    _assert_links_su2(links)
    overrelax_update_links(links, mu=1, mask=~mask)
    _assert_links_su2(links)
    sweep(links, beta=2.0, generator=gen)
    _assert_links_su2(links)


@pytest.mark.slow
@pytest.mark.parametrize("beta", [1.0, 2.3, 4.0])
def test_plaquette_in_unit_interval(beta: float) -> None:
    out = run_lattice_mc(
        lattice_shape=(4, 4, 4, 4),
        beta=beta,
        n_therm=20,
        n_meas=10,
        n_sep=1,
        device="cpu",
        seed=11,
    )
    p = out["avg_plaquette"]
    assert 0.0 < p < 1.0


@pytest.mark.slow
def test_strong_coupling_plaquette() -> None:
    out05 = run_lattice_mc(
        lattice_shape=(4, 4, 4, 4),
        beta=0.5,
        n_therm=30,
        n_meas=20,
        n_sep=1,
        device="cpu",
        seed=42,
    )
    expected05 = 0.5 / 4.0
    tol05 = 0.08 + out05["avg_plaquette_err"]
    assert abs(out05["avg_plaquette"] - expected05) < tol05

    out10 = run_lattice_mc(
        lattice_shape=(4, 4, 4, 4),
        beta=1.0,
        n_therm=30,
        n_meas=20,
        n_sep=1,
        device="cpu",
        seed=43,
    )
    expected10 = 1.0 / 4.0
    tol10 = 0.15 + out10["avg_plaquette_err"]
    assert abs(out10["avg_plaquette"] - expected10) < tol10


@pytest.mark.slow
def test_weak_coupling_monotonicity() -> None:
    betas = [1.0, 2.0, 4.0, 8.0]
    plaquettes = []
    for beta in betas:
        out = run_lattice_mc(
            lattice_shape=(4, 4, 4, 4),
            beta=beta,
            n_therm=25,
            n_meas=15,
            n_sep=1,
            device="cpu",
            seed=99 + int(beta),
        )
        plaquettes.append(out["avg_plaquette"])
    for i in range(len(plaquettes) - 1):
        assert plaquettes[i] < plaquettes[i + 1] + 0.05
    assert plaquettes[-1] > 0.75


@pytest.mark.slow
def test_glueball_correlator_finite_positive() -> None:
    out = run_lattice_mc(
        lattice_shape=(4, 4, 4, 8),
        beta=2.0,
        n_therm=20,
        n_meas=12,
        n_sep=1,
        device="cpu",
        seed=7,
    )
    corr = out["glueball_correlator"]
    assert len(corr) == 8 // 2 + 1
    assert all(math.isfinite(c) for c in corr)
    assert corr[0] > 0.0


def test_connected_correlator_ensemble_no_per_config_zero_sum() -> None:
    """Global vacuum subtraction must not force periodic zero-sum across tau."""
    t_len = 6
    n_meas = 40
    gen = torch.Generator().manual_seed(0)
    o_samples = 50.0 + 3.0 * torch.randn((n_meas, t_len), generator=gen)
    c, err = connected_correlator_ensemble(o_samples)
    assert c[0].item() > 0.0
    assert c[1].item() > -3.0 * err[1].item()
    tau_sum = c[0].item() + 2.0 * c[1].item() + 2.0 * c[2].item() + c[3].item()
    assert abs(tau_sum) > 1.0


@pytest.mark.slow
def test_glueball_correlator_physics_regression() -> None:
    out = run_lattice_mc(
        lattice_shape=(4, 4, 4, 8),
        beta=2.0,
        n_therm=80,
        n_meas=250,
        n_sep=2,
        n_smear=8,
        device="cpu",
        seed=17,
    )
    corr = out["glueball_correlator"]
    err = out["glueball_correlator_err"]
    assert corr[0] > 0.0
    assert corr[1] > 0.0
    assert corr[0] >= corr[1]
    assert corr[1] > -3.0 * err[1]
    m0 = effective_mass(torch.tensor(corr, dtype=torch.float64), tau=0)
    m1 = effective_mass(torch.tensor(corr, dtype=torch.float64), tau=1)
    assert math.isfinite(m0) and m0 > 0.0
    assert math.isfinite(m1) and m1 > 0.0


def test_effective_mass_positive_on_synthetic_exponential() -> None:
    mass = 0.4
    corr = torch.exp(-mass * torch.arange(5, dtype=torch.float64))
    m_eff = effective_mass(corr, tau=1)
    assert m_eff > 0.0
    assert abs(m_eff - mass) < 1e-6


def test_cold_start_plaquette_high_beta() -> None:
    links = identity_links((4, 4, 4, 4))
    p = average_plaquette(links)
    assert abs(p - 1.0) < 1e-12


def test_normalize_quaternion() -> None:
    q = torch.randn(3, 4)
    n = normalize_quaternion(q)
    norms = torch.linalg.vector_norm(n, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms))


@pytest.mark.slow
def test_wilson_loops_in_unit_interval_decreasing() -> None:
    out = run_lattice_mc(
        lattice_shape=(4, 4, 4, 8),
        beta=2.3,
        n_therm=40,
        n_meas=60,
        n_sep=2,
        device="cpu",
        seed=21,
    )
    w = out["wilson_loops"]
    assert 0.0 < w["1x1"]["value"] <= 1.0
    assert 0.0 < w["2x2"]["value"] <= 1.0
    assert w["2x2"]["value"] < w["1x1"]["value"]
    assert w["2x1"]["value"] < w["1x1"]["value"]
    assert w["1x2"]["value"] < w["1x1"]["value"]


@pytest.mark.slow
def test_string_tension_positive() -> None:
    out = run_lattice_mc(
        lattice_shape=(4, 4, 4, 8),
        beta=2.3,
        n_therm=50,
        n_meas=80,
        n_sep=2,
        device="cpu",
        seed=23,
    )
    sigma = out["string_tension"]
    assert math.isfinite(sigma["value"])
    assert sigma["value"] > -sigma["err"]


@pytest.mark.slow
def test_gevp_ground_mass_positive() -> None:
    out = run_lattice_mc(
        lattice_shape=(4, 4, 4, 8),
        beta=2.0,
        n_therm=60,
        n_meas=100,
        n_sep=2,
        device="cpu",
        seed=29,
    )
    mass = out["gevp"]["ground_mass"]["value"]
    assert math.isfinite(mass)
    assert mass > 0.0
