# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""SU(3) links: unitarity, identity plaquette, 4⁴ smoke MC."""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.geometry.gauge._core.data_paths import SU3LatticeLinkField
from omnibias.geometry.gauge.lattice._core.su3_kernels import (
    average_su3_plaquette,
    identity_su3_links,
    random_su3_links,
    reunitarize,
    su3_landau_overrelax,
)
from omnibias.geometry.gauge.lattice._core.su3_montecarlo import run_su3_lattice_mc
from omnibias.geometry.gauge.lattice.montecarlo import run_lattice_mc


def test_identity_plaquette_is_one() -> None:
    links = identity_su3_links((2, 2, 2, 2))
    field = SU3LatticeLinkField(links=links)
    assert field.spacing == 1.0
    assert average_su3_plaquette(links) == pytest.approx(1.0, abs=1e-12)
    det = np.linalg.det(links)
    np.testing.assert_allclose(np.abs(det), 1.0, atol=1e-12)


def test_reunitarize_restores_su3() -> None:
    rng = np.random.default_rng(1)
    raw = rng.normal(size=(4, 2, 2, 2, 2, 3, 3)) + 1j * rng.normal(
        size=(4, 2, 2, 2, 2, 3, 3)
    )
    u = reunitarize(raw)
    dag = np.conjugate(np.swapaxes(u, -1, -2))
    ident = np.einsum("...ij,...jk->...ik", dag, u)
    eye = np.broadcast_to(np.eye(3), ident.shape)
    np.testing.assert_allclose(ident, eye, atol=1e-8)
    np.testing.assert_allclose(np.abs(np.linalg.det(u)), 1.0, atol=1e-8)


def test_landau_fixes_identity() -> None:
    ident = identity_su3_links((2, 2, 2, 2))
    out = su3_landau_overrelax(ident, n_steps=2)
    np.testing.assert_allclose(out, ident, atol=1e-12)


def test_su3_mc_smoke_4_cubed() -> None:
    out = run_su3_lattice_mc(
        lattice_shape=(4, 4, 4, 4),
        beta=5.5,
        n_therm=1,
        n_meas=2,
        n_sep=1,
        seed=0,
        cold_start=True,
    )
    assert out["gauge_group"] == "su(3)"
    assert out["yang_mills_claim"] is False
    assert out["continuum_claim"] is False
    assert 0.0 < float(out["avg_plaquette"]) <= 1.0 + 1e-12
    dispatched = run_lattice_mc(
        gauge_group="su(3)",
        lattice_shape=(4, 4, 4, 4),
        beta=5.5,
        n_therm=1,
        n_meas=2,
        n_sep=1,
        seed=1,
    )
    assert dispatched["gauge_group"] == "su(3)"
    _ = random_su3_links((2, 2, 2, 2), np.random.default_rng(0))


def test_su3_mc_physical_plaquette_beta_5p7() -> None:
    """SU(3) Wilson at β=5.7 has ⟨P⟩ ≈ 0.55, not a disordered 0.03."""
    out = run_su3_lattice_mc(
        lattice_shape=(4, 4, 4, 4),
        beta=5.7,
        n_therm=80,
        n_meas=20,
        n_sep=1,
        seed=0,
        cold_start=True,
    )
    plaq = float(out["avg_plaquette"])
    assert 0.45 <= plaq <= 0.65
    assert out["yang_mills_claim"] is False
