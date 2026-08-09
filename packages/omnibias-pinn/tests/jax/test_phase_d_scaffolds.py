# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Phase D scaffold tests: viscous enclosure + boundary-free axisym."""

from __future__ import annotations

from omnibias.pinn.certified.viscous_perturbation import (
    verify_viscous_perturbation_enclosure,
    viscous_perturbation_enclosure,
)
from omnibias.pinn.jax.discovery.euler3d_axisym import (
    AxisymFreeDiscoveryConfig,
    run_euler3d_axisym_free_discovery,
)


def test_viscous_enclosure_closes_for_small_nu() -> None:
    cert = viscous_perturbation_enclosure(
        inviscid_residual_sup=1e-4,
        viscosity=1e-6,
        enstrophy_bound=10.0,
        window_length=1.0,
        tol=1e-2,
    )
    assert cert["honesty"]["continuum_navier_stokes_claim"] is False
    assert cert["enclosure_closed"] is True
    report = verify_viscous_perturbation_enclosure(cert)
    assert report["replay_match"] is True


def test_viscous_enclosure_blocks_large_nu() -> None:
    cert = viscous_perturbation_enclosure(
        inviscid_residual_sup=0.1,
        viscosity=1.0,
        enstrophy_bound=10.0,
        window_length=1.0,
        tol=1e-2,
    )
    assert cert["enclosure_closed"] is False


def test_axisym_free_discovery_honesty() -> None:
    out = run_euler3d_axisym_free_discovery(AxisymFreeDiscoveryConfig(n_radial=4, n_axial=4))
    assert out["domain"]["wall_stabilization"] is False
    assert out["honesty"]["navier_stokes_proof_claim"] is False
    assert out["honesty"]["continuum_navier_stokes_claim"] is False
