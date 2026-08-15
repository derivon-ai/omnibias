# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Regularized spectral inverse: planted recovery, honesty, Landau G."""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.geometry.gauge._core.data_paths import LatticeLinkField
from omnibias.geometry.gauge._core.landau_gluon import gluon_propagator_p2
from omnibias.geometry.gauge._core.loop_language import identity_numpy_links
from omnibias.geometry.gauge._core.spectral_density import (
    SPECTRAL_RECOVERY_ATOL,
    kallen_lehmann_kernel,
    planted_spectral_vectors,
    reconstruct_spectral_density,
    refuse_spectral_as_mass_gap,
    refuse_unregularized_spectral_inverse,
)


def test_planted_tikhonov_recovers_rho() -> None:
    omega, p2, rho, green = planted_spectral_vectors()
    out = reconstruct_spectral_density(green, omega, p2, method="tikhonov", lam=1e-8)
    assert out["yang_mills_claim"] is False
    assert out["continuum_claim"] is False
    assert out["ill_posed"] is True
    assert out["reconstructed"] is True
    assert float(np.max(np.abs(out["rho"] - rho))) <= SPECTRAL_RECOVERY_ATOL
    kernel = kallen_lehmann_kernel(p2, omega)
    assert kernel.shape == (p2.size, omega.size)


def test_nnls_clip_keeps_planted_rho_nonnegative() -> None:
    omega, p2, rho, green = planted_spectral_vectors()
    out = reconstruct_spectral_density(green, omega, p2, method="nnls_clip", lam=1e-8)
    assert np.all(out["rho"] >= -1e-14)
    assert float(np.max(np.abs(out["rho"] - rho))) <= SPECTRAL_RECOVERY_ATOL


def test_unregularized_inverse_raises() -> None:
    omega, p2, _rho, green = planted_spectral_vectors()
    with pytest.raises(ValueError, match="unregularized"):
        reconstruct_spectral_density(green, omega, p2, lam=0.0)
    with pytest.raises(ValueError, match="unregularized"):
        refuse_unregularized_spectral_inverse(0.0)


def test_mass_gap_claim_raises() -> None:
    omega, p2, _rho, green = planted_spectral_vectors()
    with pytest.raises(ValueError, match="mass-gap"):
        reconstruct_spectral_density(green, omega, p2, yang_mills_claim=True)
    with pytest.raises(ValueError, match="mass-gap"):
        refuse_spectral_as_mass_gap({"yang_mills_claim": True})


def test_landau_g_reconstruct_is_finite_and_unclaimed() -> None:
    field = LatticeLinkField(links=identity_numpy_links((4, 4, 4, 4)))
    table, _report = gluon_propagator_p2(field, already_fixed=True)
    omega = np.linspace(0.5, 2.5, 6)
    p2 = np.maximum(table.values["p2"][:12], 0.2)
    green = table.values["G_p2"][:12]
    out = reconstruct_spectral_density(
        green, omega, p2, source="lattice_landau_G", lam=1e-2
    )
    assert out["source"] == "lattice_landau_G"
    assert np.all(np.isfinite(out["rho"]))
    assert out["yang_mills_claim"] is False
    assert out["continuum_claim"] is False
    assert out["ill_posed"] is True
