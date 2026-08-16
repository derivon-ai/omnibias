# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Joint discoverer shares σ across Wilson and a spectrum channel."""

from __future__ import annotations

import pytest
from omnibias.geometry.gauge._core.ensemble_language import finite_t_scan_table
from omnibias.symbolic.ensemble_families import (
    JointLawDiscoverer,
    planted_spectrum_from_sigma,
    planted_wilson_area_table,
)


def test_joint_recovers_shared_sigma() -> None:
    sigma = 0.2
    out = JointLawDiscoverer().discover(
        {
            "wilson": planted_wilson_area_table(sigma=sigma, kappa=0.04),
            "spectrum": planted_spectrum_from_sigma(sigma=sigma),
            "finite_t": finite_t_scan_table(sigma0=sigma),
        }
    )
    assert out.sigma == pytest.approx(sigma, rel=5e-2)
    assert out.spectrum_mass == pytest.approx(sigma**0.5, rel=5e-2)
    assert out.melting_consistent is True
    assert out.passed is True
    assert out.yang_mills_claim is False
    assert out.continuum_claim is False


def test_joint_torelon_and_gevp_verifier() -> None:
    sigma = 0.25
    length = 4.0
    out = JointLawDiscoverer().discover(
        {
            "wilson": planted_wilson_area_table(sigma=sigma, kappa=0.0),
            "spectrum": planted_spectrum_from_sigma(
                sigma=sigma, torelon_length=length
            ),
        },
        gevp_mass=sigma * length,
        torelon_length=length,
    )
    assert out.predicted_mass == pytest.approx(1.0, rel=5e-2)
    assert out.passed is True
