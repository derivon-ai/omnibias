# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Coupled Path B system discoverer on planted tables."""

from __future__ import annotations

import pytest
from omnibias.geometry.gauge._core.ensemble_language import finite_t_scan_table
from omnibias.symbolic.ensemble_families import (
    CoupledConfinementDiscoverer,
    planted_decoupling_table,
    planted_wilson_area_table,
)
from omnibias.symbolic.ensemble_piecewise import planted_hybrid_wilson_table


def test_coupled_planted_system() -> None:
    out = CoupledConfinementDiscoverer().discover(
        {
            "wilson": planted_wilson_area_table(sigma=0.2, kappa=0.04),
            "dressing": planted_decoupling_table(),
            "finite_t": planted_hybrid_wilson_table(),
        }
    )
    assert out.wilson_passed is True
    assert out.dressing_passed is True
    assert out.piecewise_passed is True
    assert out.sigma == pytest.approx(0.2, rel=5e-2)
    assert out.passed is True
    assert out.yang_mills_claim is False
    assert out.continuum_claim is False


def test_coupled_melting_scan_without_area() -> None:
    out = CoupledConfinementDiscoverer().discover(
        {
            "wilson": planted_wilson_area_table(sigma=0.2),
            "finite_t": finite_t_scan_table(sigma0=0.2),
        }
    )
    assert out.piecewise_passed is True
    assert out.passed is True
