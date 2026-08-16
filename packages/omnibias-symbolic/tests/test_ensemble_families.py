# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Named Path B families recover planted IR / Wilson laws."""

from __future__ import annotations

import pytest
from omnibias.symbolic.ensemble_families import (
    NamedFamilyDiscoverer,
    planted_decoupling_table,
    planted_gribov_stingl_table,
    planted_wilson_area_table,
)


def test_decoupling_recovers_planted() -> None:
    out = NamedFamilyDiscoverer().fit(planted_decoupling_table(), family="decoupling")
    assert out.passed is True
    assert out.parameters["Z"] == pytest.approx(1.0, rel=5e-2)
    assert out.parameters["M2"] == pytest.approx(0.5, rel=5e-2)
    assert out.skill > 0.0
    assert out.yang_mills_claim is False


def test_gribov_stingl_recovers_planted() -> None:
    out = NamedFamilyDiscoverer().fit(
        planted_gribov_stingl_table(), family="gribov_stingl"
    )
    assert out.passed is True
    assert out.parameters["Z"] == pytest.approx(1.0, rel=1e-1)
    assert out.skill > 0.0


def test_area_perimeter_recovers_sigma() -> None:
    out = NamedFamilyDiscoverer().fit(
        planted_wilson_area_table(sigma=0.2, kappa=0.05), family="area_perimeter"
    )
    assert out.passed is True
    assert out.parameters["sigma"] == pytest.approx(0.2, rel=5e-2)
    assert out.parameters["kappa"] == pytest.approx(0.05, rel=5e-2)
    assert out.skill > 0.0


def test_luscher_improves_or_matches_when_gamma_present() -> None:
    table = planted_wilson_area_table(sigma=0.2, kappa=0.05, gamma=0.15)
    base = NamedFamilyDiscoverer().fit(table, family="area_perimeter")
    luscher = NamedFamilyDiscoverer().fit(table, family="luscher")
    assert luscher.passed is True
    assert luscher.parameters["gamma"] == pytest.approx(0.15, rel=1e-1)
    assert luscher.model_rmse <= base.model_rmse + 1e-9
