# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Path B statistical discoverer: planted laws, mix-in spies, lazy export."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

geometry = pytest.importorskip("omnibias.geometry")
from omnibias.geometry.gauge._core.data_paths import LatticeLinkField
from omnibias.geometry.gauge._core.ensemble_language import LEGAL_ENSEMBLE_ATOMS
from omnibias.geometry.gauge._core.loop_language import identity_numpy_links
from omnibias.symbolic.ensemble_discovery import (
    PLANTED_AREA_SIGMA,
    PLANTED_BETA_EXPONENT,
    PLANTED_POLYAKOV_MASS,
    StatisticalLawDiscoverer,
    discover_planted_area_perimeter,
    discover_planted_order_parameter_scaling,
    discover_planted_polyakov_mass,
    discover_planted_spectral_density,
    planted_order_parameter_table,
)
from omnibias.symbolic.gauge_discovery import make_yang_mills_bpst_split
from omnibias.symbolic.loop_discovery import planted_area_law_table


def test_order_parameter_scaling_recovers_exponent() -> None:
    out = discover_planted_order_parameter_scaling()
    assert out["passed"] is True
    assert out["yang_mills_claim"] is False
    assert out["continuum_claim"] is False
    assert float(out["exponent_recovered"]) == pytest.approx(
        PLANTED_BETA_EXPONENT, rel=5e-3
    )
    names = out["diagnostics"]["dictionary_names"]
    assert set(names) == set(LEGAL_ENSEMBLE_ATOMS)
    assert "gevp" not in names


def test_polyakov_mass_recovers_planted_slope() -> None:
    out = discover_planted_polyakov_mass()
    assert out["passed"] is True
    assert float(out["mass_recovered"]) == pytest.approx(PLANTED_POLYAKOV_MASS, rel=5e-3)
    assert out["yang_mills_claim"] is False
    assert out["continuum_claim"] is False


def test_area_perimeter_creutz_jump() -> None:
    out = discover_planted_area_perimeter()
    assert out["passed"] is True
    assert float(out["creutz_below"]) == pytest.approx(PLANTED_AREA_SIGMA, abs=1e-12)
    assert float(out["creutz_above"]) == pytest.approx(0.0, abs=1e-12)
    assert out["yang_mills_claim"] is False
    assert out["continuum_claim"] is False


def test_spectral_density_recovers_planted_rho() -> None:
    out = discover_planted_spectral_density()
    assert out["passed"] is True
    assert out["ill_posed"] is True
    assert out["yang_mills_claim"] is False
    assert out["continuum_claim"] is False


def test_discoverer_rejects_jet_before_stlsq() -> None:
    train, val, test, _conns, _pts = make_yang_mills_bpst_split(seed=1, counts=(8, 4, 4))
    called = {"stlsq": False}

    def _boom(*_args, **_kwargs):
        called["stlsq"] = True
        raise AssertionError("fit_sparse_equation must not run")

    disc = StatisticalLawDiscoverer()
    with patch("omnibias.symbolic.ensemble_discovery.fit_sparse_equation", _boom):
        with pytest.raises(ValueError, match="ensemble"):
            disc.discover(train, val, test, lhs_name="log_abs_P")
    assert called["stlsq"] is False


def test_discoverer_rejects_loop_table_and_single_config() -> None:
    loop = planted_area_law_table(n_rows=12)
    field = LatticeLinkField(links=identity_numpy_links((2, 2, 2, 2)))
    disc = StatisticalLawDiscoverer()
    called = {"stlsq": False}

    def _boom(*_args, **_kwargs):
        called["stlsq"] = True
        raise AssertionError("fit_sparse_equation must not run")

    with patch("omnibias.symbolic.ensemble_discovery.fit_sparse_equation", _boom):
        with pytest.raises(ValueError, match="LoopObservableTable|language"):
            disc.discover(loop, loop, loop, lhs_name="log_abs_P")
        with pytest.raises(ValueError, match="single"):
            disc.discover(field, field, field, lhs_name="log_abs_P")
    assert called["stlsq"] is False


@pytest.mark.parametrize("illegal", ["tr(F^2)", "W(1,1)", "inverse_laplacian", "gevp"])
def test_ensemble_extra_raises_before_stlsq(illegal: str) -> None:
    table = planted_order_parameter_table(n_rows=12)
    n = 12
    counts = (6, 3, 3)
    from omnibias.symbolic.ensemble_discovery import _split_table

    train, val, test = _split_table(table, counts)

    def extra(_tab):
        return {illegal: np.ones(n // 2)}

    called = {"stlsq": False}

    def _boom(*_args, **_kwargs):
        called["stlsq"] = True
        raise AssertionError("fit_sparse_equation must not run")

    disc = StatisticalLawDiscoverer()
    with patch("omnibias.symbolic.ensemble_discovery.fit_sparse_equation", _boom):
        with pytest.raises(
            ValueError,
            match="ensemble|holonomy|language|inverse Laplacian|certificate|allowlisted",
        ):
            disc.discover(
                train, val, test, lhs_name="log_abs_P", extra_columns_fn=extra
            )
    assert called["stlsq"] is False


def test_lazy_ensemble_export() -> None:
    from omnibias.symbolic import StatisticalLawDiscoverer as exported

    assert exported is StatisticalLawDiscoverer
