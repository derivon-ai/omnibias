# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Loop-law discoverer: plaquette identity, planted area law, language-split spies."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

geometry = pytest.importorskip("omnibias.geometry")
from omnibias.geometry.gauge._core.loop_language import (  # noqa: E402
    LEGAL_LOOP_ATOMS,
    LOOP_PLAQUETTE,
    LOOP_W11,
)
from omnibias.symbolic.gauge_discovery import (  # noqa: E402
    GaugeLawDiscoverer,
    make_yang_mills_bpst_split,
)
from omnibias.symbolic.loop_discovery import (  # noqa: E402
    CERT_COLUMN_NAMES,
    PLANTED_AREA_SIGMA,
    PLAQUETTE_RMSE_FLOOR,
    LoopLawDiscoverer,
    discover_planted_area_law,
    discover_wilson_plaquette_law,
    planted_area_law_table,
)

PLAQUETTE_COEF_RTOL = 1e-6


def test_wilson_plaquette_law_recovers_unit_coefficient() -> None:
    out = discover_wilson_plaquette_law(seed=0, n_configs=2)
    assert out["yang_mills_claim"] is False
    assert out["continuum_claim"] is False
    assert out["diagnostics"]["yang_mills_claim"] is False
    assert float(out["test_rmse"]) < PLAQUETTE_RMSE_FLOOR
    terms = out["selected_terms"]
    assert len(terms) == 1
    assert terms[0]["name"] == LOOP_PLAQUETTE
    assert float(terms[0]["coefficient"]) == pytest.approx(1.0, rel=PLAQUETTE_COEF_RTOL)
    names = out["diagnostics"]["dictionary_names"]
    assert set(names) == set(LEGAL_LOOP_ATOMS)
    assert not set(names) & CERT_COLUMN_NAMES


def test_planted_area_law_creutz_recovers_sigma() -> None:
    out = discover_planted_area_law(sigma=PLANTED_AREA_SIGMA)
    assert out["passed"] is True
    assert float(out["creutz"]) == pytest.approx(PLANTED_AREA_SIGMA, rel=1e-12)
    assert out["yang_mills_claim"] is False
    assert out["continuum_claim"] is False
    assert "gevp" not in out["dictionary_names"]
    assert "transfer_gap" not in out["dictionary_names"]


def test_loop_discoverer_rejects_jet_before_stlsq() -> None:
    train, val, test, _conns, _pts = make_yang_mills_bpst_split(seed=1, counts=(8, 4, 4))
    called = {"stlsq": False}

    def _boom(*_args, **_kwargs):
        called["stlsq"] = True
        raise AssertionError("fit_sparse_equation must not run")

    disc = LoopLawDiscoverer()
    with patch("omnibias.symbolic.loop_discovery.fit_sparse_equation", _boom):
        with pytest.raises(ValueError, match="holonomy"):
            disc.discover(train, val, test)
    assert called["stlsq"] is False


@pytest.mark.parametrize("illegal", ["tr(F^2)", "inverse_laplacian"])
def test_loop_extra_raises_before_stlsq(illegal: str) -> None:
    table = planted_area_law_table(n_rows=12)
    table = type(table)(
        values={**table.values, LOOP_PLAQUETTE: np.ones(12)},
        source="lattice_links",
    )
    counts = (6, 3, 3)
    from omnibias.symbolic.loop_discovery import _split_table

    train, val, test = _split_table(table, counts)

    def extra(_tab):
        return {illegal: np.ones(train.values[LOOP_W11].shape[0])}

    called = {"stlsq": False}

    def _boom(*_args, **_kwargs):
        called["stlsq"] = True
        raise AssertionError("fit_sparse_equation must not run")

    disc = LoopLawDiscoverer()
    with patch("omnibias.symbolic.loop_discovery.fit_sparse_equation", _boom):
        with pytest.raises(ValueError, match="language split|inverse Laplacian|holonomy"):
            disc.discover(
                train, val, test, lhs_name=LOOP_W11, extra_columns_fn=extra
            )
    assert called["stlsq"] is False


def test_lazy_loop_export() -> None:
    from omnibias.symbolic import LoopLawDiscoverer as exported

    assert exported is LoopLawDiscoverer
