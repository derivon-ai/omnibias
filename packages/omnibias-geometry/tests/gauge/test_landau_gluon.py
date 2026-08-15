# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Landau-gauge gluon 2-point: orbit return, transversality, jet refuse."""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.geometry.gauge._core.covariant_jet import GaugeCovariantJet
from omnibias.geometry.gauge._core.data_paths import LatticeLinkField
from omnibias.geometry.gauge._core.landau_gluon import (
    LANDAU_ORBIT_ATOL,
    LANDAU_RESIDUAL_ATOL,
    LANDAU_TRANSVERSE_ATOL,
    gauge_transform_numpy,
    gluon_propagator_p2,
    landau_gauge_fix,
    landau_residual,
    random_site_gauge,
)
from omnibias.geometry.gauge._core.loop_language import identity_numpy_links
from omnibias.geometry.gauge.lattice._core.kernels import (
    algebra_from_links,
    landau_gauge_overrelax,
)


def test_identity_is_landau_and_has_zero_propagator() -> None:
    links = identity_numpy_links((4, 4, 4, 4))
    field = LatticeLinkField(links=links)
    fixed = landau_gauge_fix(field, n_steps=2, omega=1.0)
    assert fixed["yang_mills_claim"] is False
    assert fixed["continuum_claim"] is False
    assert fixed["ill_posed"] is False
    assert float(np.max(np.abs(algebra_from_links(np, fixed["links"])))) < 1e-15
    table, report = gluon_propagator_p2(field, already_fixed=True)
    assert table.source == "landau_gluon"
    assert float(np.max(np.abs(table.values["G_p2"]))) < 1e-15
    assert report["ill_posed"] is False


def test_gauge_transformed_identity_returns_to_orbit() -> None:
    ident = identity_numpy_links((4, 4, 4, 4))
    rng = np.random.default_rng(3)
    transformed = gauge_transform_numpy(ident, random_site_gauge((4, 4, 4, 4), rng))
    assert landau_residual(transformed) > LANDAU_ORBIT_ATOL
    out = landau_gauge_fix(
        LatticeLinkField(links=transformed), n_steps=36, omega=1.0
    )
    algebra = np.asarray(algebra_from_links(np, out["links"]), dtype=np.float64)
    assert float(np.max(np.abs(algebra))) < LANDAU_ORBIT_ATOL
    assert float(out["residual"]) < LANDAU_RESIDUAL_ATOL
    assert float(out["functional"]) == pytest.approx(1.0, abs=1e-6)


def test_landau_transversality_on_random_links() -> None:
    rng = np.random.default_rng(4)
    raw = rng.normal(size=(4, 4, 4, 4, 4, 4))
    links = raw / np.maximum(np.linalg.norm(raw, axis=-1, keepdims=True), 1e-30)
    table, report = gluon_propagator_p2(
        LatticeLinkField(links=links), n_steps=120, omega=1.7
    )
    assert table.values["p2"].shape == table.values["G_p2"].shape
    assert float(report["residual"]) < LANDAU_RESIDUAL_ATOL
    assert float(report["transverse_residual"]) < LANDAU_TRANSVERSE_ATOL
    assert report["yang_mills_claim"] is False
    assert report["continuum_claim"] is False


def test_from_lattice_links_still_refused_after_landau() -> None:
    links = identity_numpy_links((2, 2, 2, 2))
    fixed = landau_gauge_fix(LatticeLinkField(links=links), n_steps=1)
    with pytest.raises(ValueError, match="lattice"):
        GaugeCovariantJet.from_lattice_links(fixed["links"])


def test_jacobi_kernel_fixes_identity() -> None:
    ident = identity_numpy_links((2, 2, 2, 2))
    out = np.asarray(landau_gauge_overrelax(np, ident, n_steps=3, omega=1.7))
    assert float(np.max(np.abs(out - ident))) < 1e-15
