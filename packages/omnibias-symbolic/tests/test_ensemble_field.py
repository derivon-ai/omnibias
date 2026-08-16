# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Interpolate-then-jet on a planted static potential."""

from __future__ import annotations

import pytest
from omnibias.geometry.gauge._core.covariant_jet import GaugeCovariantJet
from omnibias.geometry.gauge._core.instanton import bpst_instanton_arrays
from omnibias.geometry.gauge._core.lie_algebra import su
from omnibias.symbolic.ensemble_field import ensemble_field_law, planted_static_potential_table


def test_interpolant_jet_recovers_potential_slope() -> None:
    table, d_v = planted_static_potential_table(sigma=0.2, gamma=0.15)
    out = ensemble_field_law(table, analytic_partial=d_v, hidden=96)
    assert out.passed is True
    assert out.interpolant_rmse < 0.15
    assert out.derivative_rmse < 0.25
    assert out.yang_mills_claim is False
    assert out.notes is not None
    assert out.notes["kind"] == "interpolant_jet"


def test_refuses_gauge_jet_as_source() -> None:
    table, _ = planted_static_potential_table()
    pts = __import__("numpy").random.default_rng(0).uniform(-1.0, 1.0, size=(6, 4))
    a, da, dda = bpst_instanton_arrays(pts)
    jet = GaugeCovariantJet.from_arrays(a, da, dda, algebra=su(2), coupling=1.0)
    with pytest.raises(ValueError, match="ensemble"):
        ensemble_field_law(table, source=jet)
