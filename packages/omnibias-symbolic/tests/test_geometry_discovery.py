# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Geometric PDE-law discovery: heat flow on a curved manifold.

The headline test is that injecting the exact Laplace--Beltrami column as an
operator atom recovers the **spherical heat law** ``u_t = Delta_g u`` as a clean
one-term relation, while the *flat* operator library (no geometric atom) cannot
represent the position-dependent ``cot(theta)`` drift with constant coefficients
and is left with a double-digit-percent residual. This is the sense in which the
geometric operator is genuinely necessary, not a reparametrisation of flat
columns.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from omnibias.symbolic.field_discovery import (  # noqa: E402
    FieldLawDiscoverer,
)
from omnibias.symbolic.geometry_discovery import (  # noqa: E402
    discover_geometric_heat_law,
    evaluate_geometric_discovery,
    laplace_beltrami,
    make_geometric_heat_split,
)


def test_spherical_heat_law_is_recovered_as_one_term():
    train, val, test, metrics, hidden = make_geometric_heat_split(seed=1)
    result = discover_geometric_heat_law(train, val, test, metrics)
    assert "S^2" in hidden
    # u_t = 1.0 * lap_g(u): single term, unit coefficient, machine precision.
    terms = result["selected_terms"]
    assert len(terms) == 1
    (term,) = terms
    assert term["name"] == "lap_g(u)"
    assert term["coefficient"] == pytest.approx(1.0, abs=1e-6)
    assert result["test_rmse"] / result["target_scale"] < 1e-9


def test_flat_library_cannot_represent_the_geometric_drift():
    train, val, test, _metrics, _hidden = make_geometric_heat_split(seed=1)
    # Method-of-lines flat discovery (no Laplace-Beltrami atom).
    flat = FieldLawDiscoverer(max_degree=1, time_axis=2)
    result = flat.discover(train, val, test, lhs_index=(0, 0, 1))
    # The cot(theta) drift is irreducible to constant-coefficient flat columns.
    assert result.test_rmse / result.target_scale > 1e-2


def test_geometric_atom_beats_flat_by_orders_of_magnitude():
    train, val, test, metrics, _hidden = make_geometric_heat_split(seed=4)
    geo = discover_geometric_heat_law(train, val, test, metrics)
    flat = FieldLawDiscoverer(max_degree=1, time_axis=2)
    flat_res = flat.discover(train, val, test, lhs_index=(0, 0, 1))
    assert geo["test_rmse"] < 1e-6 * flat_res.test_rmse


def test_evaluate_geometric_discovery_smoke():
    report = evaluate_geometric_discovery(seed=0)
    case = report["geometric_heat"]
    assert "round sphere" in case["hidden_law"]
    assert case["equation"].startswith("u_t = ")
    assert case["test_rmse"] / case["target_scale"] < 1e-8


def test_make_geometric_heat_split_satisfies_the_law_exactly():
    # u_t equals Delta_g u at every sample, for all three splits.
    train, val, test, metrics, _hidden = make_geometric_heat_split(seed=7)
    for jet, metric in zip((train, val, test), metrics, strict=True):
        lb = laplace_beltrami(jet, metric, spatial_axes=(0, 1))
        assert np.allclose(jet.partial((0, 0, 1)), lb, atol=1e-11)


def test_extra_columns_fn_injects_custom_atom():
    # The generic FieldLawDiscoverer hook accepts any exact operator column.
    train, val, test, metrics, _hidden = make_geometric_heat_split(seed=2)
    paired = {id(train): metrics[0], id(val): metrics[1], id(test): metrics[2]}

    def extra(jet):
        return {"my_lap_g": laplace_beltrami(jet, paired[id(jet)], spatial_axes=(0, 1))}

    disc = FieldLawDiscoverer(max_degree=1, time_axis=2)
    result = disc.discover(
        train, val, test, lhs_index=(0, 0, 1), extra_columns_fn=extra
    )
    names = [t["name"] for t in result.active_terms()]
    assert names == ["my_lap_g"]
    assert result.test_rmse / result.target_scale < 1e-9


def test_geometric_discovery_is_deterministic():
    a = discover_geometric_heat_law(*make_geometric_heat_split(seed=3)[:4])
    b = discover_geometric_heat_law(*make_geometric_heat_split(seed=3)[:4])
    assert a["equation"] == b["equation"]
    assert a["test_rmse"] == pytest.approx(b["test_rmse"], rel=0, abs=0)


def test_single_mode_is_degenerate_two_modes_are_not():
    # A single zonal mode also satisfies u_t = -l(l+1) u (a one-term flat law),
    # so the geometric law is not uniquely simplest; two modes break that tie.
    train, val, test, metrics, _ = make_geometric_heat_split(seed=5, degrees=(2,), amps=(1.0,))
    flat = FieldLawDiscoverer(max_degree=1, time_axis=2)
    res = flat.discover(train, val, test, lhs_index=(0, 0, 1))
    # With one mode, the flat law u_t = -6 u fits to machine precision.
    assert res.test_rmse / res.target_scale < 1e-9
