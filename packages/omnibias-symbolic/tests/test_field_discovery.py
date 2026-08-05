# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Multivariate / space-time PDE law discovery from exact closed-form field jets.

:class:`FieldLawDiscoverer` recovers the canonical PDE laws -- Laplace, heat, wave,
viscous Burgers (nonlinear) and 2-D heat -- to machine precision from analytic
field jets, using the physically-motivated ``time_axis`` (method-of-lines) and
``rhs_orders`` (elliptic principal-part) library restrictions. An end-to-end run
fits a :class:`NeuralFieldND` to advection samples and recovers ``u_t = -c u_x``
from its closed-form derivatives.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from omnibias.symbolic.field_discovery import (  # noqa: E402
    FieldLawDiscoverer,
    build_field_relation_library,
    discover_field_pde_law,
    evaluate_field_pde_discovery,
    extract_field_jet,
    fit_neural_field_nd,
    make_burgers_field_split,
    make_heat2d_field_split,
    make_heat_field_split,
    make_laplace_field_split,
    make_wave_field_split,
)


def _terms(result: dict[str, object]) -> dict[str, float]:
    return {str(r["name"]): float(r["coefficient"]) for r in result["selected_terms"]}  # type: ignore[index]


# ----- canonical PDE recovery (analytic, machine precision) -----------------


def test_recovers_laplace_equation() -> None:
    tr, va, te, _ = make_laplace_field_split(seed=0)
    res = discover_field_pde_law(tr, va, te, lhs_index=(2, 0), max_degree=1, rhs_orders=(2,))
    terms = _terms(res)
    assert set(terms) == {"u_yy"}
    assert terms["u_yy"] == pytest.approx(-1.0, abs=1e-6)
    assert res["test_rmse"] < 1e-8


def test_recovers_heat_equation() -> None:
    tr, va, te, _ = make_heat_field_split(diffusivity=0.12, seed=0)
    res = discover_field_pde_law(tr, va, te, lhs_index=(0, 1), max_degree=1, time_axis=1)
    terms = _terms(res)
    assert set(terms) == {"u_xx"}
    assert terms["u_xx"] == pytest.approx(0.12, abs=1e-6)
    assert res["test_rmse"] < 1e-8


def test_recovers_wave_equation() -> None:
    tr, va, te, _ = make_wave_field_split(speed=1.3, seed=0)
    res = discover_field_pde_law(tr, va, te, lhs_index=(0, 2), max_degree=1, time_axis=1)
    terms = _terms(res)
    assert set(terms) == {"u_xx"}
    assert terms["u_xx"] == pytest.approx(1.3**2, abs=1e-5)
    assert res["test_rmse"] < 1e-7


def test_recovers_viscous_burgers_equation() -> None:
    tr, va, te, _ = make_burgers_field_split(viscosity=0.1, seed=0)
    res = discover_field_pde_law(tr, va, te, lhs_index=(0, 1), max_degree=2, time_axis=1)
    terms = _terms(res)
    assert set(terms) == {"u*u_x", "u_xx"}
    assert terms["u*u_x"] == pytest.approx(-1.0, abs=1e-4)
    assert terms["u_xx"] == pytest.approx(0.1, abs=1e-4)
    assert res["test_rmse"] < 1e-8


def test_recovers_2d_heat_equation() -> None:
    tr, va, te, _ = make_heat2d_field_split(diffusivity=0.1, seed=0)
    res = discover_field_pde_law(tr, va, te, lhs_index=(0, 0, 1), max_degree=1, time_axis=2)
    terms = _terms(res)
    assert set(terms) == {"u_xx", "u_yy"}
    assert terms["u_xx"] == pytest.approx(0.1, abs=1e-5)
    assert terms["u_yy"] == pytest.approx(0.1, abs=1e-5)
    assert res["test_rmse"] < 1e-8


def test_evaluate_field_pde_discovery_smoke() -> None:
    report = evaluate_field_pde_discovery(seed=1)
    assert set(report) == {"laplace", "heat", "wave", "burgers", "heat2d"}
    for case in report.values():
        assert float(case["test_rmse"]) < 1e-6  # type: ignore[index]


# ----- library restrictions -------------------------------------------------


def test_time_axis_restriction_drops_equal_and_higher_time_derivatives() -> None:
    tr, _, _, _ = make_heat_field_split(seed=0)
    _, names = build_field_relation_library(tr, lhs_index=(0, 1), max_degree=1, time_axis=1)
    # No RHS atom may carry a t-derivative of order >= the LHS (u_t, u_tt, u_xt out).
    assert "u_t" not in names
    assert "u_tt" not in names
    assert "u_xt" not in names
    assert "u_xx" in names
    assert "u_x" in names
    assert "u" in names


def test_rhs_orders_restriction_keeps_only_requested_total_orders() -> None:
    tr, _, _, _ = make_laplace_field_split(seed=0)
    _, names = build_field_relation_library(tr, lhs_index=(2, 0), max_degree=1, rhs_orders=(2,))
    assert set(names) == {"u_yy", "u_xy"}


def test_exclude_removes_named_atoms() -> None:
    tr, _, _, _ = make_heat_field_split(seed=0)
    _, names = build_field_relation_library(
        tr, lhs_index=(0, 1), max_degree=1, time_axis=1, exclude=("u",)
    )
    assert "u" not in names
    assert "u_xx" in names


# ----- end-to-end neural field ----------------------------------------------


def test_neural_field_recovers_advection_law() -> None:
    c = 0.7

    def sample(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
        r = np.random.default_rng(seed)
        x = r.uniform(-3.0, 3.0, n)
        t = r.uniform(0.0, 2.0, n)
        return np.stack([x, t], axis=1), np.sin(x - c * t)

    Xtr, ytr = sample(1500, 1)
    Xva, _ = sample(800, 2)
    Xte, _ = sample(800, 3)
    fld = fit_neural_field_nd(
        Xtr, ytr, hidden=400, ridge=1e-7, activation="tanh", bandwidth=2.0,
        var_names=("x", "t"), seed=7,
    )
    assert fld.train_rmse < 1e-3
    jets = [extract_field_jet(fld, X, max_order=1) for X in (Xtr, Xva, Xte)]
    result = FieldLawDiscoverer(max_degree=1, time_axis=1, complexity_weight=5e-3).discover(
        *jets, lhs_index=(0, 1)
    )
    terms = {str(r["name"]): float(r["coefficient"]) for r in result.active_terms()}
    assert "u_x" in terms
    assert terms["u_x"] == pytest.approx(-c, abs=2e-2)
    assert result.test_rmse / result.target_scale < 5e-2
    # residual diagnostics are populated on the selected law
    assert "differential_entropy" in result.diagnostics
    assert "max_feature_residual_mi" in result.diagnostics


# ----- guards ---------------------------------------------------------------


def test_build_field_relation_library_rejects_bad_degree() -> None:
    tr, _, _, _ = make_heat_field_split(seed=0)
    with pytest.raises(ValueError):
        build_field_relation_library(tr, lhs_index=(0, 1), max_degree=0)


def test_discover_rejects_empty_library() -> None:
    # Excluding every admissible atom leaves no candidate columns.
    tr, va, te, _ = make_laplace_field_split(seed=0)
    with pytest.raises(ValueError):
        discover_field_pde_law(
            tr, va, te, lhs_index=(2, 0), max_degree=1, rhs_orders=(2,),
            exclude=("u_yy", "u_xy"),
        )
