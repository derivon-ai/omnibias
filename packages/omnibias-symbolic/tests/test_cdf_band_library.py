# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""CDF/band feature library for symbolic regression (logistic-law recovery)."""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.symbolic import (
    SplitData,
    build_cdf_band_library,
    build_taylor_library,
    discover_interpretable_surrogate,
    fit_cdf_band_library_plan,
    fit_sparse_equation,
    rmse,
)


def _logistic(u: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-u))


def _logistic_split(*, n: int = 1200, seed: int = 0) -> SplitData:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-6.0, 6.0, size=(n, 1))
    y = _logistic(x[:, 0]) + rng.normal(0.0, 0.01, size=n)
    order = rng.permutation(n)
    n_tr, n_va = int(0.6 * n), int(0.2 * n)
    tr, va, te = order[:n_tr], order[n_tr : n_tr + n_va], order[n_tr + n_va :]
    return SplitData(
        x_train=x[tr], y_train=y[tr], x_val=x[va], y_val=y[va], x_test=x[te], y_test=y[te]
    )


# ----- exact recovery -------------------------------------------------------


def test_cdf_feature_recovers_exact_logistic_column() -> None:
    x = np.linspace(-5.0, 5.0, 400).reshape(-1, 1)
    y = _logistic((x[:, 0] - 0.5) / 0.5)  # == sigmoid((x-0.5)/0.5)
    design, names = build_cdf_band_library(
        x, bases=("sigmoid",), locations=np.array([0.5]), scales=np.array([0.5])
    )
    assert design.shape == (400, 1)
    eq = fit_sparse_equation(design, y, names, alpha=1e-12, threshold=1e-8)
    pred = eq.predict(design)
    assert rmse(y, pred) < 1e-8
    active = eq.active_terms()
    assert len(active) == 1
    assert "sigmoid" in str(active[0]["name"])
    assert float(active[0]["coefficient"]) == pytest.approx(1.0, abs=1e-6)


# ----- AutoML integration ---------------------------------------------------


def test_automl_selects_cdf_band_on_logistic_law() -> None:
    data = _logistic_split(seed=1)
    result = discover_interpretable_surrogate(data, include_cdf_band=True)
    assert result["family"] == "cdf_band"
    metrics = result["metrics"]
    assert isinstance(metrics, dict)
    assert float(metrics["rmse"]) < 0.03


def test_cdf_band_beats_polynomial_on_logistic() -> None:
    data = _logistic_split(seed=2)
    plan = fit_cdf_band_library_plan(data.x_train, bases=("sigmoid", "tanh"))
    d_tr, names = plan.builder(data.x_train)
    d_te, _ = plan.builder(data.x_test)
    eq = fit_sparse_equation(d_tr, data.y_train, names, alpha=1e-8, threshold=1e-4)
    cdf_rmse = rmse(data.y_test, eq.predict(d_te))

    t_tr, t_names = build_taylor_library(data.x_train, max_degree=3)
    t_te, _ = build_taylor_library(data.x_test, max_degree=3)
    teq = fit_sparse_equation(t_tr, data.y_train, t_names, alpha=1e-8, threshold=1e-4)
    poly_rmse = rmse(data.y_test, teq.predict(t_te))

    assert cdf_rmse < poly_rmse
    assert cdf_rmse < 0.03


def test_explicit_specs_disable_auto_cdf_band() -> None:
    # Passing specs should not silently inject the CDF family.
    from omnibias.symbolic import default_surrogate_specs

    data = _logistic_split(seed=3)
    result = discover_interpretable_surrogate(data, specs=default_surrogate_specs())
    assert result["family"] in {"taylor", "fourier", "taylor_fourier"}


# ----- plan is leakage-free (consistent grid across splits) -----------------


def test_plan_builder_uses_fixed_grid_across_splits() -> None:
    data = _logistic_split(seed=4)
    plan = fit_cdf_band_library_plan(data.x_train, bases=("sigmoid", "tanh"), n_locations=4)
    _, names_train = plan.builder(data.x_train)
    _, names_test = plan.builder(data.x_test)
    # Same fitted locations/scales -> identical column definitions on any split.
    assert names_train == names_test
    assert len(names_train) == 2 * 3 * 4  # bases x scale_mults x locations (1 feature)


# ----- guards ---------------------------------------------------------------


def test_build_cdf_band_rejects_nonpositive_scale() -> None:
    x = np.linspace(-1.0, 1.0, 10).reshape(-1, 1)
    with pytest.raises(ValueError, match="scales must be positive"):
        build_cdf_band_library(x, locations=np.array([0.0]), scales=np.array([0.0]))


def test_build_cdf_band_rejects_unknown_base() -> None:
    x = np.linspace(-1.0, 1.0, 10).reshape(-1, 1)
    with pytest.raises(ValueError, match="unknown CDF base"):
        build_cdf_band_library(
            x, bases=("not_a_cdf",), locations=np.array([0.0]), scales=np.array([1.0])
        )


def test_build_cdf_band_rejects_1d_input() -> None:
    with pytest.raises(ValueError, match="must be 2D"):
        build_cdf_band_library(
            np.linspace(-1.0, 1.0, 10), locations=np.array([0.0]), scales=np.array([1.0])
        )


def test_fit_plan_rejects_bad_scale_mults() -> None:
    x = np.linspace(-1.0, 1.0, 10).reshape(-1, 1)
    with pytest.raises(ValueError, match="scale_mults"):
        fit_cdf_band_library_plan(x, scale_mults=(0.0, 1.0))
