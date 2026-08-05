# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Self-information (surprisal) feature library for symbolic regression.

The information-theoretic twin of the CDF/band probability features: each column
is the surprisal ``-ln f((x - loc)/scale)`` of a base density ``f``, so the
search can recover log-likelihood / energy-style laws (a V-shaped logistic
self-information, an ``arctan`` log-quadratic) that monotone CDF or polynomial
bases cannot express.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from omnibias.symbolic import (
    SplitData,
    build_information_library,
    build_taylor_library,
    discover_interpretable_surrogate,
    fit_information_library_plan,
    fit_sparse_equation,
    rmse,
)


def _surprisal_sigmoid(u: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, u) + np.logaddexp(0.0, -u)


# ----- exact recovery -------------------------------------------------------


def test_information_feature_recovers_exact_sigmoid_surprisal_column() -> None:
    x = np.linspace(-5.0, 5.0, 400).reshape(-1, 1)
    y = _surprisal_sigmoid((x[:, 0] - 0.5) / 1.5)
    design, names = build_information_library(
        x, bases=("sigmoid",), locations=np.array([0.5]), scales=np.array([1.5])
    )
    assert design.shape == (400, 1)
    eq = fit_sparse_equation(design, y, names, alpha=1e-12, threshold=1e-8)
    pred = eq.predict(design)
    assert rmse(y, pred) < 1e-8
    active = eq.active_terms()
    assert len(active) == 1
    assert "surprisal_sigmoid" in str(active[0]["name"])
    assert float(active[0]["coefficient"]) == pytest.approx(1.0, abs=1e-6)


def test_information_feature_recovers_arctan_log_quadratic() -> None:
    x = np.linspace(-4.0, 4.0, 300).reshape(-1, 1)
    u = (x[:, 0] - 0.0) / 2.0
    y = math.log(math.pi) + np.log1p(u * u)  # == surprisal_arctan(u)
    design, names = build_information_library(
        x, bases=("arctan",), locations=np.array([0.0]), scales=np.array([2.0])
    )
    eq = fit_sparse_equation(design, y, names, alpha=1e-12, threshold=1e-8)
    assert rmse(y, eq.predict(design)) < 1e-8
    assert "surprisal_arctan" in str(eq.active_terms()[0]["name"])


def test_surprisal_columns_match_closed_form() -> None:
    # Spot-check the log-domain formulas against direct -ln(density).
    u = np.linspace(-3.0, 3.0, 11)
    x = u.reshape(-1, 1)
    design, names = build_information_library(
        x, bases=("sigmoid", "tanh", "arctan"), locations=np.array([0.0]), scales=np.array([1.0])
    )
    cols = {name.split("(")[0]: design[:, i] for i, name in enumerate(names)}
    s = 1.0 / (1.0 + np.exp(-u))
    assert np.allclose(cols["surprisal_sigmoid"], -np.log(s * (1.0 - s)), atol=1e-12)
    t = 0.5 * (1.0 - np.tanh(u) ** 2)
    assert np.allclose(cols["surprisal_tanh"], -np.log(t), atol=1e-12)
    dens = 1.0 / (math.pi * (1.0 + u * u))
    assert np.allclose(cols["surprisal_arctan"], -np.log(dens), atol=1e-12)


# ----- AutoML integration ---------------------------------------------------


def _surprisal_split(*, n: int = 1500, seed: int = 0) -> tuple[SplitData, str]:
    """Split whose target is exactly one auto-fittable sigmoid-surprisal column."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-6.0, 6.0, size=(n, 1))
    order = rng.permutation(n)
    n_tr, n_va = int(0.6 * n), int(0.2 * n)
    tr, va, te = order[:n_tr], order[n_tr : n_tr + n_va], order[n_tr + n_va :]
    x_tr, x_va, x_te = x[tr], x[va], x[te]
    # Build the target from the *same* train-fitted plan the discoverer will refit,
    # so the generating column is guaranteed present in the candidate library.
    plan = fit_information_library_plan(x_tr, bases=("sigmoid",))
    col = 2  # scale = 0.5*std (sharp V), loc = median quantile
    name = plan.builder(x_tr)[1][col]

    def target(xx: np.ndarray) -> np.ndarray:
        return plan.builder(xx)[0][:, col]

    y_tr = target(x_tr) + rng.normal(0.0, 0.005, size=x_tr.shape[0])
    y_va = target(x_va) + rng.normal(0.0, 0.005, size=x_va.shape[0])
    y_te = target(x_te) + rng.normal(0.0, 0.005, size=x_te.shape[0])
    return (
        SplitData(x_train=x_tr, y_train=y_tr, x_val=x_va, y_val=y_va, x_test=x_te, y_test=y_te),
        name,
    )


def test_automl_selects_information_on_a_surprisal_law() -> None:
    data, _ = _surprisal_split(seed=1)
    result = discover_interpretable_surrogate(
        data, include_information=True, information_bases=("sigmoid",)
    )
    assert result["family"] == "information"
    metrics = result["metrics"]
    assert isinstance(metrics, dict)
    assert float(metrics["rmse"]) < 0.03


def test_information_beats_polynomial_on_surprisal_law() -> None:
    data, _ = _surprisal_split(seed=2)
    plan = fit_information_library_plan(data.x_train, bases=("sigmoid",))
    d_tr, names = plan.builder(data.x_train)
    d_te, _ = plan.builder(data.x_test)
    eq = fit_sparse_equation(d_tr, data.y_train, names, alpha=1e-8, threshold=1e-4)
    info_rmse = rmse(data.y_test, eq.predict(d_te))

    t_tr, t_names = build_taylor_library(data.x_train, max_degree=3)
    t_te, _ = build_taylor_library(data.x_test, max_degree=3)
    teq = fit_sparse_equation(t_tr, data.y_train, t_names, alpha=1e-8, threshold=1e-4)
    poly_rmse = rmse(data.y_test, teq.predict(t_te))

    assert info_rmse < poly_rmse
    assert info_rmse < 0.03


def test_default_discover_does_not_inject_information() -> None:
    # include_information defaults to False, so existing behaviour is preserved.
    data, _ = _surprisal_split(seed=3)
    result = discover_interpretable_surrogate(data)  # include_information omitted
    assert result["family"] != "information"


def test_explicit_specs_disable_auto_information() -> None:
    from omnibias.symbolic import default_surrogate_specs

    data, _ = _surprisal_split(seed=4)
    result = discover_interpretable_surrogate(
        data, specs=default_surrogate_specs(), include_information=True
    )
    assert result["family"] in {"taylor", "fourier", "taylor_fourier"}


# ----- plan is leakage-free (consistent grid across splits) -----------------


def test_plan_builder_uses_fixed_grid_across_splits() -> None:
    data, _ = _surprisal_split(seed=5)
    plan = fit_information_library_plan(
        data.x_train, bases=("sigmoid", "arctan"), n_locations=4
    )
    _, names_train = plan.builder(data.x_train)
    _, names_test = plan.builder(data.x_test)
    assert names_train == names_test
    assert len(names_train) == 2 * 3 * 4  # bases x scale_mults x locations (1 feature)


# ----- guards ---------------------------------------------------------------


def test_build_information_rejects_nonpositive_scale() -> None:
    x = np.linspace(-1.0, 1.0, 10).reshape(-1, 1)
    with pytest.raises(ValueError, match="scales must be positive"):
        build_information_library(x, locations=np.array([0.0]), scales=np.array([0.0]))


def test_build_information_rejects_unknown_base() -> None:
    x = np.linspace(-1.0, 1.0, 10).reshape(-1, 1)
    with pytest.raises(ValueError, match="unknown information base"):
        build_information_library(
            x, bases=("not_a_density",), locations=np.array([0.0]), scales=np.array([1.0])
        )


def test_build_information_rejects_1d_input() -> None:
    with pytest.raises(ValueError, match="must be 2D"):
        build_information_library(
            np.linspace(-1.0, 1.0, 10), locations=np.array([0.0]), scales=np.array([1.0])
        )


def test_fit_plan_rejects_bad_scale_mults() -> None:
    x = np.linspace(-1.0, 1.0, 10).reshape(-1, 1)
    with pytest.raises(ValueError, match="scale_mults"):
        fit_information_library_plan(x, scale_mults=(0.0, 1.0))
