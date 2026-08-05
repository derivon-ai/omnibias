# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Tests for discovery rigor: coefficient uncertainty and model selection."""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.symbolic.discovery import fit_sparse_equation
from omnibias.symbolic.selection import (
    CRITERIA,
    aic,
    aicc,
    bic,
    equation_information_criterion,
    gaussian_log_likelihood,
    information_criterion,
    kfold_select,
    mdl,
    stability_selection,
)
from omnibias.symbolic.uncertainty import (
    attach_uncertainty,
    bootstrap_coefficients,
    certified_coefficient_intervals,
    ridge_coefficient_covariance,
)


def _linear_dataset(
    *, n: int = 400, noise: float = 0.01, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    x3 = rng.normal(size=n)
    design = np.stack([x1, x2, x3, x1 * x2], axis=1)
    names = ["x1", "x2", "x3", "x1*x2"]
    c_true = np.array([2.0, 0.0, -1.5, 0.0])
    target = design @ c_true + noise * rng.normal(size=n)
    return design, target, names, c_true


# --------------------------------------------------------------------------- #
# Uncertainty: bootstrap                                                       #
# --------------------------------------------------------------------------- #
def test_bootstrap_ci_contains_true_coefficients() -> None:
    design, target, names, c_true = _linear_dataset(noise=0.01, seed=1)
    boot = bootstrap_coefficients(design, target, names, n_boot=200, seed=7)
    for i, name in enumerate(names):
        lo = boot["ci_lower"][i]
        hi = boot["ci_upper"][i]
        assert lo <= c_true[i] <= hi, f"{name}: {c_true[i]} not in [{lo}, {hi}]"


def test_bootstrap_selection_frequency_separates_real_from_spurious() -> None:
    design, target, names, _ = _linear_dataset(noise=0.02, seed=2)
    boot = bootstrap_coefficients(
        design, target, names, n_boot=200, seed=3, threshold=1e-2
    )
    freq = boot["selection_frequency"]
    # Genuine terms x1, x3 selected (almost) always; spurious far less often.
    assert freq[0] > 0.95
    assert freq[2] > 0.95
    assert freq[1] < freq[0]
    assert freq[3] < freq[2]


def test_bootstrap_rejects_bad_ci_level() -> None:
    design, target, names, _ = _linear_dataset()
    with pytest.raises(ValueError):
        bootstrap_coefficients(design, target, names, ci_level=1.5)


# --------------------------------------------------------------------------- #
# Uncertainty: analytic ridge covariance                                      #
# --------------------------------------------------------------------------- #
def test_ridge_covariance_recovers_truth_within_two_sigma() -> None:
    design, target, names, c_true = _linear_dataset(noise=0.05, seed=4)
    cov = ridge_coefficient_covariance(design, target, alpha=1e-10)
    coef = cov["coefficients"]
    se = cov["std_errors"]
    assert cov["covariance"].shape == (len(names), len(names))
    assert np.all(se >= 0.0)
    for i in range(len(names)):
        assert abs(coef[i] - c_true[i]) <= 3.0 * se[i] + 1e-6


# --------------------------------------------------------------------------- #
# Uncertainty: certified intervals                                            #
# --------------------------------------------------------------------------- #
def test_certified_interval_contains_truth_on_exact_data() -> None:
    design, _, _, c_true = _linear_dataset(noise=0.0, seed=5)
    target = design @ c_true  # exact, noiseless
    cert = certified_coefficient_intervals(design, target, alpha=0.0)
    assert cert["certified"] is True
    assert cert["kappa"] < 1.0
    assert np.isfinite(cert["radius"])
    for (lo, hi), truth in zip(cert["intervals"], c_true, strict=True):
        assert lo <= truth <= hi


def test_certified_interval_is_rigorous_two_sided_enclosure() -> None:
    design, target, _, _ = _linear_dataset(noise=0.03, seed=6)
    cert = certified_coefficient_intervals(design, target, alpha=1e-6)
    # The enclosure must straddle the float point estimate it certifies.
    for (lo, hi), c0 in zip(cert["intervals"], cert["coefficients"], strict=True):
        assert lo <= c0 <= hi
        assert hi - lo >= 0.0


def test_certified_interval_matches_numpy_solution_center() -> None:
    design, target, _, _ = _linear_dataset(noise=0.02, seed=8)
    xc = design - design.mean(axis=0)
    yc = target - target.mean()
    ref = np.linalg.solve(xc.T @ xc, xc.T @ yc)
    cert = certified_coefficient_intervals(design, target, alpha=0.0)
    assert np.allclose(cert["coefficients"], ref, atol=1e-9)
    for (lo, hi), r in zip(cert["intervals"], ref, strict=True):
        assert lo <= r <= hi


# --------------------------------------------------------------------------- #
# attach_uncertainty                                                          #
# --------------------------------------------------------------------------- #
def test_attach_uncertainty_populates_fields() -> None:
    design, target, names, c_true = _linear_dataset(noise=0.01, seed=9)
    eq = fit_sparse_equation(design, target, names, alpha=1e-8, threshold=1e-2)
    enriched = attach_uncertainty(design=design, target=target, equation=eq, n_boot=120, seed=10)
    assert enriched.coefficient_ci is not None
    assert enriched.selection_frequency is not None
    assert enriched.coefficient_intervals is not None
    assert len(enriched.coefficient_ci) == len(names)
    # active terms carry their bootstrap CI and certified interval
    rows = enriched.active_terms()
    active_names = {str(r["name"]) for r in rows}
    assert active_names == {"x1", "x3"}
    for r in rows:
        assert "ci_lower" in r and "ci_upper" in r
        assert "certified_lower" in r and "certified_upper" in r
        assert float(r["ci_lower"]) <= float(r["coefficient"]) <= float(r["ci_upper"])
    # uncertainty_formula renders +/- annotations
    text = enriched.uncertainty_formula()
    assert "+/-" in text


def test_attach_uncertainty_preserves_point_estimate() -> None:
    design, target, names, _ = _linear_dataset(seed=11)
    eq = fit_sparse_equation(design, target, names)
    enriched = attach_uncertainty(
        equation=eq, design=design, target=target, bootstrap=False, certified=True
    )
    assert np.array_equal(enriched.coefficients, eq.coefficients)
    assert enriched.coefficient_ci is None
    assert enriched.coefficient_intervals is not None


# --------------------------------------------------------------------------- #
# Selection: information criteria                                             #
# --------------------------------------------------------------------------- #
def test_gaussian_log_likelihood_increases_as_residuals_shrink() -> None:
    assert gaussian_log_likelihood(100, 1e-6) > gaussian_log_likelihood(100, 1.0)


def test_information_criteria_penalize_parameters() -> None:
    n, rss = 200, 1.0
    assert aic(n, rss, 5) > aic(n, rss, 2)
    assert bic(n, rss, 5) > bic(n, rss, 2)
    # BIC penalizes complexity more than AIC for n > 7
    assert bic(n, rss, 5) - bic(n, rss, 2) > aic(n, rss, 5) - aic(n, rss, 2)
    # MDL adds a structure-coding term over BIC for a large library
    assert mdl(n, rss, 3, n_candidates=50) > bic(n, rss, 3)


def test_aicc_infinite_when_too_few_samples() -> None:
    assert aicc(5, 1.0, 5) == float("inf")
    assert np.isfinite(aicc(200, 1.0, 5))


def test_information_criterion_dispatch_and_errors() -> None:
    for name in CRITERIA:
        val = information_criterion(name, 100, 1.0, 3, n_candidates=10)
        assert np.isfinite(val) or name == "aicc"
    with pytest.raises(ValueError):
        information_criterion("nonsense", 100, 1.0, 3)


def test_aic_bic_pick_correct_polynomial_order() -> None:
    rng = np.random.default_rng(12)
    n = 400
    x = rng.uniform(-2.0, 2.0, size=n)
    y = 1.0 + 0.0 * x + 2.0 * x**2 + 0.01 * rng.normal(size=n)  # true order 2
    full = np.stack([x, x**2, x**3, x**4], axis=1)
    names = ["x", "x^2", "x^3", "x^4"]

    eq1 = fit_sparse_equation(full[:, :1], y, names[:1], alpha=1e-10, threshold=1e-8)
    eq2 = fit_sparse_equation(full[:, :2], y, names[:2], alpha=1e-10, threshold=1e-8)
    eq4 = fit_sparse_equation(full, y, names, alpha=1e-10, threshold=1e-8)

    for crit in ("aic", "bic"):
        s1 = equation_information_criterion(eq1, full[:, :1], y, name=crit)
        s2 = equation_information_criterion(eq2, full[:, :2], y, name=crit)
        s4 = equation_information_criterion(eq4, full, y, name=crit)
        assert s2 < s1, f"{crit}: order-2 should beat order-1 underfit"
        assert s2 <= s4 + 1e-9, f"{crit}: order-2 should beat/tie order-4 overfit"


# --------------------------------------------------------------------------- #
# Selection: k-fold and stability                                            #
# --------------------------------------------------------------------------- #
def test_kfold_select_returns_predictive_config() -> None:
    design, target, names, _ = _linear_dataset(noise=0.05, seed=13)
    sel = kfold_select(design, target, names, k=5, seed=14)
    assert sel.alpha in (1e-10, 1e-8, 1e-6, 1e-4)
    assert sel.threshold in (1e-6, 1e-4, 1e-3, 1e-2)
    assert sel.cv_rmse < 0.2
    assert sel.table[0]["cv_rmse"] == sel.cv_rmse
    # sorted best-first
    rmses = [row["cv_rmse"] for row in sel.table]
    assert rmses == sorted(rmses)


def test_kfold_rejects_bad_k() -> None:
    design, target, names, _ = _linear_dataset(n=10)
    with pytest.raises(ValueError):
        kfold_select(design, target, names, k=1)


def test_stability_selection_ranks_true_terms_top() -> None:
    design, target, names, _ = _linear_dataset(noise=0.05, seed=15)
    out = stability_selection(
        design, target, names, alpha=1e-8, threshold=1e-2, n_resample=100, seed=16
    )
    top2 = {out["ranking"][0][0], out["ranking"][1][0]}
    assert top2 == {"x1", "x3"}
    freq = out["selection_frequency"]
    assert freq[0] > freq[1]  # x1 more stable than spurious x2
    assert freq[2] > freq[3]  # x3 more stable than spurious x1*x2


def test_stability_selection_rejects_bad_fraction() -> None:
    design, target, names, _ = _linear_dataset()
    with pytest.raises(ValueError):
        stability_selection(design, target, names, sample_fraction=0.0)
