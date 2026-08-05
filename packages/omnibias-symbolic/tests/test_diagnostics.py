# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Residual / distributional diagnostics for symbolic discovery.

These exercise the numpy point-estimate twins of the omnibias information /
optimal-transport operators (entropy, KL, JS, mutual information) and the
residual-report glue (differential entropy, KL / W1 to a matched Gaussian,
input-residual mutual information) used as fit diagnostics and selection
objectives by the discovery engines.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from omnibias.symbolic.diagnostics import (
    DIVERGENCE_OBJECTIVES,
    differential_entropy,
    divergence_objective_term,
    entropy,
    feature_residual_mutual_information,
    gaussian_entropy,
    histogram_pmf,
    js_divergence,
    kl_divergence,
    kl_to_gaussian,
    mutual_information,
    residual_dependence_report,
    residual_distribution_report,
    surrogate_residual_diagnostics,
    wasserstein_to_gaussian,
)

# ----- discrete functionals: closed-form truths -----------------------------


def test_entropy_uniform_is_log_k() -> None:
    for k in (2, 5, 17):
        p = np.full(k, 1.0 / k)
        assert float(entropy(p)) == pytest.approx(math.log(k), abs=1e-12)


def test_entropy_point_mass_is_zero() -> None:
    p = np.array([0.0, 1.0, 0.0, 0.0])
    assert float(entropy(p)) == pytest.approx(0.0, abs=1e-12)


def test_entropy_handles_zero_bins_without_warning() -> None:
    # 0 * ln 0 := 0 -- empty histogram bins must not raise or warn.
    p = np.array([0.5, 0.0, 0.5, 0.0])
    assert float(entropy(p)) == pytest.approx(math.log(2.0), abs=1e-12)


def test_kl_self_is_zero_and_nonnegative() -> None:
    rng = np.random.default_rng(0)
    for _ in range(20):
        p = rng.random(6)
        p /= p.sum()
        q = rng.random(6)
        q /= q.sum()
        assert float(kl_divergence(p, p)) == pytest.approx(0.0, abs=1e-12)
        assert float(kl_divergence(p, q)) >= -1e-12


def test_kl_uncovered_support_is_inf_not_a_warning() -> None:
    # p has mass where q == 0 -> +inf, computed without a numpy divide warning
    # (the suite runs under filterwarnings=error).
    p = np.array([0.5, 0.0, 0.5])
    q = np.array([0.5, 0.5, 0.0])
    assert math.isinf(float(kl_divergence(p, q)))


def test_js_is_symmetric_and_bounded_by_ln2() -> None:
    rng = np.random.default_rng(1)
    for _ in range(20):
        p = rng.random(8)
        p /= p.sum()
        q = rng.random(8)
        q /= q.sum()
        d_pq = float(js_divergence(p, q))
        d_qp = float(js_divergence(q, p))
        assert d_pq == pytest.approx(d_qp, abs=1e-12)
        assert -1e-12 <= d_pq <= math.log(2.0) + 1e-12


def test_mutual_information_zero_for_product_distribution() -> None:
    px = np.array([0.2, 0.3, 0.5])
    py = np.array([0.4, 0.6])
    joint = np.outer(px, py)
    assert float(mutual_information(joint)) == pytest.approx(0.0, abs=1e-12)


def test_mutual_information_diagonal_equals_marginal_entropy() -> None:
    marg = np.array([0.2, 0.3, 0.5])
    joint = np.diag(marg)  # Y is a deterministic copy of X
    assert float(mutual_information(joint)) == pytest.approx(float(entropy(marg)), abs=1e-12)


@pytest.mark.filterwarnings("ignore")
def test_parity_with_jax_information_operators() -> None:
    """The numpy functionals are bit-identical twins of omnibias.jax.information."""
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from omnibias.jax import information as ji

    rng = np.random.default_rng(7)
    p = rng.random(6)
    p /= p.sum()
    q = rng.random(6)
    q /= q.sum()
    joint = rng.random((4, 5))
    joint /= joint.sum()
    assert float(entropy(p)) == pytest.approx(float(ji.entropy(jnp.asarray(p))), rel=1e-12)
    assert float(kl_divergence(p, q)) == pytest.approx(
        float(ji.kl_divergence(jnp.asarray(p), jnp.asarray(q))), rel=1e-12
    )
    assert float(js_divergence(p, q)) == pytest.approx(
        float(ji.js_divergence(jnp.asarray(p), jnp.asarray(q))), rel=1e-12
    )
    assert float(mutual_information(joint)) == pytest.approx(
        float(ji.mutual_information(jnp.asarray(joint))), rel=1e-12
    )


# ----- histogram / Gaussian glue --------------------------------------------


def test_histogram_pmf_sums_to_one() -> None:
    rng = np.random.default_rng(2)
    pmf, edges = histogram_pmf(rng.normal(size=1000), bins=16)
    assert pmf.shape == (16,)
    assert edges.shape == (17,)
    assert float(pmf.sum()) == pytest.approx(1.0, abs=1e-12)


def test_histogram_pmf_empty_raises() -> None:
    with pytest.raises(ValueError, match="at least one sample"):
        histogram_pmf(np.array([]))


def test_gaussian_entropy_closed_form() -> None:
    for sigma in (0.5, 1.0, 3.0):
        assert gaussian_entropy(sigma) == pytest.approx(
            0.5 * math.log(2.0 * math.pi * math.e * sigma * sigma), abs=1e-12
        )


def test_gaussian_entropy_rejects_nonpositive_std() -> None:
    with pytest.raises(ValueError, match="std > 0"):
        gaussian_entropy(0.0)


def test_differential_entropy_recovers_uniform_support() -> None:
    # Uniform(a, b) has differential entropy ln(b - a).
    rng = np.random.default_rng(3)
    samples = rng.uniform(-2.0, 2.0, size=40000)
    assert differential_entropy(samples, bins=64) == pytest.approx(math.log(4.0), abs=0.05)


def test_differential_entropy_recovers_gaussian_reference() -> None:
    rng = np.random.default_rng(4)
    sigma = 1.7
    samples = rng.normal(0.0, sigma, size=40000)
    assert differential_entropy(samples, bins=64) == pytest.approx(gaussian_entropy(sigma), abs=0.1)


def test_kl_to_gaussian_small_for_gaussian_large_for_bimodal() -> None:
    rng = np.random.default_rng(5)
    gauss = rng.normal(0.0, 1.0, size=20000)
    bimodal = np.concatenate([rng.normal(-3.0, 0.3, 10000), rng.normal(3.0, 0.3, 10000)])
    assert kl_to_gaussian(gauss, bins=40) < 0.02
    assert kl_to_gaussian(bimodal, bins=40) > 0.5


def test_kl_to_gaussian_zero_variance_is_zero() -> None:
    assert kl_to_gaussian(np.full(100, 2.5)) == 0.0


def test_wasserstein_to_gaussian_small_for_gaussian_positive_for_uniform() -> None:
    rng = np.random.default_rng(6)
    gauss = rng.normal(0.0, 1.0, size=20000)
    uniform = rng.uniform(-2.0, 2.0, size=20000)
    assert wasserstein_to_gaussian(gauss) < 0.02
    assert wasserstein_to_gaussian(uniform) > 0.1


def test_wasserstein_to_gaussian_guards() -> None:
    assert wasserstein_to_gaussian(np.full(50, 1.0)) == 0.0
    with pytest.raises(ValueError, match="at least one sample"):
        wasserstein_to_gaussian(np.array([]))


def test_feature_residual_mi_zero_for_independent_positive_for_dependent() -> None:
    rng = np.random.default_rng(8)
    x = rng.uniform(-3.0, 3.0, size=20000)
    indep = rng.normal(0.0, 1.0, size=20000)
    dependent = np.sin(2.0 * x)  # fully determined by x
    mi_indep = feature_residual_mutual_information(x, indep, bins=24)
    mi_dep = feature_residual_mutual_information(x, dependent, bins=24)
    assert mi_indep < 0.05
    assert mi_dep > mi_indep + 0.3


def test_feature_residual_mi_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="equal length"):
        feature_residual_mutual_information(np.zeros(10), np.zeros(9))


# ----- reports + objective dispatch -----------------------------------------


def test_residual_distribution_report_keys_and_gaussian_values() -> None:
    rng = np.random.default_rng(9)
    r = rng.normal(0.0, 0.5, size=8000)
    report = residual_distribution_report(r, bins=40)
    assert set(report) == {
        "rmse",
        "std",
        "differential_entropy",
        "gaussian_reference_entropy",
        "kl_to_gaussian",
        "wasserstein_to_gaussian",
    }
    assert report["std"] == pytest.approx(0.5, abs=0.02)
    # near-Gaussian residual: actual entropy close to the Gaussian reference.
    assert report["differential_entropy"] == pytest.approx(
        report["gaussian_reference_entropy"], abs=0.1
    )
    assert report["kl_to_gaussian"] < 0.02


def test_residual_distribution_report_degenerate_is_safe() -> None:
    report = residual_distribution_report(np.zeros(64))
    assert report["rmse"] == 0.0
    assert report["std"] == 0.0
    assert math.isnan(float(report["differential_entropy"]))
    assert report["kl_to_gaussian"] == 0.0
    assert report["wasserstein_to_gaussian"] == 0.0


def test_residual_dependence_report_shape() -> None:
    rng = np.random.default_rng(10)
    x = rng.uniform(-1.0, 1.0, size=(2000, 3))
    r = rng.normal(size=2000)
    report = residual_dependence_report(x, r, bins=12)
    mis = report["feature_residual_mi"]
    assert isinstance(mis, list)
    assert len(mis) == 3
    assert report["max_feature_residual_mi"] == pytest.approx(max(mis))


def test_surrogate_residual_diagnostics_combines_reports() -> None:
    rng = np.random.default_rng(11)
    x = rng.uniform(-2.0, 2.0, size=(3000, 2))
    y_true = rng.normal(size=3000)
    y_pred = np.zeros(3000)
    diag = surrogate_residual_diagnostics(x, y_true, y_pred, bins=32, dependence_bins=12)
    assert "kl_to_gaussian" in diag
    assert "feature_residual_mi" in diag
    assert len(diag["feature_residual_mi"]) == 2


def test_divergence_objective_term_dispatch() -> None:
    rng = np.random.default_rng(12)
    x = rng.uniform(-2.0, 2.0, size=(4000, 1))
    r = rng.normal(0.0, 1.0, size=4000)
    assert divergence_objective_term("kl_gaussian", x, r) == pytest.approx(
        kl_to_gaussian(r), abs=1e-12
    )
    assert divergence_objective_term("wasserstein_gaussian", x, r) == pytest.approx(
        wasserstein_to_gaussian(r), abs=1e-12
    )
    expected_mi = residual_dependence_report(x, r)["max_feature_residual_mi"]
    assert divergence_objective_term("residual_mi", x, r) == pytest.approx(expected_mi, abs=1e-12)


def test_divergence_objectives_are_nonnegative() -> None:
    rng = np.random.default_rng(13)
    x = rng.uniform(-2.0, 2.0, size=(3000, 1))
    r = rng.normal(size=3000)
    for name in DIVERGENCE_OBJECTIVES:
        assert divergence_objective_term(name, x, r) >= -1e-12


def test_divergence_objective_term_unknown_raises() -> None:
    with pytest.raises(ValueError, match="unknown divergence objective"):
        divergence_objective_term("not_a_thing", np.zeros((4, 1)), np.zeros(4))
