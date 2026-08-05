# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Residual diagnostics + divergence objectives wired into the discovery engines.

Both :func:`discover_interpretable_surrogate` and :class:`NeuralJetDiscoverer`
always report an information-theoretic / optimal-transport residual diagnostic of
the selected model, and can optionally fold a divergence term into the selection
objective. These tests check the diagnostics are correct (a correct model leaves
structure-free residuals; an underfit one does not) and that the objective is
genuinely added to the selection score.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from omnibias.symbolic.diagnostics import divergence_objective_term
from omnibias.symbolic.discovery import (
    JetBundle,
    LibrarySpec,
    NeuralJetDiscoverer,
    SplitData,
    build_jet_relation_library,
    build_taylor_library,
    discover_interpretable_surrogate,
    fit_sparse_equation,
)

_DIAG_KEYS = {
    "rmse",
    "std",
    "differential_entropy",
    "gaussian_reference_entropy",
    "kl_to_gaussian",
    "wasserstein_to_gaussian",
    "feature_residual_mi",
    "max_feature_residual_mi",
}

#: The jet path additionally reports how well-posed the library it fitted was.
#: Only there: a surrogate search chooses among whole feature bases rather than
#: assembling one design matrix whose columns can go collinear.
_CONDITION_KEYS = {
    "design_condition_number",
    "standardized_condition_number",
    "max_column_scale_ratio",
}


def _quadratic_split(*, n: int = 3000, noise: float = 0.02, seed: int = 0) -> SplitData:
    """A degree-2 law (in the Taylor library) plus small Gaussian noise."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3.0, 3.0, size=(n, 1))

    def y(xx: np.ndarray) -> np.ndarray:
        return 0.5 * xx[:, 0] ** 2 - 0.3 * xx[:, 0]

    order = rng.permutation(n)
    n_tr, n_va = int(0.6 * n), int(0.2 * n)
    tr, va, te = order[:n_tr], order[n_tr : n_tr + n_va], order[n_tr + n_va :]
    return SplitData(
        x_train=x[tr],
        y_train=y(x[tr]) + rng.normal(0.0, noise, tr.size),
        x_val=x[va],
        y_val=y(x[va]) + rng.normal(0.0, noise, va.size),
        x_test=x[te],
        y_test=y(x[te]) + rng.normal(0.0, noise, te.size),
    )


def _taylor1_spec() -> LibrarySpec:
    return LibrarySpec(
        name="taylor1",
        builder=lambda xx: build_taylor_library(xx, max_degree=1),
        description="degree-1 Taylor (deliberately underfits a quadratic)",
    )


def _exp_bundles() -> tuple[JetBundle, JetBundle, JetBundle]:
    """``dy = y`` (exp) -- exactly recovered by the polynomial jet library."""

    def bundle(x: np.ndarray) -> JetBundle:
        y = np.exp(0.4 * x)
        return JetBundle(x=x, jets=np.stack([y, 0.4 * y], axis=1))

    return (
        bundle(np.linspace(-2.0, 2.0, 200)),
        bundle(np.linspace(-1.8, 1.8, 150)),
        bundle(np.linspace(-1.9, 1.9, 150)),
    )


def _surprisal_bundles() -> tuple[JetBundle, JetBundle, JetBundle]:
    """``dy = surprisal_arctan(x)`` over a wide range -- logarithmic tails a degree-2
    polynomial jet library genuinely cannot fit, leaving large structured residuals."""

    def surprisal_arctan(x: np.ndarray) -> np.ndarray:
        return np.log(np.pi) + np.log1p(x * x)

    def bundle(x: np.ndarray) -> JetBundle:
        return JetBundle(x=x, jets=np.stack([np.cos(x), surprisal_arctan(x)], axis=1))

    return (
        bundle(np.linspace(-6.0, 6.0, 300)),
        bundle(np.linspace(-5.5, 5.5, 200)),
        bundle(np.linspace(-5.8, 5.8, 200)),
    )


# ----- surrogate diagnostics ------------------------------------------------


def test_surrogate_always_reports_residual_diagnostics() -> None:
    data = _quadratic_split(seed=1)
    result = discover_interpretable_surrogate(data)
    assert "residual_diagnostics" in result
    diag = result["residual_diagnostics"]
    assert isinstance(diag, dict)
    assert set(diag) == _DIAG_KEYS


def test_surrogate_good_fit_has_gaussian_structurefree_residuals() -> None:
    data = _quadratic_split(seed=2, noise=0.02)
    diag = discover_interpretable_surrogate(data)["residual_diagnostics"]
    # residual is essentially the injected Gaussian noise
    assert float(diag["std"]) == pytest.approx(0.02, abs=0.01)
    assert float(diag["kl_to_gaussian"]) < 0.1
    assert float(diag["max_feature_residual_mi"]) < 0.1


def test_surrogate_underfit_leaves_input_residual_structure() -> None:
    data = _quadratic_split(seed=3, noise=0.02)
    good = discover_interpretable_surrogate(data)["residual_diagnostics"]
    bad = discover_interpretable_surrogate(data, specs=[_taylor1_spec()])["residual_diagnostics"]
    # the degree-1 model cannot represent the x^2 term, so the residual stays
    # strongly dependent on x: its input-residual MI is far higher.
    assert float(bad["max_feature_residual_mi"]) > float(good["max_feature_residual_mi"]) + 0.3


def test_surrogate_divergence_objective_is_added_to_score() -> None:
    data = _quadratic_split(seed=4)
    spec = _taylor1_spec()  # structured residuals -> nonzero residual_mi term
    common = {"specs": [spec], "alphas": (1e-8,), "thresholds": (1e-4,)}
    off = discover_interpretable_surrogate(data, **common)
    on = discover_interpretable_surrogate(
        data, **common, divergence_objective="residual_mi", divergence_weight=2.5
    )
    # independently reconstruct the single-fit validation residual term
    d_tr, names = spec.builder(data.x_train)
    d_va, _ = spec.builder(data.x_val)
    eq = fit_sparse_equation(d_tr, data.y_train, names, alpha=1e-8, threshold=1e-4)
    term = divergence_objective_term("residual_mi", data.x_val, data.y_val - eq.predict(d_va))
    assert term > 0.0
    off_score = float(off["selection"]["selection_score"])
    on_score = float(on["selection"]["selection_score"])
    assert on_score - off_score == pytest.approx(2.5 * term, abs=1e-9)


def test_surrogate_explicit_specs_still_report_diagnostics() -> None:
    data = _quadratic_split(seed=5)
    result = discover_interpretable_surrogate(data, specs=[_taylor1_spec()])
    assert set(result["residual_diagnostics"]) == _DIAG_KEYS


# ----- NeuralJet diagnostics ------------------------------------------------


def test_jet_result_reports_diagnostics() -> None:
    tr, va, te = _exp_bundles()
    result = NeuralJetDiscoverer(max_library_degree=2).discover(
        tr, va, te, candidate_lhs_orders=(1,)
    )
    assert set(result.diagnostics) == _DIAG_KEYS | _CONDITION_KEYS


def test_jet_diagnostics_report_a_well_posed_library_as_well_conditioned() -> None:
    """The polynomial jet library on a smooth signal is not collinear."""
    tr, va, te = _exp_bundles()
    diag = NeuralJetDiscoverer(max_library_degree=2).discover(
        tr, va, te, candidate_lhs_orders=(1,)
    ).diagnostics
    assert float(diag["standardized_condition_number"]) < 1e6
    assert math.isfinite(float(diag["max_column_scale_ratio"]))


def test_jet_exact_identity_has_negligible_residual_amplitude() -> None:
    tr, va, te = _exp_bundles()
    result = NeuralJetDiscoverer(max_library_degree=2).discover(
        tr, va, te, candidate_lhs_orders=(1,)
    )
    assert result.test_rmse < 1e-8
    # amplitude (scale-sensitive) is what matters for a near-perfect fit: the
    # residual std and Wasserstein-to-Gaussian collapse to machine scale. (The
    # scale-invariant KL / MI can still flag the smooth ~1e-9 numerical
    # micro-residual, which is honest but immaterial -- hence not asserted here.)
    assert float(result.diagnostics["std"]) < 1e-7
    assert float(result.diagnostics["wasserstein_to_gaussian"]) < 1e-7


def test_jet_underfit_has_more_residual_structure_than_exact() -> None:
    exact = NeuralJetDiscoverer(max_library_degree=2).discover(
        *_exp_bundles(), candidate_lhs_orders=(1,)
    )
    underfit = NeuralJetDiscoverer(max_library_degree=2).discover(
        *_surprisal_bundles(), candidate_lhs_orders=(1,)
    )
    # the polynomial library cannot represent the surprisal law, so it leaves a
    # large, non-Gaussian, input-dependent residual: amplitude (W1, std) and the
    # bias-corrected dependence MI all dwarf the near-exact exp identity.
    assert float(underfit.diagnostics["wasserstein_to_gaussian"]) > float(
        exact.diagnostics["wasserstein_to_gaussian"]
    ) + 1e-3
    assert float(underfit.diagnostics["std"]) > float(exact.diagnostics["std"]) + 1e-3
    assert float(underfit.diagnostics["max_feature_residual_mi"]) > 0.5


def test_jet_divergence_objective_is_added_to_score() -> None:
    tr, va, te = _surprisal_bundles()  # poly library underfits -> structured residuals
    off = NeuralJetDiscoverer(
        max_library_degree=2, alphas=(1e-8,), thresholds=(1e-4,)
    ).discover(tr, va, te, candidate_lhs_orders=(1,))
    on = NeuralJetDiscoverer(
        max_library_degree=2,
        alphas=(1e-8,),
        thresholds=(1e-4,),
        divergence_objective="residual_mi",
        divergence_weight=2.5,
    ).discover(tr, va, te, candidate_lhs_orders=(1,))
    # reconstruct the single-fit validation residual term on the jet variables
    tr_design, names = build_jet_relation_library(tr, lhs_order=1, max_degree=2)
    va_design, _ = build_jet_relation_library(va, lhs_order=1, max_degree=2)
    eq = fit_sparse_equation(tr_design, tr.jets[:, 1], names, alpha=1e-8, threshold=1e-4)
    val_feat = np.column_stack([va.x, va.jets[:, 0]])  # variables = [x, y] for lhs_order=1
    term = divergence_objective_term("residual_mi", val_feat, va.jets[:, 1] - eq.predict(va_design))
    assert term > 0.0
    assert on.selection_score - off.selection_score == pytest.approx(2.5 * term, abs=1e-9)


def test_jet_diagnostics_default_empty_on_direct_construction() -> None:
    from omnibias.symbolic.discovery import JetDiscoveryResult, SparseEquation

    eq = SparseEquation(
        term_names=("y",),
        coefficients=np.array([1.0]),
        intercept=0.0,
        alpha=1e-8,
        threshold=1e-4,
        active_mask=np.array([True]),
    )
    result = JetDiscoveryResult(
        lhs_order=1,
        equation=eq,
        validation_rmse=0.0,
        test_rmse=0.0,
        selection_score=0.0,
        target_scale=1.0,
    )
    assert result.diagnostics == {}
