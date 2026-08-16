# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Validation tests for omnibias.symbolic equation discovery."""

from __future__ import annotations

import json

import numpy as np
from omnibias.symbolic.blasius import (
    discover_blasius_explicit_expression,
    discover_blasius_from_neural_surrogate,
    discover_blasius_identity,
    discover_blasius_taylor_pade_expression,
    evaluate_blasius,
    solve_blasius,
    write_blasius_artifacts,
)
from omnibias.symbolic.discovery import (
    NeuralJetDiscoverer,
    build_hybrid_library,
    build_jet_relation_library,
    build_taylor_library,
    discover_activation_identity,
    discover_from_noisy_observations,
    discover_interpretable_surrogate,
    discover_pde_operator_law,
    evaluate_high_dim_sparse_validation,
    evaluate_poc,
    evaluate_real_world_tabular_validation,
    exact_activation_field_1d,
    extract_neural_jets,
    fit_sparse_equation,
    jet_name,
    make_symbolic_regression_dataset,
    write_artifacts,
)
from omnibias.symbolic.expressions import (
    fit_rational_expression,
    recognize_known_expression,
)


def test_taylor_library_names_polynomial_interactions() -> None:
    x = np.asarray([[2.0, 3.0, 5.0]])
    design, names = build_taylor_library(x, max_degree=2)
    assert "x1" in names
    assert "x1^2" in names
    assert "x1*x2" in names
    assert design[0, names.index("x1*x2")] == 6.0


def test_sparse_equation_formats_active_terms() -> None:
    x = np.linspace(-1.0, 1.0, 80)[:, None]
    design, names = build_hybrid_library(x, max_degree=2, max_frequency=1)
    y = 2.0 * x[:, 0] ** 2 + np.sin(x[:, 0])
    equation = fit_sparse_equation(design, y, names, threshold=1e-6)
    formula = equation.formula(lhs="y")
    active = {row["name"] for row in equation.active_terms()}
    assert "x1^2" in active
    assert "sin(x1)" in active
    assert formula.startswith("y =")


def test_sparse_equation_thresholding_is_scale_invariant() -> None:
    """Thresholding lives in a single standardized space, so rescaling a
    library column must not change which terms are selected -- only the raw
    coefficient of that column adjusts. Guards the STLSQ single-space cull."""
    x = np.linspace(-1.0, 1.0, 120)[:, None]
    design, names = build_hybrid_library(x, max_degree=2, max_frequency=1)
    y = 2.0 * x[:, 0] ** 2 + np.sin(x[:, 0])
    col = names.index("x1^2")

    eq = fit_sparse_equation(design, y, names, threshold=1e-3)
    scaled = design.copy()
    scaled[:, col] *= 1000.0
    eq_scaled = fit_sparse_equation(scaled, y, names, threshold=1e-3)

    # Same support despite the 1000x rescale of one column.
    np.testing.assert_array_equal(eq.active_mask, eq_scaled.active_mask)
    # That column's raw coefficient shrinks by exactly the rescale factor.
    np.testing.assert_allclose(
        eq_scaled.coefficients[col] * 1000.0,
        eq.coefficients[col],
        rtol=1e-6,
        atol=1e-9,
    )
    # The two parameterizations make identical predictions.
    np.testing.assert_allclose(
        eq_scaled.predict(scaled), eq.predict(design), rtol=1e-6, atol=1e-6
    )


def test_huber_stlsq_beats_ridge_on_one_outlier() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=80)
    y = rng.normal(size=80)
    target = 0.4 * x - 0.25 * x * y
    design = np.column_stack([np.ones(80), x, y, x * y])
    names = ["1", "x", "y", "xy"]
    dirty = target.copy()
    dirty[0] += 25.0
    ridge = fit_sparse_equation(design, dirty, names, alpha=1e-8, threshold=1e-4)
    huber = fit_sparse_equation(
        design, dirty, names, alpha=1e-8, threshold=1e-4, loss="huber"
    )
    true = np.asarray([0.0, 0.4, 0.0, -0.25])
    assert float(np.linalg.norm(huber.coefficients - true)) < float(
        np.linalg.norm(ridge.coefficients - true)
    )


def test_huber_stlsq_keeps_ridge_signs_on_clean_data() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=60)
    y = rng.normal(size=60)
    target = 0.4 * x - 0.25 * x * y
    design = np.column_stack([np.ones(60), x, y, x * y])
    names = ["1", "x", "y", "xy"]
    ridge = fit_sparse_equation(design, target, names, alpha=1e-8, threshold=1e-3)
    huber = fit_sparse_equation(
        design, target, names, alpha=1e-8, threshold=1e-3, loss="huber"
    )
    ridge_signs = np.sign(ridge.coefficients) * ridge.active_mask
    huber_signs = np.sign(huber.coefficients) * huber.active_mask
    np.testing.assert_array_equal(ridge_signs, huber_signs)


def test_neural_field_weight_scale_default_matches_unit_scale() -> None:
    from omnibias.symbolic.discovery import fit_neural_field_1d

    t = np.linspace(0.0, 2.0, 40)
    y = np.sin(t)
    default = fit_neural_field_1d(t, y, hidden=16, seed=0)
    scaled = fit_neural_field_1d(t, y, hidden=16, seed=0, weight_scale=1.0)
    np.testing.assert_allclose(default.W, scaled.W)
    np.testing.assert_allclose(default.c, scaled.c)
    assert default.train_rmse == scaled.train_rmse


def test_surrogate_automl_selects_hybrid_and_recovers_terms() -> None:
    data = make_symbolic_regression_dataset(n_samples=700, noise_std=0.0, seed=4)
    result = discover_interpretable_surrogate(data, complexity_weight=5e-4)
    active = {row["name"] for row in result["selected_terms"]}
    assert result["family"] == "taylor_fourier"
    assert "x1^2" in active
    assert "x2*x3" in active
    assert "sin(2*x4)" in active
    assert "cos(x4)" in active
    assert result["metrics"]["rmse"] < 1e-5


def test_pde_operator_discovery_recovers_heat_equation() -> None:
    result = discover_pde_operator_law(diffusivity=0.12)
    active = {row["name"]: row["coefficient"] for row in result["selected_terms"]}
    assert "u_xx" in active
    assert abs(active["u_xx"] - 0.12) < 1e-4
    assert result["metrics"]["rmse"] < 1e-8


def test_poc_writes_metrics_and_report(tmp_path) -> None:
    results = evaluate_poc(n_samples=500, noise_std=0.0, seed=1)
    write_artifacts(results, tmp_path)
    metrics_path = tmp_path / "metrics.json"
    report_path = tmp_path / "report.md"
    assert metrics_path.exists()
    assert report_path.exists()
    payload = json.loads(metrics_path.read_text())
    assert payload["symbolic_surrogate"]["family"] == "taylor_fourier"
    assert "neural_jet_discovery" in payload
    assert "u_xx" in report_path.read_text()


def test_neural_jet_discoverer_empty_search_raises_runtime_error() -> None:
    """Empty alpha/threshold grids must raise, not silently ``assert``-fail."""
    import pytest

    field = exact_activation_field_1d("exp")
    x = np.linspace(-1.0, 1.0, 12)
    train = extract_neural_jets(field, x, max_order=2)
    val = extract_neural_jets(field, x, max_order=2)
    test = extract_neural_jets(field, x, max_order=2)
    discoverer = NeuralJetDiscoverer(alphas=(), thresholds=())
    with pytest.raises(RuntimeError, match="no candidates"):
        discoverer.discover(train, val, test, candidate_lhs_orders=(1,))


def test_write_artifacts_rejects_non_dict_payload(tmp_path) -> None:
    """Runtime TypeError replaces assert-based type narrowing in write_artifacts."""
    import pytest

    bad = {
        "symbolic_surrogate": "not-a-dict",
        "operator_discovery": {},
        "neural_jet_discovery": {},
        "high_dimensional_sparse_validation": {},
        "real_world_tabular_validation": {},
    }
    with pytest.raises(TypeError, match="symbolic_surrogate"):
        write_artifacts(bad, tmp_path)


def test_neural_jet_extractor_uses_closed_form_fastpath_for_exp() -> None:
    field = exact_activation_field_1d("exp")
    x = np.linspace(-1.0, 1.0, 25)
    bundle = extract_neural_jets(field, x, max_order=3)
    expected = np.exp(x)
    assert bundle.jets.shape == (25, 4)
    assert jet_name(3) == "d3y"
    assert np.allclose(bundle.jets[:, 0], expected, atol=1e-10)
    assert np.allclose(bundle.jets[:, 1], expected, atol=1e-10)
    assert np.allclose(bundle.jets[:, 2], expected, atol=1e-10)
    assert np.allclose(bundle.jets[:, 3], expected, atol=1e-10)


def test_jet_relation_library_has_no_named_function_primitives() -> None:
    field = exact_activation_field_1d("sin")
    x = np.linspace(-1.0, 1.0, 16)
    bundle = extract_neural_jets(field, x, max_order=3)
    _, names = build_jet_relation_library(bundle, lhs_order=2, max_degree=2)
    assert "y" in names
    assert "y^2" in names
    assert "dy" in names
    assert all("sin" not in name and "exp" not in name and "tanh" not in name for name in names)


def test_neural_jet_discoverer_recovers_exp_identity_without_named_exp() -> None:
    result = discover_activation_identity("exp", candidate_lhs_orders=(1,))
    active = {row["name"]: row["coefficient"] for row in result.active_terms()}
    assert result.formula().startswith("dy =")
    assert "y" in active
    assert abs(active["y"] - 1.0) < 1e-6
    assert result.test_rmse < 1e-8


def test_neural_jet_discoverer_recovers_sine_identity_without_named_sin() -> None:
    result = discover_activation_identity("sin", x_range=(-np.pi, np.pi), candidate_lhs_orders=(2,))
    active = {row["name"]: row["coefficient"] for row in result.active_terms()}
    assert result.formula().startswith("d2y =")
    assert "y" in active
    assert abs(active["y"] + 1.0) < 1e-6
    assert result.test_rmse < 1e-8


def test_neural_jet_discoverer_recovers_tanh_riccati_identity() -> None:
    result = discover_activation_identity("tanh", candidate_lhs_orders=(1,))
    active = {row["name"]: row["coefficient"] for row in result.active_terms()}
    assert result.formula().startswith("dy =")
    assert "y^2" in active
    assert abs(result.equation.intercept - 1.0) < 1e-5
    assert abs(active["y^2"] + 1.0) < 1e-5
    assert result.test_rmse < 1e-8


def test_known_expression_recognizer_maps_jet_laws_to_functions() -> None:
    exp_result = discover_activation_identity("exp", candidate_lhs_orders=(1,))
    sin_result = discover_activation_identity("sin", x_range=(-np.pi, np.pi), candidate_lhs_orders=(2,))
    tanh_result = discover_activation_identity("tanh", candidate_lhs_orders=(1,))
    exp_expr = recognize_known_expression(exp_result.equation, lhs="dy")
    sin_expr = recognize_known_expression(sin_result.equation, lhs="d2y")
    tanh_expr = recognize_known_expression(tanh_result.equation, lhs="dy")
    assert exp_expr is not None and exp_expr.family == "exponential"
    assert sin_expr is not None and sin_expr.family == "harmonic"
    assert tanh_expr is not None and tanh_expr.family == "tanh_riccati"


def test_rational_expression_has_analytic_derivatives() -> None:
    x = np.linspace(0.0, 1.0, 50)
    y = 1.0 / (1.0 + x)
    expr = fit_rational_expression(x, y, numerator_degree=0, denominator_degree=1)
    probe = np.linspace(0.0, 1.0, 20)
    assert np.allclose(expr.evaluate(probe), 1.0 / (1.0 + probe), atol=1e-10)
    assert np.allclose(expr.evaluate(probe, derivative_order=1), -1.0 / (1.0 + probe) ** 2, atol=1e-8)
    assert np.allclose(expr.evaluate(probe, derivative_order=2), 2.0 / (1.0 + probe) ** 3, atol=1e-7)


def test_noisy_observation_path_fits_field_then_discovers_sine_identity() -> None:
    result = discover_from_noisy_observations(seed=5, noise_std=0.005, hidden=192)
    active = {row["name"]: row["coefficient"] for row in result["selected_terms"]}
    assert "y" in active
    assert abs(active["y"] + 1.0) < 0.2
    assert result["field_train_rmse"] < 0.03
    assert result["true_identity_rmse"] < 0.25


def test_high_dimensional_sparse_validation_recovers_hidden_terms() -> None:
    result = evaluate_high_dim_sparse_validation(n_samples=700, n_features=50, noise_std=0.02, seed=3)
    recovered = set(result["recovered_terms"])
    assert {"x3^2", "x17*x42", "sin(x8)", "x5"}.issubset(recovered)
    assert result["recovery_rate"] == 1.0
    assert result["metrics"]["rmse"] < 0.08


def test_real_world_diabetes_validation_returns_interpretable_equation() -> None:
    result = evaluate_real_world_tabular_validation(dataset="diabetes", seed=2)
    assert result["available"] is True
    symbolic = result["symbolic_autoregressor"]
    raw = result["raw_linear_baseline"]
    assert symbolic["selected_terms"]
    assert symbolic["metrics"]["rmse"] < 80.0
    assert raw["metrics"]["rmse"] < 80.0
    assert "target =" in symbolic["equation"]


def test_blasius_solver_satisfies_boundary_conditions() -> None:
    solution = solve_blasius(n_steps=1200)
    assert abs(solution.f[0]) < 1e-14
    assert abs(solution.fp[0]) < 1e-14
    assert abs(solution.fp[-1] - 1.0) < 1e-9
    assert abs(solution.fpp0 - 0.332057) < 5e-5


def test_blasius_discovery_recovers_governing_identity() -> None:
    solution = solve_blasius(n_steps=1200)
    result = discover_blasius_identity(solution)
    active = {row["name"]: row["coefficient"] for row in result["selected_terms"]}
    assert result["equation"] == "d3f = -0.5*f*d2f"
    assert "f*d2f" in active
    assert abs(active["f*d2f"] + 0.5) < 1e-10
    assert result["metrics"]["test_rmse"] < 1e-10


def test_blind_blasius_neural_surrogate_recovers_residual_coefficient() -> None:
    solution = solve_blasius(n_steps=1200)
    result = discover_blasius_from_neural_surrogate(solution, hidden=512, seed=1)
    active = {row["name"]: row["coefficient"] for row in result["selected_terms"]}
    assert "f*d2f" in active
    assert abs(active["f*d2f"] + 0.5) < 0.01
    assert result["field_train_rmse"] < 2e-6
    assert result["metrics"]["true_d3f_rmse"] < 1e-3


def test_blasius_explicit_rational_expression_is_accurate_and_residual_small() -> None:
    solution = solve_blasius(n_steps=1600)
    result = discover_blasius_explicit_expression(solution)
    metrics = result["metrics"]
    assert result["kind"] == "rational_pade_surrogate"
    assert metrics["value_rmse"] < 1e-6
    assert metrics["residual_rmse"] < 1e-3
    assert metrics["f0_abs"] < 1e-5
    assert metrics["fp0_abs"] < 1e-4
    assert metrics["fp_eta_max_error"] < 1e-4


def test_blasius_taylor_pade_expression_uses_ode_recurrence_and_boundary_fit() -> None:
    solution = solve_blasius(n_steps=1200)
    result = discover_blasius_taylor_pade_expression(solution)
    metrics = result["metrics"]
    shooting = result["shooting"]
    assert result["kind"] == "ode_recurrence_taylor_pade"
    assert abs(shooting["fpp0"] - solution.fpp0) < 2e-4
    assert metrics["value_rmse"] < 2e-3
    assert metrics["residual_rmse"] < 2e-3
    assert metrics["f0_abs"] < 1e-12
    assert metrics["fp0_abs"] < 1e-12
    assert metrics["fp_eta_max_error"] < 1e-8


def test_blasius_artifacts_are_written(tmp_path) -> None:
    payload = evaluate_blasius()
    write_blasius_artifacts(payload, tmp_path)
    assert (tmp_path / "blasius_metrics.json").exists()
    report = (tmp_path / "blasius_report.md").read_text()
    assert "Blasius Boundary-Layer Discovery" in report
    assert "d3f = -0.5*f*d2f" in report
    assert "Blind neural-surrogate recovery" in report
    assert "Explicit rational expression" in report
    assert "ODE-derived Taylor/Pade expression" in report
