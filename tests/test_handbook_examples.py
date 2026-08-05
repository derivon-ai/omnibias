# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Executable mirror of every code example in ``docs/handbook/``.

The handbook (the "Discovery & Calculus Handbook") is written to be
AI-/vibe-coding-friendly: each function ships a copy-paste snippet whose printed
result the prose relies on. This test re-runs those snippets and asserts the key
numbers so a refactor can never silently make the book lie.

The checks are grouped one test per chapter, matching ``docs/handbook/0N-*.md``.
Optional downstream packages (``omnibias-curvature``) are skipped gracefully so
the test runs on any lane that has the stable surface plus ``omnibias-symbolic``.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

jax = pytest.importorskip("jax")
pytest.importorskip("omnibias.symbolic")

jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]


# ===================================================================== ch.1 ==
def test_handbook_ch1_neural_jet_1d() -> None:
    from omnibias.symbolic import (
        NeuralJetDiscoverer,
        build_cdf_band_library,
        build_hybrid_library,
        build_information_library,
        build_jet_fractional_features,
        build_jet_relation_library,
        discover_activation_identity,
        discover_from_noisy_observations,
        discover_interpretable_surrogate,
        discover_pde_operator_law,
        evaluate_high_dim_sparse_validation,
        exact_activation_field_1d,
        extract_neural_jets,
        fit_neural_field_1d,
        fit_sparse_equation,
        gl_fractional_derivative,
        jet_name,
        mae,
        make_high_dim_sparse_dataset,
        make_symbolic_regression_dataset,
        rmse,
        split_x_grid,
    )
    from omnibias.symbolic.discovery import build_jet_cdf_features, fit_jet_cdf_plan

    x = np.linspace(-np.pi, np.pi, 400)
    y = np.sin(x) + 0.01 * np.random.default_rng(0).normal(size=x.size)
    field = fit_neural_field_1d(x, y, hidden=256, ridge=1e-4, seed=0)
    assert field.activation == "tanh"
    assert field.train_rmse < 0.05

    f_tanh = exact_activation_field_1d("tanh")
    jts = extract_neural_jets(f_tanh, np.linspace(-1, 1, 5), max_order=1)
    assert np.allclose(jts.jets[:, 1], 1 - jts.jets[:, 0] ** 2)

    fe = fit_neural_field_1d(
        np.linspace(-2, 2, 300), np.exp(np.linspace(-2, 2, 300)), hidden=256, seed=0
    )
    b = extract_neural_jets(fe, np.linspace(-2, 2, 300), max_order=2)
    assert b.jets.shape == (300, 3)
    assert (b.name(0), b.name(1), b.name(2)) == ("y", "dy", "d2y")
    assert (jet_name(0), jet_name(1), jet_name(2)) == ("y", "dy", "d2y")

    xtr, xva, xte = split_x_grid(xmin=-2.0, xmax=2.0)
    field = exact_activation_field_1d("exp")
    mk = lambda xx: extract_neural_jets(field, xx, max_order=2)  # noqa: E731
    res = NeuralJetDiscoverer(max_library_degree=1).discover(
        mk(xtr), mk(xva), mk(xte), candidate_lhs_orders=(1,)
    )
    assert res.formula() == "dy = 1*y"
    assert res.active_terms()[0]["name"] == "y"

    assert discover_activation_identity("exp", candidate_lhs_orders=(1,)).formula() == "dy = 1*y"
    assert discover_activation_identity("sin", candidate_lhs_orders=(2,)).formula() == "d2y = -1*y"
    assert (
        discover_activation_identity("tanh", candidate_lhs_orders=(1,)).formula()
        == "dy = 1 - 1*y^2"
    )

    out = discover_from_noisy_observations(noise_std=0.01)
    assert "d2y" in out["equation"]

    rng = np.random.default_rng(0)
    xs = rng.normal(size=(200, 3))
    ys = 2.0 * xs[:, 0] - 0.5 * xs[:, 2]
    eq = fit_sparse_equation(xs, ys, ["a", "b", "c"], alpha=1e-8, threshold=1e-3)
    formula = eq.formula(lhs="y")
    assert "a" in formula and "c" in formula and "*b" not in formula

    bt = extract_neural_jets(exact_activation_field_1d("tanh"), np.linspace(-1, 1, 50), max_order=1)
    _, names = build_jet_relation_library(bt, lhs_order=1, max_degree=2)
    assert names == ["x", "y", "x^2", "x*y", "y^2"]

    assert round(rmse(np.zeros(4), np.array([0.1, -0.1, 0.2, -0.2])), 4) == 0.1581
    assert round(mae(np.zeros(4), np.array([0.1, -0.1, 0.2, -0.2])), 4) == 0.15

    xg = np.linspace(0, 4, 400)
    half = gl_fractional_derivative(np.exp(xg), alpha=0.5, h=xg[1] - xg[0])
    assert half.shape == (400,)

    bf = extract_neural_jets(exact_activation_field_1d("exp"), np.linspace(0, 2, 200), max_order=1)
    _, fnames = build_jet_fractional_features(bf, orders=(0.25, 0.5, 0.75))
    assert fnames == ["D^0.25(y)", "D^0.5(y)", "D^0.75(y)"]

    bc = extract_neural_jets(exact_activation_field_1d("tanh"), np.linspace(-2, 2, 120), max_order=2)
    locs, scales = fit_jet_cdf_plan(bc, lhs_order=2, bases=("sigmoid",), n_locations=3)
    dcdf, cnames = build_jet_cdf_features(
        bc, lhs_order=2, bases=("sigmoid",), per_variable_locations=locs, per_variable_scales=scales
    )
    assert dcdf.shape[1] == len(cnames) > 0

    data = make_symbolic_regression_dataset(seed=0)
    sur = discover_interpretable_surrogate(data)
    assert "equation" in sur

    xh = np.random.default_rng(0).uniform(-1, 1, size=(8, 2))
    dz, _ = build_hybrid_library(xh, max_degree=2, max_frequency=1)
    assert dz.shape[0] == 8
    cdf, _ = build_cdf_band_library(
        np.random.default_rng(0).uniform(-2, 2, size=(16, 1)),
        bases=("sigmoid",),
        locations=np.array([0.0]),
        scales=np.array([1.0]),
    )
    info, _ = build_information_library(
        np.random.default_rng(0).uniform(-2, 2, size=(16, 1)),
        bases=("arctan",),
        locations=np.array([0.0]),
        scales=np.array([1.0]),
    )
    assert cdf.shape[0] == 16 and info.shape[0] == 16

    _, _hidden = make_high_dim_sparse_dataset(n_features=60, seed=0)
    rep = evaluate_high_dim_sparse_validation(n_features=60, seed=0)
    assert rep["recovery_rate"] >= 0.75

    assert "u_xx" in discover_pde_operator_law(diffusivity=0.12)["equation"]

    # --- Coefficient uncertainty & model selection -------------------------- #
    from omnibias.symbolic import (
        attach_uncertainty,
        bootstrap_coefficients,
        certified_coefficient_intervals,
        equation_information_criterion,
        kfold_select,
        ridge_coefficient_covariance,
        stability_selection,
    )

    rng_u = np.random.default_rng(1)
    Xu = rng_u.normal(size=(400, 4))
    Xu[:, 3] = Xu[:, 0] * Xu[:, 1]
    yu = 2.0 * Xu[:, 0] - 1.5 * Xu[:, 2] + 0.01 * rng_u.normal(size=400)
    unames = ["x1", "x2", "x3", "x1x2"]

    boot = bootstrap_coefficients(Xu, yu, unames, n_boot=200, seed=7)
    assert boot["ci_lower"][0] <= 2.0 <= boot["ci_upper"][0]
    assert boot["ci_lower"][2] <= -1.5 <= boot["ci_upper"][2]
    assert boot["selection_frequency"][0] > 0.95
    assert boot["selection_frequency"][2] > 0.95

    cov = ridge_coefficient_covariance(Xu, yu, alpha=1e-10)
    assert abs(cov["coefficients"][0] - 2.0) < 0.05
    assert np.all(cov["std_errors"] >= 0.0)

    yu_exact = 2.0 * Xu[:, 0] - 1.5 * Xu[:, 2]
    cert = certified_coefficient_intervals(Xu[:, [0, 2]], yu_exact, alpha=0.0)
    assert cert["certified"] is True
    assert cert["radius"] < 1e-9
    lo0, hi0 = cert["intervals"][0]
    assert lo0 <= 2.0 <= hi0

    equ = fit_sparse_equation(Xu, yu, unames, threshold=1e-2)
    equ = attach_uncertainty(equ, Xu, yu, n_boot=120, seed=2)
    assert "+/-" in equ.uncertainty_formula()
    assert equ.coefficient_intervals is not None

    rng_o = np.random.default_rng(12)
    xx = rng_o.uniform(-2, 2, size=400)
    yy = 1.0 + 2.0 * xx**2 + 0.01 * rng_o.normal(size=400)
    full = np.stack([xx, xx**2, xx**3, xx**4], axis=1)
    onames = ["x", "x^2", "x^3", "x^4"]
    eq2 = fit_sparse_equation(full[:, :2], yy, onames[:2], threshold=1e-8)
    eq4 = fit_sparse_equation(full, yy, onames, threshold=1e-8)
    b2 = equation_information_criterion(eq2, full[:, :2], yy, name="bic")
    b4 = equation_information_criterion(eq4, full, yy, name="bic")
    assert b2 <= b4 + 1e-9

    sel = kfold_select(Xu, yu, unames, k=5, seed=14)
    assert sel.cv_rmse < 0.2
    ss = stability_selection(Xu, yu, unames, alpha=1e-8, threshold=1e-2, n_resample=100, seed=16)
    assert {ss["ranking"][0][0], ss["ranking"][1][0]} == {"x1", "x3"}


# ===================================================================== ch.2 ==
def test_handbook_ch2_vector_calculus_pde() -> None:
    from omnibias.symbolic import (
        FieldLawDiscoverer,
        analytic_field_jet,
        discover_field_pde_law,
        evaluate_field_pde_discovery,
        extract_field_jet,
        field_anisotropic_laplacian,
        field_curl,
        field_derivative_jet,
        field_divergence,
        field_grad_norm_sq,
        field_gradient,
        field_hessian,
        field_ito_generator,
        field_laplacian,
        field_operator_columns,
        field_value,
        field_wirtinger,
        fit_neural_field_nd,
        make_burgers_field_split,
        make_heat_field_split,
        make_wave_field_split,
    )

    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, size=(400, 2))
    u = np.sin(x[:, 0]) * np.exp(0.3 * x[:, 1])
    fnd = fit_neural_field_nd(x, u, hidden=400, seed=0, var_names=("x", "y"))
    assert fnd.dim == 2
    jet = extract_field_jet(fnd, x, max_order=2)
    assert (jet.order, jet.dim, jet.n) == (2, 2, 400)
    assert jet.value().shape == (400,) and jet.partial((1, 0)).shape == (400,)

    xv, yv = x[:, 0], x[:, 1]
    parts = {
        (0, 0): xv**2 - yv**2,
        (1, 0): 2 * xv,
        (0, 1): -2 * yv,
        (2, 0): np.full_like(xv, 2.0),
        (1, 1): np.zeros_like(xv),
        (0, 2): np.full_like(xv, -2.0),
    }
    hjet = analytic_field_jet(x, parts, order=2, var_names=("x", "y"))
    assert np.allclose(field_laplacian(hjet), 0.0)
    assert field_derivative_jet(jet, 0).order == 1
    assert np.allclose(field_value(jet), jet.value())
    assert field_gradient(jet).shape == (400, 2)
    h = field_hessian(jet)
    assert h.shape == (400, 2, 2) and np.allclose(h[:, 0, 1], h[:, 1, 0])
    assert field_grad_norm_sq(jet).shape == (400,)

    rng = np.random.default_rng(1)
    xvec = rng.uniform(-1, 1, size=(200, 2))
    fx = fit_neural_field_nd(xvec, rng.normal(size=200), hidden=64, seed=1, var_names=("x", "y"))
    fy = fit_neural_field_nd(xvec, rng.normal(size=200), hidden=64, seed=2, var_names=("x", "y"))
    comps = [extract_field_jet(fx, xvec, max_order=1), extract_field_jet(fy, xvec, max_order=1)]
    assert field_divergence(comps).shape == (200,)
    assert field_curl(comps).shape == (200,)
    assert (
        field_ito_generator(
            jet, np.array([0.1, -0.2]), np.array([[0.5, 0.0], [0.0, 0.3]])
        ).shape
        == (400,)
    )
    assert field_anisotropic_laplacian(jet, np.array([[2.0, 0.3], [0.3, 1.0]])).shape == (400,)

    xw = np.random.default_rng(0).uniform(-1, 1, size=(40, 2))
    xwc, ywc = xw[:, 0], xw[:, 1]
    uw = analytic_field_jet(xw, {(0, 0): xwc**2 - ywc**2, (1, 0): 2 * xwc, (0, 1): -2 * ywc}, order=1)
    vw = analytic_field_jet(xw, {(0, 0): 2 * xwc * ywc, (1, 0): 2 * ywc, (0, 1): 2 * xwc}, order=1)
    _, dzbar = field_wirtinger(uw, vw)
    assert np.allclose(dzbar, 0.0)

    cols = field_operator_columns(jet, include_laplacian=True)
    assert "lap(u)" in cols and "u_xx" in cols

    tr, va, te, _ = make_heat_field_split(seed=0)
    rheat = FieldLawDiscoverer(max_degree=1, time_axis=1).discover(tr, va, te, lhs_index=(0, 1))
    assert "u_xx" in rheat.formula() and rheat.test_rmse < 1e-6

    tr, va, te, _ = make_wave_field_split(seed=0)
    assert "u_xx" in discover_field_pde_law(tr, va, te, lhs_index=(0, 2), time_axis=1)["equation"]

    tr, va, te, _ = make_burgers_field_split(seed=0)
    outb = discover_field_pde_law(tr, va, te, lhs_index=(0, 1), max_degree=2, time_axis=1)
    assert "u*u_x" in outb["equation"]

    repf = evaluate_field_pde_discovery(seed=0)
    assert "heat" in repf and "wave" in repf


# ===================================================================== ch.3 ==
def test_handbook_ch3_differential_geometry() -> None:
    from omnibias.symbolic import (
        analytic_field_jet,
        christoffel_symbols,
        discover_geometric_heat_law,
        evaluate_geometric_discovery,
        flat_metric_field,
        gaussian_curvature_2d,
        laplace_beltrami,
        make_geometric_heat_split,
        metric_determinant,
        metric_inverse,
        pullback_metric_field,
        scalar_curvature,
        warped_product_metric_field,
    )

    xf = np.random.default_rng(0).uniform(-1, 1, size=(20, 2))
    assert np.allclose(scalar_curvature(flat_metric_field(xf)), 0.0)

    theta = np.linspace(0.4, np.pi - 0.4, 40)
    xs2 = np.column_stack([theta, np.zeros_like(theta)])
    sphere = warped_product_metric_field(
        xs2, np.sin(theta), np.cos(theta), -np.sin(theta), var_names=("theta", "phi")
    )
    assert np.allclose(scalar_curvature(sphere), 2.0)
    assert metric_inverse(sphere).shape == (40, 2, 2)
    assert np.allclose(metric_determinant(sphere), np.sin(theta) ** 2)
    g_sym = christoffel_symbols(sphere)
    assert np.allclose(g_sym[:, 0, 1, 1], -np.sin(theta) * np.cos(theta))

    up = {
        (0, 0): np.cos(theta),
        (1, 0): -np.sin(theta),
        (0, 1): np.zeros_like(theta),
        (2, 0): -np.cos(theta),
        (1, 1): np.zeros_like(theta),
        (0, 2): np.zeros_like(theta),
    }
    ujet = analytic_field_jet(xs2, up, order=2, var_names=("theta", "phi"))
    assert np.allclose(laplace_beltrami(ujet, sphere), -2.0 * np.cos(theta))

    xh = np.linspace(-0.5, 0.5, 30)
    xh3 = np.column_stack([xh, np.zeros_like(xh)])
    hyp = warped_product_metric_field(xh3, np.exp(xh), np.exp(xh), np.exp(xh), var_names=("x", "y"))
    assert np.allclose(gaussian_curvature_2d(hyp), -1.0)
    assert np.allclose(scalar_curvature(hyp), -2.0)

    rng = np.random.default_rng(0)
    tp = np.column_stack([rng.uniform(0.5, np.pi - 0.5, 60), rng.uniform(0, 2 * np.pi, 60)])
    th, ph = tp[:, 0], tp[:, 1]

    def comp(val, dth, dph, dthth, dthph, dphph):  # noqa: ANN001, ANN202
        return analytic_field_jet(
            tp,
            {(0, 0): val, (1, 0): dth, (0, 1): dph, (2, 0): dthth, (1, 1): dthph, (0, 2): dphph},
            order=2,
            var_names=("theta", "phi"),
        )

    phi = [
        comp(
            np.sin(th) * np.cos(ph),
            np.cos(th) * np.cos(ph),
            -np.sin(th) * np.sin(ph),
            -np.sin(th) * np.cos(ph),
            -np.cos(th) * np.sin(ph),
            -np.sin(th) * np.cos(ph),
        ),
        comp(
            np.sin(th) * np.sin(ph),
            np.cos(th) * np.sin(ph),
            np.sin(th) * np.cos(ph),
            -np.sin(th) * np.sin(ph),
            np.cos(th) * np.cos(ph),
            -np.sin(th) * np.sin(ph),
        ),
        comp(np.cos(th), -np.sin(th), np.zeros_like(th), -np.cos(th), np.zeros_like(th), np.zeros_like(th)),
    ]
    g = pullback_metric_field(phi, var_names=("theta", "phi"))
    assert np.allclose(g.g[:, 0, 0], 1.0) and np.allclose(g.g[:, 1, 1], np.sin(th) ** 2)

    tr, va, te, metrics, _ = make_geometric_heat_split(seed=0)
    outg = discover_geometric_heat_law(tr, va, te, metrics)
    assert "lap_g(u)" in outg["equation"]
    assert "lap_g(u)" in evaluate_geometric_discovery(seed=0)["geometric_heat"]["equation"]


# ===================================================================== ch.4 ==
def test_handbook_ch4_exterior_calculus() -> None:
    from omnibias.symbolic import (
        closedness_residual,
        codifferential,
        electromagnetic_field_2form,
        evaluate_exterior_calculus,
        exterior_derivative,
        extract_field_jet,
        field_laplacian,
        fit_neural_field_nd,
        gradient_form,
        hodge_laplacian,
        hodge_star,
        one_form,
        scalar_form,
        wedge,
    )

    rng = np.random.default_rng(0)
    xe = rng.uniform(-0.6, 0.6, size=(48, 3))
    enames = ("x", "y", "z")
    fjet = extract_field_jet(
        fit_neural_field_nd(xe, rng.normal(size=48), hidden=32, seed=0, var_names=enames),
        xe,
        max_order=3,
    )
    gjets = [
        extract_field_jet(
            fit_neural_field_nd(xe, rng.normal(size=48), hidden=24, seed=10 + i, var_names=enames),
            xe,
            max_order=3,
        )
        for i in range(3)
    ]
    f0 = scalar_form(fjet)
    omega = one_form(gjets)
    assert f0.degree == 0 and omega.degree == 1
    df = exterior_derivative(f0)
    assert df.degree == 1 and df.order == fjet.order - 1
    back = hodge_star(hodge_star(omega))
    assert np.allclose(back.value((0,)), omega.value((0,)))
    assert codifferential(omega).degree == 0
    assert np.allclose(hodge_laplacian(f0).value(()), -field_laplacian(fjet))
    assert wedge(df, df).max_abs() < 1e-12
    f_2form = electromagnetic_field_2form(omega)
    assert closedness_residual(f_2form) < 1e-10
    assert closedness_residual(gradient_form(fjet)) < 1e-10
    exrep = evaluate_exterior_calculus(seed=0)
    assert all(v < 1e-8 for v in exrep.values())


# ===================================================================== ch.5 ==
def test_handbook_ch5_information_theory() -> None:
    import jax.numpy as jnp
    from omnibias.core.verified.information import (
        binned_distribution_enclosure,
        entropy_enclosure,
        kl_divergence_enclosure,
    )
    from omnibias.jax.information import (
        cross_entropy,
        entropy,
        f_divergence,
        hellinger_distance,
        kl_divergence,
        mutual_information,
        renyi_divergence,
        total_variation_distance,
    )
    from omnibias.symbolic import discover_interpretable_surrogate, make_symbolic_regression_dataset
    from omnibias.symbolic.diagnostics import (
        differential_entropy,
        feature_residual_mutual_information,
        gaussian_entropy,
        kl_to_gaussian,
        surrogate_residual_diagnostics,
    )
    from omnibias.symbolic.diagnostics import entropy as np_entropy
    from omnibias.symbolic.diagnostics import kl_divergence as np_kl

    assert abs(float(entropy(jnp.array([0.5, 0.5]))) - np.log(2)) < 1e-9
    assert abs(float(entropy(jnp.array([1.0, 0.0])))) < 1e-12
    p = jnp.array([0.2, 0.3, 0.5])
    q = jnp.array([0.1, 0.4, 0.5])
    assert round(float(kl_divergence(p, q)), 5) == 0.05232
    assert float(cross_entropy(p, p)) > 0
    joint = jnp.array([[0.5, 0.0], [0.0, 0.5]])
    assert round(float(mutual_information(joint)), 5) == 0.69315
    assert round(float(total_variation_distance(p, q)), 4) == 0.1
    assert float(hellinger_distance(p, q)) > 0
    assert np.isfinite(float(renyi_divergence(p, q, 0.5)))
    assert abs(float(f_divergence(p, q, lambda t: t * jnp.log(t))) - 0.05232) < 1e-4

    assert round(float(np_entropy(np.array([0.5, 0.5]))), 5) == 0.69315
    assert float(np_kl(np.array([0.2, 0.8]), np.array([0.5, 0.5]))) > 0
    s = np.random.default_rng(0).normal(0, 2.0, size=20000)
    assert abs(differential_entropy(s, bins=64) - gaussian_entropy(2.0)) < 0.1
    white = np.random.default_rng(0).normal(size=5000)
    skewed = np.random.default_rng(0).exponential(size=5000)
    assert kl_to_gaussian(white) < kl_to_gaussian(skewed)
    xf = np.random.default_rng(0).uniform(-2, 2, size=4000)
    mi_indep = feature_residual_mutual_information(xf, np.random.default_rng(1).normal(size=4000))
    mi_dep = feature_residual_mutual_information(xf, np.sin(3 * xf))
    assert mi_indep < mi_dep
    rngd = np.random.default_rng(0)
    xd = rngd.uniform(-1, 1, size=(500, 2))
    yd = xd[:, 0] ** 2
    repd = surrogate_residual_diagnostics(xd, yd, yd + rngd.normal(0, 1e-3, size=500))
    assert "rmse" in repd and "max_feature_residual_mi" in repd
    data = make_symbolic_regression_dataset(seed=0)
    outdiv = discover_interpretable_surrogate(data, divergence_objective="kl_gaussian", divergence_weight=0.5)
    assert "equation" in outdiv

    he = entropy_enclosure([0.5, 0.5])
    assert he.lo <= np.log(2) <= he.hi
    kle = kl_divergence_enclosure([0.2, 0.3, 0.5], [0.1, 0.4, 0.5])
    kl_true = 0.2 * np.log(2.0) + 0.3 * np.log(0.75)
    assert kle.lo <= kl_true <= kle.hi and (kle.hi - kle.lo) < 1e-9
    masses = binned_distribution_enclosure("sigmoid", [-3.0, -1.0, 0.0, 1.0, 3.0], loc=0.0, scale=1.0)
    h_enc = entropy_enclosure(masses)
    assert h_enc.lo <= h_enc.hi and len(masses) == 4


# ===================================================================== ch.6 ==
def test_handbook_ch6_optimal_transport() -> None:
    import jax.numpy as jnp
    from omnibias.core.verified.transport import (
        certified_wasserstein1,
        certified_wasserstein1_samples,
        certified_wasserstein2_gaussian,
    )
    from omnibias.jax.information import (
        sinkhorn_distance,
        sliced_wasserstein,
        wasserstein1,
        wasserstein1_cdf,
        wasserstein2_gaussian,
        wassersteinp,
    )
    from omnibias.symbolic.diagnostics import wasserstein_to_gaussian

    u = jnp.array([0.0, 1.0, 2.0])
    v = jnp.array([0.5, 1.5, 2.5])
    assert abs(float(wasserstein1(u, v)) - 0.5) < 1e-9
    assert abs(float(wassersteinp(u, v, p=2.0)) - 0.5) < 1e-9
    assert np.isfinite(float(wasserstein1_cdf("sigmoid", jnp.array([-1.2, -0.3, 0.1, 0.8, 1.5]))))
    assert abs(float(wasserstein2_gaussian(0.0, 1.0, 1.0, 2.0)) - np.sqrt(2)) < 1e-9

    rng = np.random.default_rng(0)
    xc = jnp.asarray(rng.normal(size=(64, 3)))
    yc = jnp.asarray(rng.normal(size=(64, 3)) + 1.0)
    dirs = jnp.asarray(rng.normal(size=(16, 3)))
    dirs = dirs / jnp.linalg.norm(dirs, axis=1, keepdims=True)
    assert float(sliced_wasserstein(xc, yc, dirs)) > 0
    assert (
        float(
            sinkhorn_distance(
                jnp.array([0.5, 0.5]),
                jnp.array([0.5, 0.5]),
                jnp.array([[0.0, 1.0], [1.0, 0.0]]),
                epsilon=0.05,
            )
        )
        < 0.2
    )

    iv1 = certified_wasserstein1_samples([0.0, 1.0, 2.0], [0.5, 1.5, 2.5])
    assert iv1.lo <= 0.5 <= iv1.hi
    ivc = certified_wasserstein1("sigmoid", [-1.2, -0.3, 0.1, 0.8, 1.5], loc=0.0, scale=1.0)
    assert ivc.lo <= ivc.hi
    ivg = certified_wasserstein2_gaussian(0.0, 1.0, 1.0, 2.0)
    assert ivg.lo <= np.sqrt(2) <= ivg.hi
    assert wasserstein_to_gaussian(np.random.default_rng(0).normal(size=4000)) < 0.2


# ===================================================================== ch.7 ==
def test_handbook_ch7_information_geometry() -> None:
    import jax.numpy as jnp
    from omnibias.jax.information import (
        exponential_family_cumulants,
        fisher_information,
        fit_natural_parameter,
        glm_mean,
        moment_match,
    )

    natural_gradient = pytest.importorskip("omnibias.curvature.natural_gradient")
    damped_solve = natural_gradient.damped_solve
    glm_loss_gradient = natural_gradient.glm_loss_gradient
    glm_natural_gradient_step = natural_gradient.glm_natural_gradient_step
    natural_gradient_step = natural_gradient.natural_gradient_step

    cum = [round(float(c), 4) for c in exponential_family_cumulants(0.0, order=3)]
    assert cum == [0.6931, 0.5, 0.25, 0.0]
    theta = jnp.array([-1.0, 0.0, 1.0])
    assert abs(float(glm_mean(theta)[1]) - 0.5) < 1e-9
    assert abs(float(fisher_information(theta)[1]) - 0.25) < 1e-9
    th_hat = moment_match(jnp.array(0.6))
    assert abs(float(glm_mean(th_hat)) - 0.6) < 1e-6
    assert abs(float(fit_natural_parameter(jnp.array([0.0, 1.0, 1.0, 0.0, 1.0]))) - float(th_hat)) < 1e-6

    f_mat = jnp.array([[4.0, 1.0], [1.0, 3.0]])
    gvec = jnp.array([1.0, 2.0])
    assert damped_solve(f_mat, gvec, damping=1e-6).shape == (2,)
    assert natural_gradient_step(jnp.array([0.0, 0.0]), gvec, f_mat, learning_rate=0.5).shape == (2,)

    rng = np.random.default_rng(0)
    n_batch, n_hidden, n_dim = 32, 4, 3
    x = jnp.asarray(rng.normal(size=(n_batch, n_dim)))
    y = jnp.asarray((rng.uniform(size=n_batch) < 0.5).astype(float))
    w = jnp.asarray(rng.normal(size=(n_hidden, n_dim)) * 0.3)
    beta = jnp.asarray(rng.normal(size=n_hidden) * 0.1)
    c = jnp.asarray(rng.normal(size=n_hidden) * 0.3)
    bsc = jnp.asarray(0.0)
    grad = glm_loss_gradient(x, y, w, beta, c, bsc, family="bernoulli")
    assert int(grad.shape[0]) == 1 + 2 * n_hidden + n_hidden * n_dim
    b1, c1, beta1, w1 = glm_natural_gradient_step(x, y, w, beta, c, bsc, family="bernoulli", damping=1e-2)
    assert w1.shape == (n_hidden, n_dim) and c1.shape == (n_hidden,)
    assert beta1.shape == (n_hidden,) and b1.shape == ()
