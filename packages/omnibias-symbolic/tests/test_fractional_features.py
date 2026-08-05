# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Fractional-calculus features: Grunwald-Letnikov derivatives in jet discovery.

The non-local / fractional twin of the CDF and surprisal jet features. The GL
operator is validated against its integer-order limits and the canonical
``omnibias.fractional`` kernel, and end-to-end the :class:`NeuralJetDiscoverer`
recovers a fractional differential law that no integer-order jet library can.
"""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.symbolic.discovery import (
    JetBundle,
    NeuralJetDiscoverer,
    _gl_weights,
    build_jet_fractional_features,
    build_jet_fractional_features_closed_form,
    discover_fractional_order_law,
    exact_activation_field_1d,
    extract_neural_jets,
    gl_fractional_derivative,
    split_x_grid,
)

# ----- GL operator limits ---------------------------------------------------


def test_alpha_zero_is_identity() -> None:
    y = np.sin(np.linspace(0.0, 3.0, 60))
    assert np.allclose(gl_fractional_derivative(y, alpha=0.0, h=0.05), y)


def test_alpha_one_is_backward_difference() -> None:
    x = np.linspace(0.0, 2.0, 80)
    h = float(x[1] - x[0])
    y = np.sin(x)
    got = gl_fractional_derivative(y, alpha=1.0, h=h)
    expected = np.concatenate([[y[0] / h], (y[1:] - y[:-1]) / h])
    assert np.allclose(got, expected)


def test_alpha_two_is_second_backward_difference() -> None:
    x = np.linspace(0.0, 2.0, 80)
    h = float(x[1] - x[0])
    y = x**2  # exact second derivative 2
    d2 = gl_fractional_derivative(y, alpha=2.0, h=h)
    assert np.allclose(d2[3:], 2.0, atol=1e-9)  # interior; skip left-edge bias


def test_gl_is_linear() -> None:
    rng = np.random.default_rng(0)
    y1 = rng.normal(size=50)
    y2 = rng.normal(size=50)
    lhs = gl_fractional_derivative(2.0 * y1 - 3.0 * y2, alpha=0.5, h=0.1)
    rhs = 2.0 * gl_fractional_derivative(y1, alpha=0.5, h=0.1) - 3.0 * gl_fractional_derivative(
        y2, alpha=0.5, h=0.1
    )
    assert np.allclose(lhs, rhs)


def test_gl_guards() -> None:
    with pytest.raises(ValueError, match="alpha must be"):
        gl_fractional_derivative(np.zeros(4), alpha=-0.5, h=0.1)
    with pytest.raises(ValueError, match="h must be"):
        gl_fractional_derivative(np.zeros(4), alpha=0.5, h=0.0)
    with pytest.raises(ValueError, match="at least one"):
        gl_fractional_derivative(np.zeros(0), alpha=0.5, h=0.1)


def test_parity_with_fractional_package() -> None:
    kernels = pytest.importorskip("omnibias.fractional._core.kernels")
    assert np.allclose(_gl_weights(0.5, 40), kernels.gl_weights(0.5, 40))
    assert np.allclose(_gl_weights(1.3, 25), kernels.gl_weights(1.3, 25))
    x = np.linspace(0.0, 1.0, 40)
    h = float(x[1] - x[0])
    y = np.cos(x)
    matrix = kernels.gl_matrix(0.7, 40, h)
    assert np.allclose(gl_fractional_derivative(y, alpha=0.7, h=h), matrix @ y)


# ----- feature builder ------------------------------------------------------


def test_feature_shapes_names_and_dy_correspondence() -> None:
    field = exact_activation_field_1d("tanh")
    bundle = extract_neural_jets(field, np.linspace(-1.0, 1.0, 200), max_order=3)
    feats, names = build_jet_fractional_features(bundle, orders=(0.5, 1.0))
    assert feats.shape == (200, 2)
    assert names == ["D^0.5(y)", "D^1(y)"]
    # the D^1 column is the backward difference of y, ~ the true first jet
    corr = np.corrcoef(feats[5:, 1], bundle.jets[5:, 1])[0, 1]
    assert corr > 0.999


def test_feature_builder_guards() -> None:
    bundle = JetBundle(x=np.linspace(0.0, 1.0, 10), jets=np.zeros((10, 2)))
    with pytest.raises(ValueError, match="at least one order"):
        build_jet_fractional_features(bundle, orders=())
    with pytest.raises(ValueError, match="source_order"):
        build_jet_fractional_features(bundle, orders=(0.5,), source_order=5)
    nonuniform = JetBundle(x=np.array([0.0, 1.0, 3.0, 4.0]), jets=np.zeros((4, 2)))
    with pytest.raises(ValueError, match="uniform"):
        build_jet_fractional_features(nonuniform, orders=(0.5,))
    decreasing = JetBundle(x=np.array([3.0, 2.0, 1.0, 0.0]), jets=np.zeros((4, 2)))
    with pytest.raises(ValueError, match="strictly increasing"):
        build_jet_fractional_features(decreasing, orders=(0.5,))


# ----- end-to-end discovery -------------------------------------------------


def _fractional_law_bundle(x: np.ndarray, alpha: float, coeff_frac: float, coeff_y: float) -> JetBundle:
    y = np.sin(2.0 * x)
    h = float(x[1] - x[0])
    g = gl_fractional_derivative(y, alpha=alpha, h=h)
    target = coeff_frac * g + coeff_y * y
    return JetBundle(x=x, jets=np.stack([y, target], axis=1))


def test_neuraljet_recovers_fractional_relaxation_law() -> None:
    tr, va, te = split_x_grid(xmin=0.0, xmax=4.0, n_train=200, n_val=150, n_test=150)
    btr = _fractional_law_bundle(tr, 0.5, 1.0, 0.3)
    bva = _fractional_law_bundle(va, 0.5, 1.0, 0.3)
    bte = _fractional_law_bundle(te, 0.5, 1.0, 0.3)

    disc = NeuralJetDiscoverer(max_library_degree=1, fractional_orders=(0.5,))
    res = disc.discover(btr, bva, bte)
    assert res.test_rmse < 1e-6
    terms = {str(t["name"]): float(t["coefficient"]) for t in res.active_terms()}
    assert terms.get("D^0.5(y)", 0.0) == pytest.approx(1.0, abs=1e-3)
    assert terms.get("y", 0.0) == pytest.approx(0.3, abs=1e-3)


def test_fractional_features_beat_polynomial_only() -> None:
    tr, va, te = split_x_grid(xmin=0.0, xmax=4.0, n_train=200, n_val=150, n_test=150)
    btr = _fractional_law_bundle(tr, 0.5, 1.0, 0.3)
    bva = _fractional_law_bundle(va, 0.5, 1.0, 0.3)
    bte = _fractional_law_bundle(te, 0.5, 1.0, 0.3)
    with_frac = NeuralJetDiscoverer(max_library_degree=1, fractional_orders=(0.5,)).discover(btr, bva, bte)
    poly_only = NeuralJetDiscoverer(max_library_degree=1).discover(btr, bva, bte)
    assert with_frac.test_rmse < poly_only.test_rmse * 1e-2


def test_fractional_skipped_when_source_equals_lhs() -> None:
    # source_order == lhs_order must not inject the LHS as its own feature
    tr, va, te = split_x_grid(xmin=0.0, xmax=4.0, n_train=120, n_val=80, n_test=80)
    btr = _fractional_law_bundle(tr, 0.5, 1.0, 0.3)
    bva = _fractional_law_bundle(va, 0.5, 1.0, 0.3)
    bte = _fractional_law_bundle(te, 0.5, 1.0, 0.3)
    disc = NeuralJetDiscoverer(
        max_library_degree=1, fractional_orders=(0.5,), fractional_source_order=1
    )
    res = disc.discover(btr, bva, bte, candidate_lhs_orders=(1,))
    names = [str(t["name"]) for t in res.active_terms()]
    assert all("D^0.5(dy)" not in n for n in names)


# ----- closed-form (analytic-class) fractional order discovery -------------


def _poly_tower(x: np.ndarray, px: list[float]) -> np.ndarray:
    """Derivative tower ``[P, P', P'', ...]`` of ``P = sum px[k] x^k`` (columns)."""
    cols = []
    for d in range(len(px)):
        col = np.zeros_like(x)
        for k in range(d, len(px)):
            fac = 1.0
            for j in range(d):
                fac *= k - j
            col = col + px[k] * fac * x ** (k - d)
        cols.append(col)
    return np.stack(cols, axis=1)


def _closed_form_law_bundle(
    x: np.ndarray, px: list[float], alpha: float, c_frac: float, c_y: float
) -> JetBundle:
    """``[P, P', ..., P^deg, target]`` with ``target = c_frac D^alpha(P) + c_y P``."""
    tower = _poly_tower(x, px)
    frac_col, _ = build_jet_fractional_features_closed_form(
        JetBundle(x=x, jets=tower), orders=(alpha,), source_order=0, kind="caputo"
    )
    target = c_frac * frac_col[:, 0] + c_y * tower[:, 0]
    return JetBundle(x=x, jets=np.concatenate([tower, target[:, None]], axis=1))


def test_closed_form_feature_matches_analytic_power_law() -> None:
    pytest.importorskip("omnibias.fractional.jax.ops.analytic")
    import math

    x = np.linspace(0.0, 2.0, 60)
    px = [0.0, 0.0, 1.0]  # P(x) = x^2
    feats, names = build_jet_fractional_features_closed_form(
        JetBundle(x=x, jets=_poly_tower(x, px)), orders=(0.5,), kind="caputo"
    )
    assert names == ["D^0.5(y)"]
    coef = math.gamma(3) / math.gamma(3 - 0.5)
    assert np.allclose(feats[:, 0], coef * x ** (2 - 0.5), rtol=1e-10, atol=1e-12)


def test_discover_fractional_order_law_recovers_order_and_coeffs() -> None:
    pytest.importorskip("omnibias.fractional.jax.ops.analytic")
    px = [1.0, 2.0, 0.5]  # tower width 3 -> jets columns 0,1,2 ; target at 3
    tr_x, va_x, te_x = split_x_grid(xmin=0.0, xmax=3.0, n_train=120, n_val=90, n_test=90)
    tr = _closed_form_law_bundle(tr_x, px, 0.5, 1.0, 0.3)
    va = _closed_form_law_bundle(va_x, px, 0.5, 1.0, 0.3)
    te = _closed_form_law_bundle(te_x, px, 0.5, 1.0, 0.3)

    res = discover_fractional_order_law(
        tr, va, te,
        candidate_orders=(0.25, 0.5, 0.75),
        fractional_source_order=0,
        kind="caputo",
        tower_width=3,
        candidate_lhs_orders=(3,),
        max_library_degree=1,
    )
    assert res.fractional_order == pytest.approx(0.5)
    assert res.test_rmse < 1e-6
    terms = {str(t["name"]): float(t["coefficient"]) for t in res.active_terms()}
    assert terms.get("D^0.5(y)", 0.0) == pytest.approx(1.0, abs=1e-3)
    assert terms.get("y", 0.0) == pytest.approx(0.3, abs=1e-3)


def test_discover_fractional_order_law_beats_wrong_orders() -> None:
    pytest.importorskip("omnibias.fractional.jax.ops.analytic")
    px = [1.0, 2.0, 0.5]
    tr_x, va_x, te_x = split_x_grid(xmin=0.0, xmax=3.0, n_train=120, n_val=90, n_test=90)
    tr = _closed_form_law_bundle(tr_x, px, 0.5, 1.0, 0.3)
    va = _closed_form_law_bundle(va_x, px, 0.5, 1.0, 0.3)
    te = _closed_form_law_bundle(te_x, px, 0.5, 1.0, 0.3)
    # Only wrong candidate orders available -> the closed-form fit cannot be exact.
    res = discover_fractional_order_law(
        tr, va, te,
        candidate_orders=(0.25, 0.75),
        tower_width=3,
        candidate_lhs_orders=(3,),
    )
    assert res.test_rmse > 1e-3  # no exact fit without alpha* = 0.5


def test_discover_fractional_order_law_guards() -> None:
    pytest.importorskip("omnibias.fractional.jax.ops.analytic")
    x = np.linspace(0.0, 2.0, 20)
    bundle = _closed_form_law_bundle(x, [1.0, 2.0, 0.5], 0.5, 1.0, 0.3)
    with pytest.raises(ValueError, match="at least one candidate order"):
        discover_fractional_order_law(bundle, bundle, bundle, candidate_orders=())
