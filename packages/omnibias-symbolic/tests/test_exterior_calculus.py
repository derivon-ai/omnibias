# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Exterior calculus / de Rham--Hodge complex on the closed-form neural jet.

The headline guarantee is the fundamental identity ``d . d = 0``, which here is
*exact* (machine precision) for any order because the mixed partials are exact:
on a 0-form it is ``curl(grad f) = 0``, on a 1-form it is ``div(curl F) = 0``, and
on the electromagnetic potential it is the homogeneous Maxwell law ``dF = 0``. We
also pin down ``delta . delta = 0``, the Hodge-star roundtrip ``** = (-1)^{k(m-k)}``,
``delta = (-1)^{m(k+1)+1} * d *``, the Hodge Laplacian as *minus* the
component-wise Laplacian on flat space, and the grad / curl / div correspondence
that shows ``d`` *is* the single operator behind all three.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from omnibias.symbolic.exterior_discovery import (  # noqa: E402
    DifferentialForm,
    closedness_residual,
    coclosedness_residual,
    codifferential,
    curl_form,
    differential_form,
    electromagnetic_field_2form,
    evaluate_exterior_calculus,
    exterior_derivative,
    gradient_form,
    hodge_laplacian,
    hodge_star,
    one_form,
    scalar_form,
    wedge,
)
from omnibias.symbolic.field_discovery import (  # noqa: E402
    analytic_field_jet,
    extract_field_jet,
    field_curl,
    field_divergence,
    field_gradient,
    field_laplacian,
    fit_neural_field_nd,
)


def _neural_scalar(X, names, *, seed, order=3, hidden=24):
    rng = np.random.default_rng(seed)
    f = fit_neural_field_nd(X, rng.normal(size=X.shape[0]), hidden=hidden, seed=seed,
                            var_names=names)
    return extract_field_jet(f, X, max_order=order)


def _neural_vector(X, names, *, seed, order=3):
    return [_neural_scalar(X, names, seed=seed + i, order=order) for i in range(len(names))]


# --------------------------------------------------------------------------- #
# d . d = 0  (the unifying identity)
# --------------------------------------------------------------------------- #
def test_d_squared_is_zero_on_scalar_is_curl_grad():
    rng = np.random.default_rng(0)
    X = rng.uniform(-0.6, 0.6, size=(48, 3))
    fjet = _neural_scalar(X, ("x", "y", "z"), seed=1)
    df = gradient_form(fjet)  # 1-form (gradient)
    ddf = exterior_derivative(df)  # 2-form == curl(grad f) == 0
    assert ddf.degree == 2
    assert ddf.max_abs() < 1e-10


def test_d_squared_is_zero_on_one_form_is_div_curl():
    rng = np.random.default_rng(2)
    X = rng.uniform(-0.6, 0.6, size=(48, 3))
    omega = one_form(_neural_vector(X, ("x", "y", "z"), seed=5))
    dd = exterior_derivative(exterior_derivative(omega))  # 3-form == div(curl) == 0
    assert dd.degree == 3
    assert dd.max_abs() < 1e-10


def test_d_squared_zero_exact_on_analytic_trig_field():
    # f = sin(x) cos(y): d(df) must vanish identically (exact partials).
    rng = np.random.default_rng(3)
    X = rng.uniform(-1.0, 1.0, size=(40, 2))
    x, y = X[:, 0], X[:, 1]
    sx, cx, sy, cy = np.sin(x), np.cos(x), np.sin(y), np.cos(y)
    partials = {
        (0, 0): sx * cy, (1, 0): cx * cy, (0, 1): -sx * sy,
        (2, 0): -sx * cy, (1, 1): -cx * sy, (0, 2): -sx * cy,
    }
    fjet = analytic_field_jet(X, partials, order=2, var_names=("x", "y"))
    ddf = exterior_derivative(gradient_form(fjet))
    assert ddf.degree == 2
    assert np.max(np.abs(ddf.value((0, 1)))) < 1e-13


# --------------------------------------------------------------------------- #
# delta . delta = 0
# --------------------------------------------------------------------------- #
def test_codifferential_squared_is_zero():
    rng = np.random.default_rng(4)
    X = rng.uniform(-0.6, 0.6, size=(40, 3))
    omega = one_form(_neural_vector(X, ("x", "y", "z"), seed=7))
    two_form = exterior_derivative(omega)  # order 2
    dd = codifferential(codifferential(two_form))  # 0-form, order 0
    assert dd.degree == 0
    assert dd.max_abs() < 1e-9


# --------------------------------------------------------------------------- #
# Hodge star: roundtrip and adjoint relation
# --------------------------------------------------------------------------- #
def test_hodge_star_roundtrip_sign():
    rng = np.random.default_rng(8)
    X = rng.uniform(-0.6, 0.6, size=(24, 3))
    names = ("x", "y", "z")
    m = 3
    f = scalar_form(_neural_scalar(X, names, seed=9, order=1))
    omega = one_form(_neural_vector(X, names, seed=11, order=1))
    two = exterior_derivative(one_form(_neural_vector(X, names, seed=13, order=2)))
    for form in (f, omega, two):
        k = form.degree
        rt = hodge_star(hodge_star(form))
        sign = (-1) ** (k * (m - k))
        for index in form.components:
            assert np.allclose(rt.value(index), sign * form.value(index), atol=1e-11)


def test_codifferential_equals_signed_star_d_star():
    rng = np.random.default_rng(14)
    X = rng.uniform(-0.6, 0.6, size=(30, 3))
    omega = one_form(_neural_vector(X, ("x", "y", "z"), seed=15, order=2))
    m, k = 3, 1
    sign = (-1) ** (m * (k + 1) + 1)
    star_d_star = hodge_star(exterior_derivative(hodge_star(omega)))
    assert np.allclose(
        codifferential(omega).value(()), sign * star_d_star.value(()), atol=1e-10
    )


# --------------------------------------------------------------------------- #
# Hodge Laplacian = -componentwise Laplacian on flat space
# --------------------------------------------------------------------------- #
def test_hodge_laplacian_scalar_is_minus_laplacian():
    rng = np.random.default_rng(16)
    X = rng.uniform(-0.6, 0.6, size=(40, 3))
    fjet = _neural_scalar(X, ("x", "y", "z"), seed=17, order=2)
    lap_form = hodge_laplacian(scalar_form(fjet))
    assert np.allclose(lap_form.value(()), -field_laplacian(fjet), atol=1e-10)


def test_hodge_laplacian_one_form_is_componentwise():
    rng = np.random.default_rng(18)
    X = rng.uniform(-0.6, 0.6, size=(36, 3))
    comps = _neural_vector(X, ("x", "y", "z"), seed=19, order=2)
    hl = hodge_laplacian(one_form(comps))
    for i in range(3):
        assert np.allclose(hl.value((i,)), -field_laplacian(comps[i]), atol=1e-9)


# --------------------------------------------------------------------------- #
# grad / curl / div correspondence -- d is the single operator behind all three
# --------------------------------------------------------------------------- #
def test_exterior_derivative_recovers_gradient():
    rng = np.random.default_rng(20)
    X = rng.uniform(-0.6, 0.6, size=(30, 3))
    fjet = _neural_scalar(X, ("x", "y", "z"), seed=21, order=2)
    grad_form = gradient_form(fjet)
    grad = field_gradient(fjet)
    for i in range(3):
        assert np.allclose(grad_form.value((i,)), grad[:, i], atol=1e-12)


def test_hodge_star_of_d_recovers_curl():
    rng = np.random.default_rng(22)
    X = rng.uniform(-0.6, 0.6, size=(30, 3))
    comps = _neural_vector(X, ("x", "y", "z"), seed=23, order=2)
    curl_vec = hodge_star(curl_form(comps))  # *(d omega) is the curl 1-form
    fc = field_curl(comps)
    for i in range(3):
        assert np.allclose(curl_vec.value((i,)), fc[:, i], atol=1e-11)


def test_codifferential_recovers_negative_divergence():
    rng = np.random.default_rng(24)
    X = rng.uniform(-0.6, 0.6, size=(30, 3))
    comps = _neural_vector(X, ("x", "y", "z"), seed=25, order=2)
    div_from_delta = codifferential(one_form(comps)).value(())
    assert np.allclose(div_from_delta, -field_divergence(comps), atol=1e-11)


def test_curl_of_rigid_rotation_is_two_analytic():
    # omega = -y dx + x dy + 0 dz; curl = (0, 0, 2) exactly.
    rng = np.random.default_rng(26)
    X = rng.uniform(-1.0, 1.0, size=(20, 3))
    x, y = X[:, 0], X[:, 1]
    z = np.zeros_like(x)
    names = ("x", "y", "z")
    w0 = analytic_field_jet(
        X, {(0, 0, 0): -y, (0, 1, 0): -np.ones_like(x), (1, 0, 0): z, (0, 0, 1): z,
            (2, 0, 0): z, (0, 2, 0): z, (0, 0, 2): z, (1, 1, 0): z, (1, 0, 1): z,
            (0, 1, 1): z},
        order=2, var_names=names,
    )
    w1 = analytic_field_jet(
        X, {(0, 0, 0): x, (1, 0, 0): np.ones_like(x), (0, 1, 0): z, (0, 0, 1): z,
            (2, 0, 0): z, (0, 2, 0): z, (0, 0, 2): z, (1, 1, 0): z, (1, 0, 1): z,
            (0, 1, 1): z},
        order=2, var_names=names,
    )
    w2 = analytic_field_jet(
        X, {a: z for a in w0.partials}, order=2, var_names=names
    )
    curl_vec = hodge_star(curl_form([w0, w1, w2]))
    assert np.allclose(curl_vec.value((0,)), 0.0, atol=1e-13)
    assert np.allclose(curl_vec.value((1,)), 0.0, atol=1e-13)
    assert np.allclose(curl_vec.value((2,)), 2.0, atol=1e-13)


# --------------------------------------------------------------------------- #
# Maxwell + closedness
# --------------------------------------------------------------------------- #
def test_homogeneous_maxwell_dF_is_zero():
    rng = np.random.default_rng(28)
    X = rng.uniform(-0.6, 0.6, size=(40, 3))
    potential = one_form(_neural_vector(X, ("x", "y", "z"), seed=29, order=3))
    field = electromagnetic_field_2form(potential)
    assert field.degree == 2
    assert closedness_residual(field) < 1e-9  # dF = d(dA) = 0


def test_gradient_field_is_closed_but_generic_one_form_is_not():
    rng = np.random.default_rng(30)
    X = rng.uniform(-0.6, 0.6, size=(40, 3))
    names = ("x", "y", "z")
    fjet = _neural_scalar(X, names, seed=31, order=2)
    assert closedness_residual(gradient_form(fjet)) < 1e-10
    # A generic (non-gradient) 1-form has a non-trivial curl.
    generic = one_form(_neural_vector(X, names, seed=33, order=2))
    assert closedness_residual(generic) > 1e-3


# --------------------------------------------------------------------------- #
# Wedge algebra
# --------------------------------------------------------------------------- #
def test_wedge_graded_commutativity_and_nilpotency():
    rng = np.random.default_rng(34)
    X = rng.uniform(-0.6, 0.6, size=(24, 3))
    names = ("x", "y", "z")
    a = one_form(_neural_vector(X, names, seed=35, order=1))
    b = one_form(_neural_vector(X, names, seed=38, order=1))
    ab = wedge(a, b)
    ba = wedge(b, a)
    # 1-forms: a ^ b = -(b ^ a).
    for index in ab.components:
        assert np.allclose(ab.value(index), -ba.value(index), atol=1e-12)
    # alpha ^ alpha = 0 for an odd-degree form.
    assert wedge(a, a).max_abs() < 1e-12


# --------------------------------------------------------------------------- #
# Certification report
# --------------------------------------------------------------------------- #
def test_evaluate_exterior_calculus_all_identities_hold():
    report = evaluate_exterior_calculus(seed=1)
    assert set(report) == {
        "dd_scalar_curl_grad",
        "dd_oneform_div_curl",
        "delta_squared",
        "star_roundtrip",
        "hodge_laplacian_vs_laplacian",
    }
    for key, residual in report.items():
        assert residual < 1e-9, f"{key} = {residual}"


# --------------------------------------------------------------------------- #
# Guards
# --------------------------------------------------------------------------- #
def test_exterior_derivative_of_top_form_raises():
    rng = np.random.default_rng(40)
    X = rng.uniform(-0.6, 0.6, size=(8, 2))
    top = exterior_derivative(one_form(_neural_vector(X, ("x", "y"), seed=41, order=2)))
    assert top.degree == 2  # top-degree 2-form in 2-D
    with pytest.raises(ValueError, match="top-degree"):
        exterior_derivative(top)


def test_codifferential_of_scalar_raises():
    rng = np.random.default_rng(42)
    X = rng.uniform(-0.6, 0.6, size=(8, 3))
    with pytest.raises(ValueError, match="0-form"):
        codifferential(scalar_form(_neural_scalar(X, ("x", "y", "z"), seed=43, order=2)))


def test_hodge_laplacian_requires_order_two():
    rng = np.random.default_rng(44)
    X = rng.uniform(-0.6, 0.6, size=(8, 3))
    f1 = scalar_form(_neural_scalar(X, ("x", "y", "z"), seed=45, order=1))
    with pytest.raises(ValueError, match="order >= 2"):
        hodge_laplacian(f1)


def test_one_form_component_count_guard():
    rng = np.random.default_rng(46)
    X = rng.uniform(-0.6, 0.6, size=(8, 3))
    comps = _neural_vector(X, ("x", "y", "z"), seed=47, order=1)
    with pytest.raises(ValueError, match="needs 3 components"):
        one_form(comps[:2])


def test_differential_form_validation_rejects_missing_components():
    rng = np.random.default_rng(48)
    X = rng.uniform(-0.6, 0.6, size=(8, 3))
    comps = _neural_vector(X, ("x", "y", "z"), seed=49, order=1)
    with pytest.raises(ValueError, match="increasing k-indices"):
        differential_form(1, {(0,): comps[0], (1,): comps[1]})  # missing (2,)


def test_wedge_dimension_and_degree_guards():
    rng = np.random.default_rng(50)
    X3 = rng.uniform(-0.6, 0.6, size=(8, 3))
    X2 = rng.uniform(-0.6, 0.6, size=(8, 2))
    a3 = one_form(_neural_vector(X3, ("x", "y", "z"), seed=51, order=1))
    a2 = one_form(_neural_vector(X2, ("x", "y"), seed=52, order=1))
    with pytest.raises(ValueError, match="equal dimension"):
        wedge(a3, a2)
    two = exterior_derivative(one_form(_neural_vector(X2, ("x", "y"), seed=53, order=2)))
    with pytest.raises(ValueError, match="exceeds dim"):
        wedge(two, a2)  # degree 2 + 1 > dim 2


def test_coclosedness_residual_of_scalar_is_zero():
    rng = np.random.default_rng(54)
    X = rng.uniform(-0.6, 0.6, size=(8, 3))
    f = scalar_form(_neural_scalar(X, ("x", "y", "z"), seed=55, order=1))
    assert coclosedness_residual(f) == 0.0
