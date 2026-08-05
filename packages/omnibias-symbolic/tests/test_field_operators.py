# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Multivariate closed-form field jets and the vector-calculus operator surface.

The headline guarantee is that :func:`extract_field_jet` reads off *exact* mixed
partials of a random-feature field (one activation-tower evaluation per order),
verified here against JAX autodiff to machine precision. The gradient / divergence
/ curl / Laplacian / Hessian / Ito / anisotropic-Laplacian operators are then
checked against analytic vector fields with known answers, and the structural
calculus identities ``curl(grad phi) = 0`` and ``div(curl F) = 0`` are confirmed.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from omnibias.symbolic.field_discovery import (  # noqa: E402
    FieldJet,
    analytic_field_jet,
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
    field_partial_name,
    field_value,
    field_wirtinger,
    fit_neural_field_nd,
)


def _jax():
    import jax

    jax.config.update("jax_enable_x64", True)
    return jax


# ----- exact closed-form partials vs autodiff -------------------------------


def _field_callable(field_nd):
    import jax.numpy as jnp
    from omnibias.jax import get_activation

    spec = get_activation(field_nd.activation)

    def u(xrow):
        xs = (xrow - field_nd.x_mean) / field_nd.x_scale
        z = field_nd.W @ xs + field_nd.beta
        return jnp.sum(field_nd.c * spec.forward(z)) + field_nd.b

    return u


@pytest.mark.parametrize("dim", [2, 3])
def test_extract_field_jet_matches_autodiff(dim: int) -> None:
    jax = _jax()
    import jax.numpy as jnp

    rng = np.random.default_rng(dim)
    X = rng.uniform(-1.2, 1.2, size=(50, dim))
    y = rng.uniform(-1.0, 1.0, size=50)
    fld = fit_neural_field_nd(X, y, hidden=80, activation="tanh", seed=dim + 1)
    jet = extract_field_jet(fld, X, max_order=2)

    u = _field_callable(fld)
    val_ad = jax.vmap(u)(jnp.asarray(X))
    grad_ad = jax.vmap(jax.grad(u))(jnp.asarray(X))
    hess_ad = jax.vmap(jax.hessian(u))(jnp.asarray(X))

    assert np.allclose(field_value(jet), np.asarray(val_ad), atol=1e-12)
    assert np.allclose(field_gradient(jet), np.asarray(grad_ad), atol=1e-11)
    assert np.allclose(field_hessian(jet), np.asarray(hess_ad), atol=1e-10)
    lap = np.asarray(np.trace(np.asarray(hess_ad), axis1=1, axis2=2))
    assert np.allclose(field_laplacian(jet), lap, atol=1e-10)


def test_extract_field_jet_third_order_matches_autodiff() -> None:
    jax = _jax()
    import jax.numpy as jnp

    rng = np.random.default_rng(11)
    X = rng.uniform(-1.0, 1.0, size=(30, 2))
    fld = fit_neural_field_nd(X, rng.uniform(-1, 1, 30), hidden=64, seed=3)
    jet = extract_field_jet(fld, X, max_order=3)
    u = _field_callable(fld)

    # build the (3, 0) partial by differentiating along axis 0 three times
    def g0_0(x):
        return jax.grad(u)(x)[0]

    def g00(x):
        return jax.grad(g0_0)(x)[0]

    def g000(x):
        return jax.grad(g00)(x)[0]

    d300 = jax.vmap(g000)(jnp.asarray(X))
    assert np.allclose(jet.partial((3, 0)), np.asarray(d300), atol=1e-9)


# ----- analytic operator correctness (known vector fields) ------------------


def test_divergence_and_curl_3d_rigid_rotation() -> None:
    # F = (-y, x, 0): div F = 0, curl F = (0, 0, 2).
    rng = np.random.default_rng(0)
    X = rng.uniform(-2.0, 2.0, size=(40, 3))
    x, y = X[:, 0], X[:, 1]
    z0 = np.zeros(40)
    one = np.ones(40)
    names = ("x", "y", "z")
    f0 = analytic_field_jet(
        X,
        {(0, 0, 0): -y, (1, 0, 0): z0, (0, 1, 0): -one, (0, 0, 1): z0},
        order=1,
        var_names=names,
    )
    f1 = analytic_field_jet(
        X,
        {(0, 0, 0): x, (1, 0, 0): one, (0, 1, 0): z0, (0, 0, 1): z0},
        order=1,
        var_names=names,
    )
    f2 = analytic_field_jet(
        X,
        {(0, 0, 0): z0, (1, 0, 0): z0, (0, 1, 0): z0, (0, 0, 1): z0},
        order=1,
        var_names=names,
    )
    assert np.allclose(field_divergence([f0, f1, f2]), 0.0)
    curl = field_curl([f0, f1, f2])
    assert curl.shape == (40, 3)
    assert np.allclose(curl[:, 0], 0.0)
    assert np.allclose(curl[:, 1], 0.0)
    assert np.allclose(curl[:, 2], 2.0)


def test_curl_2d_scalar_vorticity() -> None:
    rng = np.random.default_rng(1)
    X = rng.uniform(-1.0, 1.0, size=(25, 2))
    x, y = X[:, 0], X[:, 1]
    one = np.ones(25)
    zero = np.zeros(25)
    names = ("x", "y")
    f0 = analytic_field_jet(X, {(0, 0): -y, (1, 0): zero, (0, 1): -one}, order=1, var_names=names)
    f1 = analytic_field_jet(X, {(0, 0): x, (1, 0): one, (0, 1): zero}, order=1, var_names=names)
    assert np.allclose(field_curl([f0, f1]), 2.0)


def test_curl_of_gradient_is_zero() -> None:
    # Structural identity: a scalar potential has symmetric mixed partials.
    rng = np.random.default_rng(2)
    X = rng.uniform(-1.0, 1.0, size=(30, 3))
    fld = fit_neural_field_nd(X, rng.uniform(-1, 1, 30), hidden=48, seed=4)
    phi = extract_field_jet(fld, X, max_order=2)
    grad_components = [field_derivative_jet(phi, i) for i in range(3)]
    curl = field_curl(grad_components)
    assert np.allclose(curl, 0.0, atol=1e-12)


def test_divergence_of_curl_is_zero() -> None:
    rng = np.random.default_rng(3)
    X = rng.uniform(-1.0, 1.0, size=(30, 3))
    comps = [
        extract_field_jet(fit_neural_field_nd(X, rng.uniform(-1, 1, 30), hidden=40, seed=s), X, max_order=3)
        for s in (5, 6, 7)
    ]
    # curl components as order-2 derivative-field combinations
    curl0 = _sub(field_derivative_jet(comps[2], 1), field_derivative_jet(comps[1], 2))
    curl1 = _sub(field_derivative_jet(comps[0], 2), field_derivative_jet(comps[2], 0))
    curl2 = _sub(field_derivative_jet(comps[1], 0), field_derivative_jet(comps[0], 1))
    assert np.allclose(field_divergence([curl0, curl1, curl2]), 0.0, atol=1e-11)


def _sub(a: FieldJet, b: FieldJet) -> FieldJet:
    from omnibias.core.multi_index import multi_indices

    order = min(a.order, b.order)
    partials = {alpha: a.partials[alpha] - b.partials[alpha] for alpha in multi_indices(a.dim, order)}
    return FieldJet(X=a.X, order=order, partials=partials, var_names=a.var_names)


def test_field_derivative_jet_shifts_partials() -> None:
    rng = np.random.default_rng(8)
    X = rng.uniform(-1.0, 1.0, size=(20, 2))
    fld = fit_neural_field_nd(X, rng.uniform(-1, 1, 20), hidden=32, seed=9)
    jet = extract_field_jet(fld, X, max_order=3)
    dx = field_derivative_jet(jet, 0)
    assert dx.order == 2
    assert np.array_equal(dx.partial((0, 0)), jet.partial((1, 0)))
    assert np.array_equal(dx.partial((1, 0)), jet.partial((2, 0)))
    assert np.array_equal(dx.partial((0, 1)), jet.partial((1, 1)))


def test_grad_norm_sq_and_axes_subset() -> None:
    rng = np.random.default_rng(4)
    X = rng.uniform(-1.0, 1.0, size=(30, 3))
    fld = fit_neural_field_nd(X, rng.uniform(-1, 1, 30), hidden=48, seed=5)
    jet = extract_field_jet(fld, X, max_order=2)
    g = field_gradient(jet)
    assert np.allclose(field_grad_norm_sq(jet), np.sum(g * g, axis=1))
    # spatial-only Laplacian over axes (0, 1) equals u_xx + u_yy
    lap01 = field_laplacian(jet, axes=(0, 1))
    assert np.allclose(lap01, jet.partial((2, 0, 0)) + jet.partial((0, 2, 0)))


def test_ito_generator_matches_definition() -> None:
    rng = np.random.default_rng(6)
    X = rng.uniform(-1.0, 1.0, size=(35, 2))
    fld = fit_neural_field_nd(X, rng.uniform(-1, 1, 35), hidden=48, seed=6)
    jet = extract_field_jet(fld, X, max_order=2)
    drift = np.array([0.4, -0.25])
    cov = np.array([[1.5, 0.3], [0.3, 0.8]])
    got = field_ito_generator(jet, drift, cov)
    expected = field_gradient(jet) @ drift + 0.5 * np.einsum(
        "ij,nij->n", cov, field_hessian(jet)
    )
    assert np.allclose(got, expected)


def test_wirtinger_detects_holomorphy_cauchy_riemann() -> None:
    # f(z) = z^2 = (x^2 - y^2) + i(2xy) is holomorphic: d_zbar f = 0, d_z f = 2z.
    rng = np.random.default_rng(12)
    X = rng.uniform(-1.5, 1.5, size=(40, 2))
    x, y = X[:, 0], X[:, 1]
    names = ("x", "y")
    u = analytic_field_jet(
        X,
        {(0, 0): x**2 - y**2, (1, 0): 2 * x, (0, 1): -2 * y},
        order=1,
        var_names=names,
    )
    v = analytic_field_jet(
        X,
        {(0, 0): 2 * x * y, (1, 0): 2 * y, (0, 1): 2 * x},
        order=1,
        var_names=names,
    )
    d_z, d_zbar = field_wirtinger(u, v)
    assert np.allclose(d_zbar, 0.0, atol=1e-12)
    assert np.allclose(d_z, 2.0 * (x + 1j * y), atol=1e-12)


def test_wirtinger_nonholomorphic_has_nonzero_dbar() -> None:
    # f = conj(z) = x - i y is anti-holomorphic: d_zbar f = 1, d_z f = 0.
    rng = np.random.default_rng(13)
    X = rng.uniform(-1.0, 1.0, size=(20, 2))
    x = X[:, 0]
    names = ("x", "y")
    one, zero = np.ones(20), np.zeros(20)
    u = analytic_field_jet(X, {(0, 0): x, (1, 0): one, (0, 1): zero}, order=1, var_names=names)
    v = analytic_field_jet(X, {(0, 0): -X[:, 1], (1, 0): zero, (0, 1): -one}, order=1, var_names=names)
    d_z, d_zbar = field_wirtinger(u, v)
    assert np.allclose(d_zbar, 1.0, atol=1e-12)
    assert np.allclose(d_z, 0.0, atol=1e-12)


def test_anisotropic_laplacian_identity_metric_is_laplacian() -> None:
    rng = np.random.default_rng(7)
    X = rng.uniform(-1.0, 1.0, size=(30, 3))
    fld = fit_neural_field_nd(X, rng.uniform(-1, 1, 30), hidden=48, seed=8)
    jet = extract_field_jet(fld, X, max_order=2)
    iso = field_anisotropic_laplacian(jet, np.eye(3))
    assert np.allclose(iso, field_laplacian(jet))


# ----- operator-column dictionary / naming ----------------------------------


def test_field_partial_name_is_readable() -> None:
    names = ("x", "t")
    assert field_partial_name((0, 0), names) == "u"
    assert field_partial_name((1, 0), names) == "u_x"
    assert field_partial_name((0, 1), names) == "u_t"
    assert field_partial_name((1, 1), names) == "u_xt"
    assert field_partial_name((2, 0), names) == "u_xx"
    assert field_partial_name((1, 0), names, lhs="p") == "p_x"


def test_field_operator_columns_contents() -> None:
    X = np.zeros((4, 2))
    z = np.zeros(4)
    jet = analytic_field_jet(
        X,
        {(0, 0): z, (1, 0): z, (0, 1): z, (2, 0): z, (1, 1): z, (0, 2): z},
        order=2,
        var_names=("x", "t"),
    )
    cols = field_operator_columns(
        jet, include_laplacian=True, include_grad_norm_sq=True, spatial_axes=(0,)
    )
    assert set(cols) == {"u", "u_x", "u_t", "u_xx", "u_xt", "u_tt", "lap(u)", "|grad u|^2"}
    plain = field_operator_columns(jet, max_partial_order=1, include_laplacian=False)
    assert set(plain) == {"u", "u_x", "u_t"}


# ----- guards ---------------------------------------------------------------


def test_extract_field_jet_rejects_negative_order() -> None:
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(5, 2))
    fld = fit_neural_field_nd(X, rng.uniform(-1, 1, 5), hidden=8, seed=0)
    with pytest.raises(ValueError):
        extract_field_jet(fld, X, max_order=-1)


def test_gradient_needs_order_one() -> None:
    X = np.zeros((3, 2))
    jet = analytic_field_jet(X, {(0, 0): np.zeros(3)}, order=0, var_names=("x", "y"))
    with pytest.raises(ValueError):
        field_gradient(jet)


def test_hessian_needs_order_two() -> None:
    X = np.zeros((3, 2))
    jet = analytic_field_jet(
        X, {(0, 0): np.zeros(3), (1, 0): np.zeros(3), (0, 1): np.zeros(3)}, order=1, var_names=("x", "y")
    )
    with pytest.raises(ValueError):
        field_hessian(jet)


def test_analytic_field_jet_missing_partial_raises() -> None:
    X = np.zeros((3, 2))
    with pytest.raises(ValueError):
        analytic_field_jet(X, {(0, 0): np.zeros(3)}, order=1, var_names=("x", "y"))


def test_curl_rejects_unsupported_shape() -> None:
    X = np.zeros((3, 2))
    jet = analytic_field_jet(
        X, {(0, 0): np.zeros(3), (1, 0): np.zeros(3), (0, 1): np.zeros(3)}, order=1, var_names=("x", "y")
    )
    with pytest.raises(ValueError):
        field_curl([jet])  # 1 component in 2-D is unsupported


def test_fit_neural_field_nd_requires_2d() -> None:
    with pytest.raises(ValueError):
        fit_neural_field_nd(np.zeros(10), np.zeros(10))
