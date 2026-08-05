# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Riemannian-geometry operator surface on the closed-form neural jet.

Field-function derivatives are exact closed form (the neural jet); the metric and
its derivatives are inputs -- analytic for the warped-product fixtures, or the
closed-form pullback ``g = J^T J`` of a learned chart. These tests pin down:

* metric algebra (inverse / determinant / Christoffel) vs autodiff of the metric;
* the Laplace--Beltrami operator vs the hand formula on the sphere and reduction
  to the flat Laplacian when ``g = I``;
* the Riemann / Ricci / scalar curvature against the *known* constant curvatures
  of the plane (``R = 0``), sphere (``R = +2``) and hyperbolic plane (``R = -2``);
* the pullback metric of a learned chart and its first/second derivatives vs JAX
  autodiff of ``J^T J``, and the end-to-end *chart -> pullback metric -> curvature*
  pipeline on an analytic sphere embedding.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from omnibias.symbolic.field_discovery import (  # noqa: E402
    analytic_field_jet,
    extract_field_jet,
    field_laplacian,
    fit_neural_field_nd,
)
from omnibias.symbolic.geometry_discovery import (  # noqa: E402
    MetricField,
    analytic_metric_field,
    christoffel_symbols,
    covariant_hessian,
    flat_metric_field,
    gaussian_curvature_2d,
    laplace_beltrami,
    metric_determinant,
    metric_grad_norm_sq,
    metric_inverse,
    pullback_metric_field,
    ricci_tensor,
    riemann_tensor,
    scalar_curvature,
    warped_product_metric_field,
)


def _jax():
    import jax

    jax.config.update("jax_enable_x64", True)
    return jax


# --------------------------------------------------------------------------- #
# Metric fixtures (warped product ds^2 = dx^2 + f(x)^2 dy^2)
# --------------------------------------------------------------------------- #
def _sphere_metric(X):
    th = X[:, 0]
    return warped_product_metric_field(
        X, f=np.sin(th), fp=np.cos(th), fpp=-np.sin(th), var_names=("theta", "phi")
    )


def _hyperbolic_metric(X):
    x = X[:, 0]
    e = np.exp(x)
    return warped_product_metric_field(X, f=e, fp=e, fpp=e, var_names=("x", "y"))


# --------------------------------------------------------------------------- #
# Metric algebra
# --------------------------------------------------------------------------- #
def test_metric_inverse_and_determinant():
    rng = np.random.default_rng(0)
    X = rng.uniform([0.4, 0.0], [2.7, 1.0], size=(32, 2))
    mf = _sphere_metric(X)
    ginv = metric_inverse(mf)
    eye = np.einsum("nij,njk->nik", mf.g, ginv)
    assert np.allclose(eye, np.broadcast_to(np.eye(2), eye.shape), atol=1e-12)
    # det diag(1, sin^2) = sin^2
    assert np.allclose(metric_determinant(mf), np.sin(X[:, 0]) ** 2, atol=1e-12)


def _autodiff_christoffel(gfun, X):
    """Christoffel symbols Gamma[n,k,i,j] from autodiff of a metric callable."""
    jax = _jax()
    import jax.numpy as jnp

    def gamma_at(x):
        g = gfun(x)
        ginv = jnp.linalg.inv(g)
        dgrad = jax.jacfwd(gfun)(x)  # dgrad[i, j, k] = d_k g_ij
        # term[l, i, j] = d_i g_lj + d_j g_li - d_l g_ij
        term = (
            jnp.einsum("lji->lij", dgrad)
            + jnp.einsum("lij->lij", dgrad)
            - jnp.einsum("ijl->lij", dgrad)
        )
        return 0.5 * jnp.einsum("kl,lij->kij", ginv, term)

    return np.asarray(jax.vmap(gamma_at)(jnp.asarray(X)))


def test_christoffel_matches_autodiff_on_random_smooth_metric():
    jax = _jax()
    import jax.numpy as jnp

    def chart(z):
        return jnp.array([jnp.sin(z[0]) * z[1], jnp.cos(z[1]), z[0] ** 2 + 0.3 * z[1]])

    def gfun(z):
        J = jax.jacfwd(chart)(z)
        return J.T @ J + jnp.eye(2)

    rng = np.random.default_rng(3)
    X = rng.uniform(-0.5, 0.5, size=(16, 2))

    # Build a MetricField from autodiff g, dg and compare christoffel einsum.
    g = np.asarray(jax.vmap(gfun)(jnp.asarray(X)))
    dgrad = np.asarray(jax.vmap(jax.jacfwd(gfun))(jnp.asarray(X)))  # (n,i,j,k)
    dg = np.transpose(dgrad, (0, 3, 1, 2))  # -> (n,k,i,j)
    mf = analytic_metric_field(X, g, dg, var_names=("u", "v"))

    gamma = christoffel_symbols(mf)
    gamma_ad = _autodiff_christoffel(gfun, X)
    assert np.allclose(gamma, gamma_ad, atol=1e-10)
    # Symmetry in lower indices Gamma^k_ij = Gamma^k_ji.
    assert np.allclose(gamma, np.transpose(gamma, (0, 1, 3, 2)), atol=1e-12)


def test_christoffel_hyperbolic_closed_form():
    rng = np.random.default_rng(1)
    X = rng.uniform([-1.0, -1.0], [1.0, 1.0], size=(20, 2))
    mf = _hyperbolic_metric(X)
    gamma = christoffel_symbols(mf)
    e2x = np.exp(2.0 * X[:, 0])
    # Non-zero symbols of g = diag(1, e^{2x}): Gamma^x_yy = -e^{2x}, Gamma^y_xy = 1.
    assert np.allclose(gamma[:, 0, 1, 1], -e2x, atol=1e-12)
    assert np.allclose(gamma[:, 1, 0, 1], 1.0, atol=1e-12)
    assert np.allclose(gamma[:, 1, 1, 0], 1.0, atol=1e-12)
    assert np.allclose(gamma[:, 0, 0, 0], 0.0, atol=1e-12)


# --------------------------------------------------------------------------- #
# Laplace-Beltrami / covariant Hessian / metric grad-norm
# --------------------------------------------------------------------------- #
def _zonal_sphere_jet(X, degree):
    """Exact (theta,phi,t) jet of a single zonal harmonic mode (phi-independent)."""
    from numpy.polynomial.legendre import Legendre

    theta = X[:, 0]
    p, sp = np.cos(theta), np.sin(theta)
    poly = Legendre.basis(degree)
    p0 = np.asarray(poly(p))
    p1 = np.asarray(poly.deriv(1)(p))
    p2 = np.asarray(poly.deriv(2)(p))
    f = p0
    f_th = -sp * p1
    f_thth = -p * p1 + sp**2 * p2
    z = np.zeros_like(theta)
    acc = {
        (0, 0, 0): f, (1, 0, 0): f_th, (0, 1, 0): z, (0, 0, 1): z,
        (2, 0, 0): f_thth, (1, 1, 0): z, (1, 0, 1): z, (0, 2, 0): z,
        (0, 1, 1): z, (0, 0, 2): z,
    }
    return analytic_field_jet(X, acc, order=2, var_names=("theta", "phi", "t"))


def test_laplace_beltrami_sphere_matches_hand_formula():
    rng = np.random.default_rng(5)
    X = rng.uniform([0.4, 0.0, 0.0], [np.pi - 0.4, 6.2, 0.3], size=(48, 3))
    jet = _zonal_sphere_jet(X, degree=3)
    mf = _sphere_metric(X[:, :2])
    lb = laplace_beltrami(jet, mf, spatial_axes=(0, 1))
    theta = X[:, 0]
    f_thth = jet.partial((2, 0, 0))
    f_th = jet.partial((1, 0, 0))
    hand = f_thth + (np.cos(theta) / np.sin(theta)) * f_th
    assert np.allclose(lb, hand, atol=1e-11)
    # Zonal harmonic eigenvalue: Delta_g P_l = -l(l+1) P_l.
    assert np.allclose(lb, -3 * 4 * jet.partial((0, 0, 0)), atol=1e-11)


def test_laplace_beltrami_flat_reduces_to_laplacian():
    rng = np.random.default_rng(6)
    X = rng.uniform(-0.6, 0.6, size=(40, 3))
    field = fit_neural_field_nd(X, rng.normal(size=40), hidden=24, seed=2,
                               var_names=("x", "y", "z"))
    jet = extract_field_jet(field, X, max_order=2)
    mf = flat_metric_field(X)  # 3-D identity metric
    lb = laplace_beltrami(jet, mf)
    assert np.allclose(lb, field_laplacian(jet), atol=1e-12)


def test_covariant_hessian_reduces_to_plain_hessian_when_flat():
    rng = np.random.default_rng(7)
    X = rng.uniform(-0.5, 0.5, size=(24, 2))
    field = fit_neural_field_nd(X, rng.normal(size=24), hidden=16, seed=4)
    jet = extract_field_jet(field, X, max_order=2)
    mf = flat_metric_field(X)
    ch = covariant_hessian(jet, mf)
    from omnibias.symbolic.field_discovery import field_hessian

    assert np.allclose(ch, field_hessian(jet), atol=1e-12)
    # Covariant Hessian is symmetric.
    assert np.allclose(ch, np.transpose(ch, (0, 2, 1)), atol=1e-12)


def test_metric_grad_norm_sq_flat_matches_euclidean():
    rng = np.random.default_rng(8)
    X = rng.uniform(-0.5, 0.5, size=(24, 2))
    field = fit_neural_field_nd(X, rng.normal(size=24), hidden=16, seed=5)
    jet = extract_field_jet(field, X, max_order=2)
    mf = flat_metric_field(X, with_second=False)
    from omnibias.symbolic.field_discovery import field_grad_norm_sq

    assert np.allclose(
        metric_grad_norm_sq(jet, mf), field_grad_norm_sq(jet), atol=1e-12
    )


def test_metric_grad_norm_sq_sphere_value():
    # |grad u|_g^2 = u_theta^2 + csc^2(theta) u_phi^2; for a zonal mode = u_theta^2.
    rng = np.random.default_rng(9)
    X = rng.uniform([0.4, 0.0, 0.0], [np.pi - 0.4, 6.2, 0.3], size=(32, 3))
    jet = _zonal_sphere_jet(X, degree=2)
    mf = _sphere_metric(X[:, :2])
    val = metric_grad_norm_sq(jet, mf, spatial_axes=(0, 1))
    assert np.allclose(val, jet.partial((1, 0, 0)) ** 2, atol=1e-11)


# --------------------------------------------------------------------------- #
# Curvature: known constant-curvature surfaces
# --------------------------------------------------------------------------- #
def test_scalar_curvature_flat_is_zero():
    rng = np.random.default_rng(10)
    X = rng.uniform(-1.0, 1.0, size=(30, 2))
    assert np.allclose(scalar_curvature(flat_metric_field(X)), 0.0, atol=1e-12)


def test_scalar_curvature_sphere_is_plus_two():
    rng = np.random.default_rng(11)
    X = rng.uniform([0.3, 0.0], [np.pi - 0.3, 2.0 * np.pi], size=(40, 2))
    mf = _sphere_metric(X)
    assert np.allclose(scalar_curvature(mf), 2.0, atol=1e-9)
    assert np.allclose(gaussian_curvature_2d(mf), 1.0, atol=1e-9)


def test_scalar_curvature_hyperbolic_is_minus_two():
    rng = np.random.default_rng(12)
    X = rng.uniform([-1.0, -1.0], [1.0, 1.0], size=(40, 2))
    mf = _hyperbolic_metric(X)
    assert np.allclose(scalar_curvature(mf), -2.0, atol=1e-9)
    assert np.allclose(gaussian_curvature_2d(mf), -1.0, atol=1e-9)


def test_riemann_symmetries_on_sphere():
    rng = np.random.default_rng(13)
    X = rng.uniform([0.4, 0.0], [np.pi - 0.4, 2.0], size=(12, 2))
    mf = _sphere_metric(X)
    riem = riemann_tensor(mf)  # R^rho_{sigma mu nu}, [n,rho,sigma,mu,nu]
    # Antisymmetry in the last pair (mu,nu).
    assert np.allclose(riem, -np.transpose(riem, (0, 1, 2, 4, 3)), atol=1e-9)
    # Ricci is symmetric.
    ric = ricci_tensor(mf)
    assert np.allclose(ric, np.transpose(ric, (0, 2, 1)), atol=1e-9)


# --------------------------------------------------------------------------- #
# Pullback metric of a learned chart: g = J^T J (+ derivatives)
# --------------------------------------------------------------------------- #
def _chart_callable(fields):
    import jax.numpy as jnp
    from omnibias.jax import get_activation

    def phi(xy):
        outs = []
        for fld in fields:
            xs = (xy - fld.x_mean) / fld.x_scale
            z = xs @ fld.W.T + fld.beta
            spec = get_activation(fld.activation)
            outs.append(spec.forward(z) @ fld.c + fld.b)
        return jnp.stack(outs)

    return phi


def test_pullback_metric_matches_autodiff_random_chart():
    jax = _jax()
    import jax.numpy as jnp

    rng = np.random.default_rng(14)
    X = rng.uniform(-0.6, 0.6, size=(18, 2))
    fields = [
        fit_neural_field_nd(X, rng.normal(size=18), hidden=12, seed=a, var_names=("u", "v"))
        for a in range(4)
    ]
    charts = [extract_field_jet(f, X, max_order=3) for f in fields]
    mf = pullback_metric_field(charts, with_curvature=True)

    phi = _chart_callable(fields)

    def gfun(xy):
        J = jax.jacfwd(phi)(xy)
        return J.T @ J

    g_ad = np.asarray(jax.vmap(gfun)(jnp.asarray(X)))
    dg_ad = np.transpose(
        np.asarray(jax.vmap(jax.jacfwd(gfun))(jnp.asarray(X))), (0, 3, 1, 2)
    )
    ddg_ad = np.transpose(
        np.asarray(jax.vmap(jax.jacfwd(jax.jacfwd(gfun)))(jnp.asarray(X))),
        (0, 4, 3, 1, 2),
    )
    assert np.allclose(mf.g, g_ad, atol=1e-9)
    assert np.allclose(mf.dg, dg_ad, atol=1e-9)
    assert mf.ddg is not None
    assert np.allclose(mf.ddg, ddg_ad, atol=1e-8)
    # Pullback metric is symmetric positive-definite.
    assert np.allclose(mf.g, np.transpose(mf.g, (0, 2, 1)), atol=1e-12)
    assert np.all(np.linalg.eigvalsh(mf.g) > 0)


def _analytic_chart_jets(phi, X, *, var_names, max_order=3):
    """Build per-component FieldJets of an analytic chart via autodiff (exact)."""
    jax = _jax()
    import jax.numpy as jnp
    from omnibias.core.multi_index import multi_indices

    m = X.shape[1]
    amb = int(np.asarray(phi(jnp.asarray(X[0]))).shape[0])
    d1 = jax.vmap(jax.jacfwd(phi))(jnp.asarray(X))
    d2 = jax.vmap(jax.jacfwd(jax.jacfwd(phi)))(jnp.asarray(X))
    d3 = jax.vmap(jax.jacfwd(jax.jacfwd(jax.jacfwd(phi))))(jnp.asarray(X))
    val = jax.vmap(phi)(jnp.asarray(X))
    tensors = {0: np.asarray(val), 1: np.asarray(d1), 2: np.asarray(d2), 3: np.asarray(d3)}

    jets = []
    for a in range(amb):
        acc = {}
        for alpha in multi_indices(m, max_order):
            k = sum(alpha)
            axes = []
            for i, ai in enumerate(alpha):
                axes.extend([i] * ai)
            arr = tensors[k][(slice(None), a, *axes)] if k else tensors[0][:, a]
            acc[alpha] = np.asarray(arr, dtype=float)
        jets.append(analytic_field_jet(X, acc, order=max_order, var_names=var_names))
    return jets


def test_pullback_of_sphere_chart_recovers_curvature():
    import jax.numpy as jnp

    rng = np.random.default_rng(15)
    X = rng.uniform([0.5, 0.0], [np.pi - 0.5, 2.0 * np.pi], size=(24, 2))

    def phi(tp):
        th, ph = tp[0], tp[1]
        return jnp.array(
            [jnp.sin(th) * jnp.cos(ph), jnp.sin(th) * jnp.sin(ph), jnp.cos(th)]
        )

    charts = _analytic_chart_jets(phi, X, var_names=("theta", "phi"))
    mf = pullback_metric_field(charts, with_curvature=True)
    # Pullback of the unit-sphere embedding is exactly g = diag(1, sin^2 theta).
    g_expected = np.zeros((X.shape[0], 2, 2))
    g_expected[:, 0, 0] = 1.0
    g_expected[:, 1, 1] = np.sin(X[:, 0]) ** 2
    assert np.allclose(mf.g, g_expected, atol=1e-9)
    # ... and therefore has constant scalar curvature R = +2.
    assert np.allclose(scalar_curvature(mf), 2.0, atol=1e-7)


# --------------------------------------------------------------------------- #
# Guards
# --------------------------------------------------------------------------- #
def test_metricfield_shape_guards():
    X = np.zeros((4, 2))
    g = np.broadcast_to(np.eye(2), (4, 2, 2)).copy()
    with pytest.raises(ValueError, match="dg must be"):
        MetricField(X=X, g=g, dg=np.zeros((4, 2, 2)), var_names=("x", "y"))
    with pytest.raises(ValueError, match="var_names"):
        MetricField(X=X, g=g, dg=np.zeros((4, 2, 2, 2)), var_names=("x",))


def test_curvature_requires_second_derivatives():
    rng = np.random.default_rng(16)
    X = rng.uniform(-1.0, 1.0, size=(8, 2))
    mf = flat_metric_field(X, with_second=False)  # ddg is None
    assert mf.ddg is None
    with pytest.raises(ValueError, match="second metric derivatives"):
        scalar_curvature(mf)


def test_laplace_beltrami_dim_guards():
    rng = np.random.default_rng(17)
    X = rng.uniform([0.4, 0.0, 0.0], [2.5, 1.0, 0.3], size=(10, 3))
    jet = _zonal_sphere_jet(X, degree=2)
    mf = _sphere_metric(X[:, :2])
    with pytest.raises(ValueError, match="spatial_axes length"):
        laplace_beltrami(jet, mf, spatial_axes=(0, 1, 2))
    with pytest.raises(ValueError, match="out of range"):
        laplace_beltrami(jet, mf, spatial_axes=(0, 9))


def test_pullback_requires_order_three_for_curvature():
    rng = np.random.default_rng(18)
    X = rng.uniform(-0.5, 0.5, size=(8, 2))
    fields = [fit_neural_field_nd(X, rng.normal(size=8), hidden=8, seed=a) for a in range(3)]
    charts = [extract_field_jet(f, X, max_order=2) for f in fields]
    with pytest.raises(ValueError, match="order >= 3"):
        pullback_metric_field(charts, with_curvature=True)
    with pytest.raises(ValueError, match="at least one"):
        pullback_metric_field([])


def test_warped_product_requires_two_dimensions():
    X = np.zeros((4, 3))
    with pytest.raises(ValueError, match="2-D"):
        warped_product_metric_field(X, f=np.ones(4), fp=np.zeros(4))
