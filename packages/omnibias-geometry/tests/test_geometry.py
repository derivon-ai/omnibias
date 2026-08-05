# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Differential-geometry validation.

Four independent checks per the validation strategy:
1. analytic / manufactured solution (the round sphere S^2),
2. symbolic cross-check with sympy (Christoffel / Ricci / scalar),
3. closed-form parity (flat metric Laplace-Beltrami == ordinary Laplacian),
4. torch vs jax cross-backend parity.

All in float64.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
import torch
from omnibias.geometry import ManifoldSpec, MetricSpec
from omnibias.geometry.jax import ops as jgeo
from omnibias.geometry.torch import ops as tgeo

R = 1.3  # sphere radius (must match conftest SPHERE_R)
THETA = np.array([0.7, 1.1, 1.9, 2.4])
PHI = np.array([0.3, 1.5, 2.2, 4.0])
COORDS = np.stack([THETA, PHI], axis=-1).astype(np.float64)


# ----------------------------------------------------------------------
# manifold builders
# ----------------------------------------------------------------------
def _torch_sphere(builders):  # type: ignore[no-untyped-def]
    g = builders["sphere_metric_factory"](torch, torch.stack)
    return ManifoldSpec("sphere_S2", 2, MetricSpec(g, dim=2, name="round_sphere"))


def _jax_sphere(builders):  # type: ignore[no-untyped-def]
    g = builders["sphere_metric_factory"](jnp, jnp.stack)
    return ManifoldSpec("sphere_S2", 2, MetricSpec(g, dim=2, name="round_sphere"))


def _torch_flat(builders):  # type: ignore[no-untyped-def]
    g = builders["flat_metric_factory"](torch, torch.stack, 2)
    return ManifoldSpec("flat_R2", 2, MetricSpec(g, dim=2, name="flat"))


def _jax_flat(builders):  # type: ignore[no-untyped-def]
    g = builders["flat_metric_factory"](jnp, jnp.stack, 2)
    return ManifoldSpec("flat_R2", 2, MetricSpec(g, dim=2, name="flat"))


def _tc():  # type: ignore[no-untyped-def]
    return torch.as_tensor(COORDS, dtype=torch.float64)


def _jc():  # type: ignore[no-untyped-def]
    return jnp.asarray(COORDS, dtype=jnp.float64)


def _np(x):  # type: ignore[no-untyped-def]
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


# ----------------------------------------------------------------------
# 1. Analytic sphere
# ----------------------------------------------------------------------
def test_sphere_christoffel_analytic(builders):  # type: ignore[no-untyped-def]
    m = _torch_sphere(builders)
    gamma = _np(tgeo.christoffel(_tc(), m))  # (B, k, i, j)
    th = THETA
    # Gamma^theta_phiphi = -sin cos ; Gamma^phi_thetaphi = Gamma^phi_phitheta = cot
    assert np.allclose(gamma[:, 0, 1, 1], -np.sin(th) * np.cos(th), atol=1e-10)
    assert np.allclose(gamma[:, 1, 0, 1], np.cos(th) / np.sin(th), atol=1e-10)
    assert np.allclose(gamma[:, 1, 1, 0], np.cos(th) / np.sin(th), atol=1e-10)
    # remaining components vanish
    assert np.allclose(gamma[:, 0, 0, 0], 0.0, atol=1e-10)
    assert np.allclose(gamma[:, 0, 0, 1], 0.0, atol=1e-10)
    assert np.allclose(gamma[:, 1, 1, 1], 0.0, atol=1e-10)


def test_sphere_scalar_curvature(builders):  # type: ignore[no-untyped-def]
    m = _torch_sphere(builders)
    sc = _np(tgeo.scalar_curvature(_tc(), m))
    assert np.allclose(sc, 2.0 / R**2, atol=1e-9)


def test_sphere_ricci_is_proportional_to_metric(builders):  # type: ignore[no-untyped-def]
    m = _torch_sphere(builders)
    c = _tc()
    ric = _np(tgeo.ricci_tensor(c, m))
    g = _np(tgeo.metric(c, m))
    # Ricci = (1/R^2) g for the round 2-sphere.
    assert np.allclose(ric, g / R**2, atol=1e-9)


# ----------------------------------------------------------------------
# 2. Symbolic (sympy) cross-check
# ----------------------------------------------------------------------
def test_sphere_curvature_matches_sympy(builders):  # type: ignore[no-untyped-def]
    sympy = pytest.importorskip("sympy")
    th, ph = sympy.symbols("theta phi", positive=True)
    R2 = R**2
    g = sympy.Matrix([[R2, 0], [0, R2 * sympy.sin(th) ** 2]])
    ginv = g.inv()
    coords = [th, ph]
    d = 2

    def christ(k, i, j):
        return sympy.Rational(1, 2) * sum(
            ginv[k, a] * (sympy.diff(g[a, j], coords[i])
                          + sympy.diff(g[a, i], coords[j])
                          - sympy.diff(g[i, j], coords[a]))
            for a in range(d)
        )

    Gamma = [[[sympy.simplify(christ(k, i, j)) for j in range(d)] for i in range(d)] for k in range(d)]

    def riem(rho, sig, mu, nu):
        term = (sympy.diff(Gamma[rho][nu][sig], coords[mu])
                - sympy.diff(Gamma[rho][mu][sig], coords[nu]))
        term += sum(Gamma[rho][mu][a] * Gamma[a][nu][sig]
                    - Gamma[rho][nu][a] * Gamma[a][mu][sig] for a in range(d))
        return sympy.simplify(term)

    Ric = sympy.Matrix(d, d, lambda s, n: sympy.simplify(sum(riem(r, s, r, n) for r in range(d))))
    scal = sympy.simplify(sum(ginv[s, n] * Ric[s, n] for s in range(d) for n in range(d)))

    scal_fn = sympy.lambdify((th, ph), scal, "numpy")
    m = _torch_sphere(builders)
    got = _np(tgeo.scalar_curvature(_tc(), m))
    exp = scal_fn(THETA, PHI) * np.ones_like(got)
    assert np.allclose(got, exp, atol=1e-9)


# ----------------------------------------------------------------------
# 2b. Symbolic cross-check on more manifolds: torus + conformal 2-metric
# ----------------------------------------------------------------------
def _sympy_curvature(g, coords):  # type: ignore[no-untyped-def]
    """Lambdified ``(scalar_fn, ricci_fn)`` from a 2x2 sympy metric matrix."""
    import sympy

    ginv = g.inv()
    d = 2

    def christ(k, i, j):  # type: ignore[no-untyped-def]
        return sympy.Rational(1, 2) * sum(
            ginv[k, a] * (sympy.diff(g[a, j], coords[i])
                          + sympy.diff(g[a, i], coords[j])
                          - sympy.diff(g[i, j], coords[a]))
            for a in range(d)
        )

    Gamma = [[[sympy.simplify(christ(k, i, j)) for j in range(d)]
              for i in range(d)] for k in range(d)]

    def riem(rho, sig, mu, nu):  # type: ignore[no-untyped-def]
        term = (sympy.diff(Gamma[rho][nu][sig], coords[mu])
                - sympy.diff(Gamma[rho][mu][sig], coords[nu]))
        term += sum(Gamma[rho][mu][a] * Gamma[a][nu][sig]
                    - Gamma[rho][nu][a] * Gamma[a][mu][sig] for a in range(d))
        return sympy.simplify(term)

    ric = sympy.Matrix(
        d, d, lambda s, n: sympy.simplify(sum(riem(r, s, r, n) for r in range(d))),
    )
    scal = sympy.simplify(sum(ginv[s, n] * ric[s, n]
                              for s in range(d) for n in range(d)))
    return sympy.lambdify(coords, scal, "numpy"), sympy.lambdify(coords, ric, "numpy")


def _torch_torus(builders):  # type: ignore[no-untyped-def]
    g = builders["torus_metric_factory"](torch, torch.stack)
    return ManifoldSpec("torus_T2", 2, MetricSpec(g, dim=2, name="torus"))


def _torch_conformal(builders):  # type: ignore[no-untyped-def]
    g = builders["conformal_metric_factory"](torch, torch.stack)
    return ManifoldSpec("conformal_2d", 2, MetricSpec(g, dim=2, name="conformal"))


def test_torus_scalar_curvature_analytic(builders):  # type: ignore[no-untyped-def]
    rmaj, rmin = builders["TORUS_R"], builders["TORUS_r"]
    got = _np(tgeo.scalar_curvature(_tc(), _torch_torus(builders)))
    # R = 2K = 2 cos(theta) / ( r (R + r cos theta) ).
    exp = 2.0 * np.cos(THETA) / (rmin * (rmaj + rmin * np.cos(THETA)))
    assert np.allclose(got, exp, atol=1e-9)


def test_torus_curvature_matches_sympy(builders):  # type: ignore[no-untyped-def]
    sympy = pytest.importorskip("sympy")
    th, ph = sympy.symbols("theta phi", real=True)
    rmaj, rmin = builders["TORUS_R"], builders["TORUS_r"]
    g = sympy.Matrix([[rmin**2, 0], [0, (rmaj + rmin * sympy.cos(th)) ** 2]])
    scalar_fn, _ = _sympy_curvature(g, (th, ph))
    got = _np(tgeo.scalar_curvature(_tc(), _torch_torus(builders)))
    assert np.allclose(got, scalar_fn(THETA, PHI) * np.ones_like(got), atol=1e-9)


def test_conformal_curvature_matches_sympy(builders):  # type: ignore[no-untyped-def]
    sympy = pytest.importorskip("sympy")
    th, ph = sympy.symbols("theta phi", real=True)
    a, b = builders["CONF_A"], builders["CONF_B"]
    e2 = sympy.exp(2 * (a * sympy.sin(th) + b * sympy.cos(ph)))
    g = sympy.Matrix([[e2, 0], [0, e2]])
    scalar_fn, ricci_fn = _sympy_curvature(g, (th, ph))
    m = _torch_conformal(builders)
    c = _tc()
    got = _np(tgeo.scalar_curvature(c, m))
    assert np.allclose(got, scalar_fn(THETA, PHI) * np.ones_like(got), atol=1e-9)
    # Ricci cross-check, evaluated per point (lambdified matrix wants scalars).
    ric = _np(tgeo.ricci_tensor(c, m))
    for i in range(len(THETA)):
        exp_ric = np.asarray(ricci_fn(THETA[i], PHI[i]), dtype=np.float64)
        assert np.allclose(ric[i], exp_ric, atol=1e-9)


def test_torus_conformal_cross_backend(builders):  # type: ignore[no-untyped-def]
    for name in ("torus_metric_factory", "conformal_metric_factory"):
        tm = ManifoldSpec("m", 2, MetricSpec(builders[name](torch, torch.stack), dim=2))
        jm = ManifoldSpec("m", 2, MetricSpec(builders[name](jnp, jnp.stack), dim=2))
        assert np.allclose(_np(tgeo.scalar_curvature(_tc(), tm)),
                           _np(jgeo.scalar_curvature(_jc(), jm)),
                           rtol=1e-9, atol=1e-9)


# ----------------------------------------------------------------------
# 3. Flat metric: Laplace-Beltrami == ordinary Laplacian
# ----------------------------------------------------------------------
def _poly_field(builders, ops_module):  # type: ignore[no-untyped-def]
    P = builders["Poly1D"]
    comp_axes = {"f": (P((0.3, 1.0, 0.5, -0.2)), P((1.0, -0.4, 0.7)))}
    return builders["AnalyticField"](
        builders["coord_spec_2d"](), builders["ComponentSpec"](("f",)),
        comp_axes, ops_module,
    )


def test_flat_laplace_beltrami_equals_laplacian(builders):  # type: ignore[no-untyped-def]
    from omnibias.fields.torch import _ops_dispatch
    from omnibias.fields.torch.ops.basic import laplacian

    field = _poly_field(builders, _ops_dispatch)
    state = field(_tc())
    m = _torch_flat(builders)
    lb = _np(tgeo.laplace_beltrami(state, "f", m))
    lap = _np(laplacian(state, "f"))
    assert np.allclose(lb, lap, rtol=1e-11, atol=1e-11)


# ----------------------------------------------------------------------
#    Sphere Laplace-Beltrami eigenfunction: Delta_g cos(theta) = -2/R^2 cos(theta)
# ----------------------------------------------------------------------
def _cos_theta_field(builders, ops_module, xp):  # type: ignore[no-untyped-def]
    comp_axes = {"f": (builders["Cos1D"](1.0, 1.0, xp=xp), builders["Const1D"](1.0))}
    return builders["AnalyticField"](
        builders["coord_spec_2d"](), builders["ComponentSpec"](("f",)),
        comp_axes, ops_module,
    )


def test_sphere_laplace_beltrami_eigenfunction(builders):  # type: ignore[no-untyped-def]
    from omnibias.fields.torch import _ops_dispatch

    field = _cos_theta_field(builders, _ops_dispatch, torch)
    state = field(_tc())
    m = _torch_sphere(builders)
    lb = _np(tgeo.laplace_beltrami(state, "f", m))
    expected = -2.0 / R**2 * np.cos(THETA)
    assert np.allclose(lb, expected, atol=1e-9)


# ----------------------------------------------------------------------
# 4. Cross-backend parity
# ----------------------------------------------------------------------
def test_curvature_cross_backend(builders):  # type: ignore[no-untyped-def]
    tm, jm = _torch_sphere(builders), _jax_sphere(builders)
    tc, jc = _tc(), _jc()
    assert np.allclose(_np(tgeo.christoffel(tc, tm)), _np(jgeo.christoffel(jc, jm)),
                       rtol=1e-9, atol=1e-9)
    assert np.allclose(_np(tgeo.ricci_tensor(tc, tm)), _np(jgeo.ricci_tensor(jc, jm)),
                       rtol=1e-9, atol=1e-9)
    assert np.allclose(_np(tgeo.scalar_curvature(tc, tm)), _np(jgeo.scalar_curvature(jc, jm)),
                       rtol=1e-9, atol=1e-9)


def test_laplace_beltrami_cross_backend(builders):  # type: ignore[no-untyped-def]
    from omnibias.fields.jax import _ops_dispatch as jd
    from omnibias.fields.torch import _ops_dispatch as td

    tf = _cos_theta_field(builders, td, torch)
    jf = _cos_theta_field(builders, jd, jnp)
    ts = tf(_tc())
    js = jf(_jc())
    t = _np(tgeo.laplace_beltrami(ts, "f", _torch_sphere(builders)))
    j = _np(jgeo.laplace_beltrami(js, "f", _jax_sphere(builders)))
    assert np.allclose(t, j, rtol=1e-9, atol=1e-9)


# ----------------------------------------------------------------------
#    Geodesic + covariant derivative smoke / parity
# ----------------------------------------------------------------------
def test_geodesic_rhs_cross_backend(builders):  # type: ignore[no-untyped-def]
    vel = np.array([[0.5, -0.3], [0.1, 0.8], [-0.2, 0.4], [0.6, 0.6]], dtype=np.float64)
    tm, jm = _torch_sphere(builders), _jax_sphere(builders)
    t = _np(tgeo.geodesic_rhs(_tc(), torch.as_tensor(vel, dtype=torch.float64), tm))
    j = _np(jgeo.geodesic_rhs(_jc(), jnp.asarray(vel, dtype=jnp.float64), jm))
    assert np.allclose(t, j, rtol=1e-9, atol=1e-9)
    # Great-circle sanity: equatorial motion (theta=pi/2, v_theta=0) has zero
    # theta-acceleration because Gamma^theta_phiphi vanishes at the equator.
    eq = np.array([[np.pi / 2, 0.0]])
    veq = np.array([[0.0, 1.0]])
    a = _np(tgeo.geodesic_rhs(
        torch.as_tensor(eq, dtype=torch.float64),
        torch.as_tensor(veq, dtype=torch.float64), tm))
    assert abs(a[0, 0]) < 1e-10


def test_covariant_derivative_scalar_is_gradient(builders):  # type: ignore[no-untyped-def]
    from omnibias.fields.torch import _ops_dispatch
    from omnibias.fields.torch.ops.basic import gradient

    field = _poly_field(builders, _ops_dispatch)
    state = field(_tc())
    m = _torch_flat(builders)
    cov = _np(tgeo.covariant_derivative(state, "f", m, kind="scalar"))
    grad = _np(gradient(state, "f", axes=("theta", "phi")))
    assert np.allclose(cov, grad, rtol=1e-12, atol=1e-12)


def _coordinate_vector_field(builders, ops_module):  # type: ignore[no-untyped-def]
    # The sphere coordinate vector field V = d/dphi: V^theta = 0, V^phi = 1.
    C = builders["Const1D"]
    comp_axes = {"vth": (C(0.0), C(1.0)), "vph": (C(1.0), C(1.0))}
    return builders["AnalyticField"](
        builders["coord_spec_2d"](), builders["ComponentSpec"](("vth", "vph")),
        comp_axes, ops_module,
    )


def test_covariant_derivative_vector_curved_sphere(builders):  # type: ignore[no-untyped-def]
    # nabla_i V^k = d_i V^k + Gamma^k_{il} V^l. For V = d/dphi (constant
    # components) the partials vanish and the Christoffel term is the whole
    # answer, so this genuinely exercises the connection on a curved manifold.
    from omnibias.fields.torch import _ops_dispatch

    state = _coordinate_vector_field(builders, _ops_dispatch)(_tc())
    m = _torch_sphere(builders)
    cov = _np(tgeo.covariant_derivative(state, ("vth", "vph"), m, kind="vector"))
    th = THETA
    # cov[:, i, k]: i = derivative axis (theta=0, phi=1); k = component.
    assert np.allclose(cov[:, 0, 1], np.cos(th) / np.sin(th), atol=1e-9)   # nabla_theta V^phi = cot
    assert np.allclose(cov[:, 1, 0], -np.sin(th) * np.cos(th), atol=1e-9)  # nabla_phi V^theta
    assert np.allclose(cov[:, 0, 0], 0.0, atol=1e-9)                       # nabla_theta V^theta
    assert np.allclose(cov[:, 1, 1], 0.0, atol=1e-9)                       # nabla_phi V^phi


def test_covariant_derivative_vector_cross_backend(builders):  # type: ignore[no-untyped-def]
    from omnibias.fields.jax import _ops_dispatch as jd
    from omnibias.fields.torch import _ops_dispatch as td

    ts = _coordinate_vector_field(builders, td)(_tc())
    js = _coordinate_vector_field(builders, jd)(_jc())
    t = _np(tgeo.covariant_derivative(ts, ("vth", "vph"), _torch_sphere(builders), kind="vector"))
    j = _np(jgeo.covariant_derivative(js, ("vth", "vph"), _jax_sphere(builders), kind="vector"))
    assert np.allclose(t, j, rtol=1e-9, atol=1e-9)
