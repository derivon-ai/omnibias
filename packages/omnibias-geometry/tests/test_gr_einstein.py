# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""General-relativity curvature ops (WS2): Einstein tensor + invariants.

Independent oracles per the validation strategy:
1. analytic / manufactured solutions -- Schwarzschild vacuum (``G = 0``,
   ``K = 48 M^2/r^6``), de Sitter FRW (``G_00 = 3H^2``, ``G + Lambda g = 0``),
   the round ``S^3`` (``G = -(1/R^2) g``, ``Weyl = 0``, ``K = 12/R^4``);
2. internal-identity cross-checks -- the 2D vanishing ``G = 0`` on a non-trivial
   conformal metric, the trace identity ``g^{mu nu} G_{mu nu} = (2-d)/2 R``,
   ``K = R^2`` in 2D, Weyl tracelessness, vacuum ``Weyl = lowered Riemann``;
3. an independent numerical path -- the contracted Bianchi identity
   ``nabla^mu G_{mu nu} = 0`` by central finite differences on a *non-Einstein*
   FRW metric;
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

# --- test coordinate batches (avoid coordinate singularities) ----------
SCHW_M = 1.0
SCHW = np.array(
    [[0.0, 6.0, 0.9, 0.5],
     [1.0, 8.0, 1.3, 2.1],
     [2.0, 10.0, 2.0, 4.0],
     [0.5, 12.0, 1.7, 1.2]],
    dtype=np.float64,
)
S3_R = 1.4
S3 = np.array(
    [[0.7, 0.9, 0.3],
     [1.1, 1.3, 1.5],
     [1.9, 2.0, 2.2],
     [2.4, 1.7, 4.0]],
    dtype=np.float64,
)
DS_H = 0.5
FRW = np.array(
    [[0.5, 0.1, 0.2, 0.3],
     [0.8, -0.1, 0.4, 0.2],
     [1.0, 0.3, -0.2, 0.1],
     [1.3, 0.2, 0.5, -0.3]],
    dtype=np.float64,
)


def _np(x):  # type: ignore[no-untyped-def]
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def _diag(stack, entries):  # type: ignore[no-untyped-def]
    n = len(entries)
    z = 0.0 * entries[0]
    return stack([stack([entries[i] if i == j else z for j in range(n)]) for i in range(n)])


# --- backend-neutral analytic metrics ----------------------------------
def _schwarzschild(xp, stack, mass):  # type: ignore[no-untyped-def]
    def g_point(x):  # (t, r, theta, phi)
        r, th = x[1], x[2]
        f = 1.0 - 2.0 * mass / r
        return _diag(stack, [-f, 1.0 / f, r**2, r**2 * xp.sin(th) ** 2])

    return g_point


def _sphere3(xp, stack, radius):  # type: ignore[no-untyped-def]
    r2 = radius**2

    def g_point(x):  # (chi, theta, phi)
        chi, th = x[0], x[1]
        s = xp.sin(chi)
        return _diag(stack, [r2 + 0.0 * chi, r2 * s**2, r2 * s**2 * xp.sin(th) ** 2])

    return g_point


def _frw(xp, stack, afn):  # type: ignore[no-untyped-def]
    def g_point(x):  # (t, x, y, z)
        t = x[0]
        a2 = afn(xp, t) ** 2
        return _diag(stack, [-1.0 + 0.0 * t, a2, a2, a2])

    return g_point


def _de_sitter_a(xp, t):  # type: ignore[no-untyped-def]
    return xp.exp(DS_H * t)


def _nonEinstein_a(xp, t):  # type: ignore[no-untyped-def]
    # Smooth, positive, and NOT of constant Hubble rate -> non-Einstein G(t).
    return xp.exp(t * t)


def _man(name, dim, gfn, sig=()):  # type: ignore[no-untyped-def]
    return ManifoldSpec(name, dim, MetricSpec(gfn, dim=dim, name=name, signature=sig))


def _tS():  # type: ignore[no-untyped-def]
    return torch.as_tensor(SCHW, dtype=torch.float64)


def _jS():  # type: ignore[no-untyped-def]
    return jnp.asarray(SCHW, dtype=jnp.float64)


LOR4 = (-1, 1, 1, 1)


# ----------------------------------------------------------------------
# 1. Schwarzschild vacuum: G = 0, but Kretschmann = 48 M^2/r^6 (real curvature)
# ----------------------------------------------------------------------
def test_schwarzschild_vacuum_einstein_zero():  # type: ignore[no-untyped-def]
    m = _man("schwarzschild", 4, _schwarzschild(torch, torch.stack, SCHW_M), LOR4)
    g = _np(tgeo.einstein_tensor(_tS(), m))
    assert np.allclose(g, 0.0, atol=1e-6)


def test_schwarzschild_kretschmann():  # type: ignore[no-untyped-def]
    m = _man("schwarzschild", 4, _schwarzschild(torch, torch.stack, SCHW_M), LOR4)
    k = _np(tgeo.kretschmann_scalar(_tS(), m))
    r = SCHW[:, 1]
    assert np.allclose(k, 48.0 * SCHW_M**2 / r**6, rtol=1e-7, atol=1e-10)


def test_schwarzschild_weyl_equals_lowered_riemann_in_vacuum():  # type: ignore[no-untyped-def]
    # In vacuum (Ricci = 0) the Weyl tensor is the full lowered Riemann tensor.
    m = _man("schwarzschild", 4, _schwarzschild(torch, torch.stack, SCHW_M), LOR4)
    c = _np(tgeo.weyl_tensor(_tS(), m))
    rd = _np(tgeo.lowered_riemann(_tS(), m))
    assert np.allclose(c, rd, atol=1e-6)
    assert np.max(np.abs(rd)) > 1e-3  # genuinely non-zero curvature


def test_schwarzschild_weyl_traceless():  # type: ignore[no-untyped-def]
    m = _man("schwarzschild", 4, _schwarzschild(torch, torch.stack, SCHW_M), LOR4)
    ginv = _np(tgeo.inverse_metric(_tS(), m))
    c = _np(tgeo.weyl_tensor(_tS(), m))
    trace = np.einsum("brm,brsmn->bsn", ginv, c)
    assert np.allclose(trace, 0.0, atol=1e-6)


def test_schwarzschild_einstein_residual_vacuum():  # type: ignore[no-untyped-def]
    m = _man("schwarzschild", 4, _schwarzschild(torch, torch.stack, SCHW_M), LOR4)
    res = _np(tgeo.einstein_equation_residual(_tS(), m))  # vacuum, Lambda=0
    assert np.allclose(res, 0.0, atol=1e-6)


# ----------------------------------------------------------------------
# 2. de Sitter FRW: G_00 = 3H^2 and G + Lambda g = 0 with Lambda = 3H^2
# ----------------------------------------------------------------------
def test_de_sitter_friedmann_g00():  # type: ignore[no-untyped-def]
    m = _man("frw_ds", 4, _frw(torch, torch.stack, _de_sitter_a), LOR4)
    c = torch.as_tensor(FRW, dtype=torch.float64)
    g = _np(tgeo.einstein_tensor(c, m))
    assert np.allclose(g[:, 0, 0], 3.0 * DS_H**2, rtol=1e-7, atol=1e-8)


def test_de_sitter_cosmological_constant_residual():  # type: ignore[no-untyped-def]
    m = _man("frw_ds", 4, _frw(torch, torch.stack, _de_sitter_a), LOR4)
    c = torch.as_tensor(FRW, dtype=torch.float64)
    lam = 3.0 * DS_H**2
    res = _np(tgeo.einstein_equation_residual(c, m, cosmological_constant=lam))
    assert np.allclose(res, 0.0, atol=1e-7)


def test_einstein_residual_matches_stress_energy():  # type: ignore[no-untyped-def]
    # Build T = G / kappa so the residual vanishes; a direct wiring check.
    m = _man("frw_ds", 4, _frw(torch, torch.stack, _de_sitter_a), LOR4)
    c = torch.as_tensor(FRW, dtype=torch.float64)
    kappa = 8.0 * np.pi
    ein = tgeo.einstein_tensor(c, m)
    t_munu = ein / kappa
    res = _np(tgeo.einstein_equation_residual(c, m, t_munu, kappa=kappa))
    assert np.allclose(res, 0.0, atol=1e-8)


# ----------------------------------------------------------------------
# 3. Round S^3 (maximally symmetric): G = -(1/R^2) g, Weyl = 0, K = 12/R^4
# ----------------------------------------------------------------------
def _s3_coords(xp):  # type: ignore[no-untyped-def]
    return xp.asarray(S3, dtype=xp.float64) if xp is jnp else torch.as_tensor(S3, dtype=torch.float64)


def test_s3_einstein_proportional_to_metric():  # type: ignore[no-untyped-def]
    m = _man("s3", 3, _sphere3(torch, torch.stack, S3_R))
    c = _s3_coords(torch)
    g = _np(tgeo.einstein_tensor(c, m))
    gm = _np(tgeo.metric(c, m))
    assert np.allclose(g, -(1.0 / S3_R**2) * gm, rtol=1e-8, atol=1e-9)


def test_s3_weyl_vanishes():  # type: ignore[no-untyped-def]
    m = _man("s3", 3, _sphere3(torch, torch.stack, S3_R))
    c = _s3_coords(torch)
    weyl = _np(tgeo.weyl_tensor(c, m))
    assert np.allclose(weyl, 0.0, atol=1e-8)


def test_s3_kretschmann_constant_curvature():  # type: ignore[no-untyped-def]
    # Maximally symmetric d-space: K = 2 d(d-1) / R^4. For d=3: 12/R^4.
    m = _man("s3", 3, _sphere3(torch, torch.stack, S3_R))
    c = _s3_coords(torch)
    k = _np(tgeo.kretschmann_scalar(c, m))
    assert np.allclose(k, 12.0 / S3_R**4, rtol=1e-8, atol=1e-9)


def test_s3_trace_identity():  # type: ignore[no-untyped-def]
    # Independent contraction path: g^{mu nu} G_{mu nu} = (2-d)/2 * R.
    m = _man("s3", 3, _sphere3(torch, torch.stack, S3_R))
    c = _s3_coords(torch)
    ginv = _np(tgeo.inverse_metric(c, m))
    ein = _np(tgeo.einstein_tensor(c, m))
    sc = _np(tgeo.scalar_curvature(c, m))
    tr = np.einsum("bmn,bmn->b", ginv, ein)
    assert np.allclose(tr, (2 - 3) / 2.0 * sc, rtol=1e-8, atol=1e-9)


# ----------------------------------------------------------------------
# 4. 2D identities on a non-trivial conformal metric
#    (uses the shared conftest builders)
# ----------------------------------------------------------------------
def _torch_conformal(builders):  # type: ignore[no-untyped-def]
    g = builders["conformal_metric_factory"](torch, torch.stack)
    return ManifoldSpec("conformal_2d", 2, MetricSpec(g, dim=2, name="conformal"))


CONF_COORDS = np.stack(
    [np.array([0.7, 1.1, 1.9, 2.4]), np.array([0.3, 1.5, 2.2, 4.0])], axis=-1,
).astype(np.float64)


def test_einstein_vanishes_in_2d(builders):  # type: ignore[no-untyped-def]
    # G_{mu nu} == 0 identically in 2D even for a curved (conformal) metric.
    m = _torch_conformal(builders)
    c = torch.as_tensor(CONF_COORDS, dtype=torch.float64)
    g = _np(tgeo.einstein_tensor(c, m))
    assert np.allclose(g, 0.0, atol=1e-9)


def test_kretschmann_equals_scalar_squared_2d(builders):  # type: ignore[no-untyped-def]
    # In 2D, K = R^2 (Gauss curvature identity), tying Kretschmann back to R.
    m = _torch_conformal(builders)
    c = torch.as_tensor(CONF_COORDS, dtype=torch.float64)
    k = _np(tgeo.kretschmann_scalar(c, m))
    sc = _np(tgeo.scalar_curvature(c, m))
    assert np.allclose(k, sc**2, rtol=1e-9, atol=1e-10)


# ----------------------------------------------------------------------
# 5. geodesic deviation
# ----------------------------------------------------------------------
def test_geodesic_deviation_flat_is_zero(builders):  # type: ignore[no-untyped-def]
    g = builders["flat_metric_factory"](torch, torch.stack, 2)
    m = ManifoldSpec("flat", 2, MetricSpec(g, dim=2))
    c = torch.as_tensor(CONF_COORDS, dtype=torch.float64)
    u = torch.as_tensor(np.full_like(CONF_COORDS, 0.4), dtype=torch.float64)
    xi = torch.as_tensor(np.full_like(CONF_COORDS, 0.7), dtype=torch.float64)
    a = _np(tgeo.geodesic_deviation(c, u, xi, m))
    assert np.allclose(a, 0.0, atol=1e-10)


def test_geodesic_deviation_nonzero_on_sphere3():  # type: ignore[no-untyped-def]
    m = _man("s3", 3, _sphere3(torch, torch.stack, S3_R))
    c = _s3_coords(torch)
    u = torch.as_tensor([[0.5, -0.3, 0.2]] * 4, dtype=torch.float64)
    xi = torch.as_tensor([[0.1, 0.8, -0.4]] * 4, dtype=torch.float64)
    a = _np(tgeo.geodesic_deviation(c, u, xi, m))
    assert np.max(np.abs(a)) > 1e-3


# ----------------------------------------------------------------------
# 6. Independent numerical path: contracted Bianchi nabla^mu G_{mu nu} = 0
#    on a *non-Einstein* FRW metric, by central finite differences.
# ----------------------------------------------------------------------
def test_contracted_bianchi_finite_difference():  # type: ignore[no-untyped-def]
    m = _man("frw_ne", 4, _frw(torch, torch.stack, _nonEinstein_a), LOR4)
    base = np.array([[0.30, 0.1, -0.2, 0.15]], dtype=np.float64)
    h = 1e-4

    def einstein_at(pt):  # type: ignore[no-untyped-def]
        return _np(tgeo.einstein_tensor(torch.as_tensor(pt, dtype=torch.float64), m))[0]

    d = 4
    # dG[a, m, n] = d_a G_{m n} by central differences.
    dG = np.zeros((d, d, d))
    for a in range(d):
        pp = base.copy()
        pp[0, a] += h
        pm = base.copy()
        pm[0, a] -= h
        dG[a] = (einstein_at(pp) - einstein_at(pm)) / (2.0 * h)
    gamma = _np(tgeo.christoffel(torch.as_tensor(base, dtype=torch.float64), m))[0]
    ginv = _np(tgeo.inverse_metric(torch.as_tensor(base, dtype=torch.float64), m))[0]
    ein = einstein_at(base)
    # nabla_a G_{m n} = d_a G_{m n} - Gamma^l_{a m} G_{l n} - Gamma^l_{a n} G_{m l}
    cov = (dG
           - np.einsum("lam,ln->amn", gamma, ein)
           - np.einsum("lan,ml->amn", gamma, ein))
    div = np.einsum("am,amn->n", ginv, cov)
    # G actually varies here (non-Einstein): confirm the test is non-trivial.
    assert np.max(np.abs(dG)) > 1e-2
    assert np.allclose(div, 0.0, atol=1e-4)


# ----------------------------------------------------------------------
# 7. cross-backend parity
# ----------------------------------------------------------------------
def test_gr_cross_backend_schwarzschild():  # type: ignore[no-untyped-def]
    tm = _man("schwarzschild", 4, _schwarzschild(torch, torch.stack, SCHW_M), LOR4)
    jm = _man("schwarzschild", 4, _schwarzschild(jnp, jnp.stack, SCHW_M), LOR4)
    for fn in ("einstein_tensor", "kretschmann_scalar", "lowered_riemann", "weyl_tensor"):
        t = _np(getattr(tgeo, fn)(_tS(), tm))
        j = _np(getattr(jgeo, fn)(_jS(), jm))
        assert np.allclose(t, j, rtol=1e-8, atol=1e-9), fn


def test_gr_cross_backend_geodesic_deviation():  # type: ignore[no-untyped-def]
    tm = _man("s3", 3, _sphere3(torch, torch.stack, S3_R))
    jm = _man("s3", 3, _sphere3(jnp, jnp.stack, S3_R))
    u = np.array([[0.5, -0.3, 0.2]] * 4, dtype=np.float64)
    xi = np.array([[0.1, 0.8, -0.4]] * 4, dtype=np.float64)
    t = _np(tgeo.geodesic_deviation(
        _s3_coords(torch), torch.as_tensor(u, dtype=torch.float64),
        torch.as_tensor(xi, dtype=torch.float64), tm))
    j = _np(jgeo.geodesic_deviation(
        _s3_coords(jnp), jnp.asarray(u, dtype=jnp.float64),
        jnp.asarray(xi, dtype=jnp.float64), jm))
    assert np.allclose(t, j, rtol=1e-8, atol=1e-9)


# ----------------------------------------------------------------------
# 8. symbolic (sympy) cross-check of the Einstein tensor on S^3
# ----------------------------------------------------------------------
def test_s3_einstein_matches_sympy():  # type: ignore[no-untyped-def]
    sympy = pytest.importorskip("sympy")
    chi, th, ph = sympy.symbols("chi theta phi", positive=True)
    r2 = S3_R**2
    g = sympy.diag(r2, r2 * sympy.sin(chi) ** 2,
                   r2 * sympy.sin(chi) ** 2 * sympy.sin(th) ** 2)
    ginv = g.inv()
    coords = [chi, th, ph]
    d = 3

    def christ(k, i, j):  # type: ignore[no-untyped-def]
        return sympy.Rational(1, 2) * sum(
            ginv[k, a] * (sympy.diff(g[a, j], coords[i])
                          + sympy.diff(g[a, i], coords[j])
                          - sympy.diff(g[i, j], coords[a])) for a in range(d))

    Gm = [[[sympy.simplify(christ(k, i, j)) for j in range(d)]
           for i in range(d)] for k in range(d)]

    def riem(rho, sig, mu, nu):  # type: ignore[no-untyped-def]
        term = (sympy.diff(Gm[rho][nu][sig], coords[mu])
                - sympy.diff(Gm[rho][mu][sig], coords[nu]))
        term += sum(Gm[rho][mu][a] * Gm[a][nu][sig]
                    - Gm[rho][nu][a] * Gm[a][mu][sig] for a in range(d))
        return sympy.simplify(term)

    Ric = sympy.Matrix(d, d, lambda s, n: sympy.simplify(
        sum(riem(r, s, r, n) for r in range(d))))
    scal = sympy.simplify(sum(ginv[s, n] * Ric[s, n]
                              for s in range(d) for n in range(d)))
    Gt = sympy.Matrix(d, d, lambda s, n: sympy.simplify(
        Ric[s, n] - sympy.Rational(1, 2) * scal * g[s, n]))
    g_fn = sympy.lambdify((chi, th, ph), Gt, "numpy")

    m = _man("s3", 3, _sphere3(torch, torch.stack, S3_R))
    got = _np(tgeo.einstein_tensor(_s3_coords(torch), m))
    for i in range(len(S3)):
        exp = np.asarray(g_fn(*S3[i]), dtype=np.float64)
        assert np.allclose(got[i], exp, atol=1e-8)
