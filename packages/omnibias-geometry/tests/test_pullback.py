# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Pullback-metric (learned-chart) validation.

The pullback metric ``g = J^T h J`` turns an immersion ``phi: R^d -> R^n`` into a
:class:`MetricSpec`, after which the existing connection / curvature / field ops
consume it unchanged. We lock this down with:

1. analytic manufactured solution: the unit-sphere embedding
   ``phi(theta, phi) = (sin th cos ph, sin th sin ph, cos th)`` induces the round
   metric ``diag(1, sin^2 theta)`` and scalar curvature ``R = 2``;
2. a constant linear chart ``phi(x) = A x`` -> ``g = A^T A`` (SPD, flat);
3. composition parity: Laplace-Beltrami through a chart-derived metric equals the
   same operator through a hand-written ``MetricSpec`` carrying ``g = A^T A``;
4. torch vs jax cross-backend parity.

All in float64.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
import torch
from omnibias.geometry import ChartSpec, ManifoldSpec, MetricSpec
from omnibias.geometry.jax import ops as jgeo
from omnibias.geometry.torch import ops as tgeo

THETA = np.array([0.7, 1.1, 1.9, 2.4])
PHI = np.array([0.3, 1.5, 2.2, 4.0])
COORDS = np.stack([THETA, PHI], axis=-1).astype(np.float64)

# A constant immersion R^2 -> R^3; g = A^T A is SPD and constant (flat).
A_LINEAR = np.array([[1.0, 0.2], [0.0, 1.0], [0.3, -0.1]], dtype=np.float64)


# ----------------------------------------------------------------------
# chart / metric builders
# ----------------------------------------------------------------------
def _sphere_phi(xp):  # type: ignore[no-untyped-def]
    def phi(x):  # x: (2,) = (theta, phi)
        th, ph = x[0], x[1]
        return xp.stack([xp.sin(th) * xp.cos(ph), xp.sin(th) * xp.sin(ph), xp.cos(th)])

    return phi


def _linear_phi(xp, asarray):  # type: ignore[no-untyped-def]
    a = asarray(A_LINEAR)

    def phi(x):  # x: (2,) -> (3,)
        return a @ x

    return phi


def _round_sphere_metric(xp, stack):  # type: ignore[no-untyped-def]
    """Analytic unit round-sphere metric diag(1, sin^2 theta)."""

    def g_point(x):
        theta = x[0]
        one = 1.0 + 0.0 * theta
        z = 0.0 * theta
        return stack([stack([one, z]), stack([z, xp.sin(theta) ** 2])])

    return g_point


def _const_metric(asarray, gmat):  # type: ignore[no-untyped-def]
    g = asarray(gmat)

    def g_point(x):
        return g + 0.0 * x[0]  # keep it a function of x for vmap / jacfwd

    return g_point


def _sphere_chart(xp):  # type: ignore[no-untyped-def]
    return ChartSpec(phi=_sphere_phi(xp), domain_dim=2, ambient_dim=3, name="sphere_S2")


def _linear_chart(xp, asarray):  # type: ignore[no-untyped-def]
    return ChartSpec(
        phi=_linear_phi(xp, asarray), domain_dim=2, ambient_dim=3, name="linear",
    )


def _tc():  # type: ignore[no-untyped-def]
    return torch.as_tensor(COORDS, dtype=torch.float64)


def _jc():  # type: ignore[no-untyped-def]
    return jnp.asarray(COORDS, dtype=jnp.float64)


def _np(x):  # type: ignore[no-untyped-def]
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


# ----------------------------------------------------------------------
# 1. Sphere embedding recovers the round metric and curvature
# ----------------------------------------------------------------------
def test_sphere_pullback_metric_matches_round_torch():  # type: ignore[no-untyped-def]
    g = _np(tgeo.pullback_metric(_tc(), _sphere_chart(torch)))
    assert np.allclose(g[:, 0, 0], 1.0, atol=1e-10)
    assert np.allclose(g[:, 1, 1], np.sin(THETA) ** 2, atol=1e-10)
    assert np.allclose(g[:, 0, 1], 0.0, atol=1e-10)
    assert np.allclose(g[:, 1, 0], 0.0, atol=1e-10)


def test_sphere_pullback_metric_matches_round_jax():  # type: ignore[no-untyped-def]
    g = _np(jgeo.pullback_metric(_jc(), _sphere_chart(jnp)))
    assert np.allclose(g[:, 0, 0], 1.0, atol=1e-10)
    assert np.allclose(g[:, 1, 1], np.sin(THETA) ** 2, atol=1e-10)
    assert np.allclose(g[:, 0, 1], 0.0, atol=1e-10)


def test_sphere_pullback_scalar_curvature_is_two():  # type: ignore[no-untyped-def]
    m = ManifoldSpec("sphere", 2, tgeo.metric_spec_from_chart(_sphere_chart(torch)))
    sc = _np(tgeo.scalar_curvature(_tc(), m))
    assert np.allclose(sc, 2.0, atol=1e-7)


def test_sphere_chart_matches_analytic_round_metric_curvature():  # type: ignore[no-untyped-def]
    chart_m = ManifoldSpec("s", 2, tgeo.metric_spec_from_chart(_sphere_chart(torch)))
    analytic_m = ManifoldSpec(
        "s", 2, MetricSpec(_round_sphere_metric(torch, torch.stack), dim=2),
    )
    c = _tc()
    assert np.allclose(
        _np(tgeo.christoffel(c, chart_m)), _np(tgeo.christoffel(c, analytic_m)),
        atol=1e-8,
    )
    assert np.allclose(
        _np(tgeo.scalar_curvature(c, chart_m)),
        _np(tgeo.scalar_curvature(c, analytic_m)),
        atol=1e-7,
    )


# ----------------------------------------------------------------------
# 2. Constant linear chart: g = A^T A is SPD and flat
# ----------------------------------------------------------------------
def test_linear_chart_pullback_is_AtA_and_spd():  # type: ignore[no-untyped-def]
    g = _np(tgeo.pullback_metric(_tc(), _linear_chart(torch, lambda a: torch.as_tensor(a, dtype=torch.float64))))
    expected = A_LINEAR.T @ A_LINEAR
    assert np.allclose(g, expected[None].repeat(len(THETA), axis=0), atol=1e-10)
    # symmetric + positive-definite
    assert np.allclose(g, np.transpose(g, (0, 2, 1)), atol=1e-12)
    eigs = np.linalg.eigvalsh(g)
    assert np.all(eigs > 0)


def test_linear_chart_is_flat():  # type: ignore[no-untyped-def]
    m = ManifoldSpec(
        "lin", 2,
        tgeo.metric_spec_from_chart(
            _linear_chart(torch, lambda a: torch.as_tensor(a, dtype=torch.float64)),
        ),
    )
    c = _tc()
    assert np.allclose(_np(tgeo.christoffel(c, m)), 0.0, atol=1e-9)
    assert np.allclose(_np(tgeo.scalar_curvature(c, m)), 0.0, atol=1e-8)


# ----------------------------------------------------------------------
# 3. Composition parity: Laplace-Beltrami via chart == via hand metric
# ----------------------------------------------------------------------
def _poly_field(builders, ops_module):  # type: ignore[no-untyped-def]
    P = builders["Poly1D"]
    comp_axes = {"f": (P((0.3, 1.0, 0.5, -0.2)), P((1.0, -0.4, 0.7)))}
    return builders["AnalyticField"](
        builders["coord_spec_2d"](), builders["ComponentSpec"](("f",)),
        comp_axes, ops_module,
    )


def test_pullback_laplace_beltrami_matches_constant_metric(builders):  # type: ignore[no-untyped-def]
    from omnibias.fields.torch import _ops_dispatch

    state = _poly_field(builders, _ops_dispatch)(_tc())
    chart = _linear_chart(torch, lambda a: torch.as_tensor(a, dtype=torch.float64))
    m_chart = ManifoldSpec("lin", 2, tgeo.metric_spec_from_chart(chart))
    g_const = A_LINEAR.T @ A_LINEAR
    m_const = ManifoldSpec(
        "lin", 2,
        MetricSpec(_const_metric(lambda a: torch.as_tensor(a, dtype=torch.float64), g_const), dim=2),
    )
    lb_chart = _np(tgeo.laplace_beltrami(state, "f", m_chart))
    lb_const = _np(tgeo.laplace_beltrami(state, "f", m_const))
    assert np.allclose(lb_chart, lb_const, rtol=1e-10, atol=1e-10)


# ----------------------------------------------------------------------
# 4. Cross-backend parity
# ----------------------------------------------------------------------
def test_pullback_metric_cross_backend():  # type: ignore[no-untyped-def]
    t = _np(tgeo.pullback_metric(_tc(), _sphere_chart(torch)))
    j = _np(jgeo.pullback_metric(_jc(), _sphere_chart(jnp)))
    assert np.allclose(t, j, rtol=1e-9, atol=1e-9)


def test_pullback_curvature_cross_backend():  # type: ignore[no-untyped-def]
    tm = ManifoldSpec("s", 2, tgeo.metric_spec_from_chart(_sphere_chart(torch)))
    jm = ManifoldSpec("s", 2, jgeo.metric_spec_from_chart(_sphere_chart(jnp)))
    assert np.allclose(
        _np(tgeo.scalar_curvature(_tc(), tm)),
        _np(jgeo.scalar_curvature(_jc(), jm)),
        rtol=1e-8, atol=1e-8,
    )


# ----------------------------------------------------------------------
# 5. Ambient (non-Euclidean) metric pulls back correctly
# ----------------------------------------------------------------------
def test_pullback_with_scaled_ambient_metric():  # type: ignore[no-untyped-def]
    # Ambient metric h = c^2 I scales the pullback by c^2: identity chart on R^2.
    c2 = 4.0

    def ident(x):
        return x

    def ambient(y):
        return c2 * torch.eye(2, dtype=y.dtype)

    chart = ChartSpec(phi=ident, domain_dim=2, ambient_dim=2, ambient_metric=ambient, name="scaled")
    g = _np(tgeo.pullback_metric(_tc(), chart))
    assert np.allclose(g, c2 * np.eye(2)[None], atol=1e-10)


# ----------------------------------------------------------------------
# 6. Schema validation
# ----------------------------------------------------------------------
def test_chartspec_rejects_ambient_smaller_than_domain():  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError, match="ambient_dim"):
        ChartSpec(phi=lambda x: x, domain_dim=3, ambient_dim=2)


def test_chartspec_rejects_nonpositive_dims():  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError, match="domain_dim"):
        ChartSpec(phi=lambda x: x, domain_dim=0, ambient_dim=2)
