# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Geodesics as least action: the metric Lagrangian bridge to omnibias-geometry.

The kinetic Lagrangian ``L = 1/2 g_ij qdot^i qdot^j`` has Euler-Lagrange residual
``EL_k = g_kj (qddot^j - geodesic_rhs^j)``, so this validates the generic
``euler_lagrange_residual`` against ``omnibias.geometry``'s ``geodesic_rhs`` and
``metric`` on the sphere S^2 and the hyperbolic (Poincare) plane. A true geodesic
(the equator / a vertical line) has zero residual. All float64, torch/jax parity.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from _traj import jax_state, to_np, torch_state
from omnibias.fields._core.quadrature import gauss_legendre
from omnibias.fields.torch.ops.integral import quadrature_nodes
from omnibias.variational.jax import ops as jv
from omnibias.variational.torch import ops as tv

pytest.importorskip("omnibias.geometry")

from omnibias.geometry._core.manifold import ManifoldSpec, MetricSpec  # noqa: E402

T = np.array([0.15, 0.5, 0.9, 1.3, 1.75], dtype=np.float64)


# ----------------------------- manifolds -----------------------------


def _sphere(xp):  # type: ignore[no-untyped-def]
    def g_point(x):  # type: ignore[no-untyped-def]
        z = xp.zeros((), dtype=x.dtype)
        o = xp.ones((), dtype=x.dtype)
        s2 = xp.sin(x[0]) ** 2
        return xp.stack([xp.stack([o, z]), xp.stack([z, s2])])

    return ManifoldSpec("S2", 2, MetricSpec(g_point, 2, name="round_sphere"))


def _hyperbolic(xp):  # type: ignore[no-untyped-def]
    def g_point(p):  # type: ignore[no-untyped-def]
        z = xp.zeros((), dtype=p.dtype)
        inv = 1.0 / (p[1] ** 2)
        return xp.stack([xp.stack([inv, z]), xp.stack([z, inv])])

    return ManifoldSpec("H2", 2, MetricSpec(g_point, 2, name="poincare_half_plane"))


# --------------------------- trajectories ----------------------------


def _sphere_curve(xp, _omega):  # type: ignore[no-untyped-def]
    # A generic (non-geodesic) curve on S^2, kept away from the poles.
    return {
        "theta": (
            lambda t: 0.9 + 0.3 * xp.sin(0.7 * t),
            lambda t: 0.3 * 0.7 * xp.cos(0.7 * t),
            lambda t: -0.3 * 0.7**2 * xp.sin(0.7 * t),
        ),
        "phi": (lambda t: 0.5 * t, lambda t: 0.5 * xp.ones_like(t), lambda t: xp.zeros_like(t)),
    }


def _hyperbolic_curve(xp, _omega):  # type: ignore[no-untyped-def]
    # A generic curve on H^2 with y > 0 throughout.
    return {
        "x": (lambda t: 0.4 * xp.sin(t), lambda t: 0.4 * xp.cos(t), lambda t: -0.4 * xp.sin(t)),
        "y": (lambda t: 1.5 + 0.5 * xp.cos(t), lambda t: -0.5 * xp.sin(t), lambda t: -0.5 * xp.cos(t)),
    }


def _equator(xp, _omega):  # type: ignore[no-untyped-def]
    # theta = pi/2 (const), phi = t: a great circle -> a geodesic.
    return {
        "theta": (
            lambda t: (math.pi / 2) * xp.ones_like(t),
            lambda t: xp.zeros_like(t),
            lambda t: xp.zeros_like(t),
        ),
        "phi": (lambda t: t, lambda t: xp.ones_like(t), lambda t: xp.zeros_like(t)),
    }


def _el_matches_lowered_geodesic(specs, manifold_fn, dof) -> None:  # type: ignore[no-untyped-def]
    from omnibias.fields.torch.ops.basic import stack_components, vector_derivative
    from omnibias.geometry.torch.ops.connection import geodesic_rhs, metric

    manifold = manifold_fn(torch)
    state = torch_state(specs, 0.0, T)
    lag = tv.metric_lagrangian(manifold, dof=dof)
    el = to_np(tv.euler_lagrange_residual(state, lag))

    q = stack_components(state, dof)
    qdot = vector_derivative(state, dof, axis="t", order=1)
    qddot = vector_derivative(state, dof, axis="t", order=2)
    g = metric(q, manifold)
    grhs = geodesic_rhs(q, qdot, manifold)
    expected = to_np(torch.einsum("bkm,bm->bk", g, qddot - grhs))
    assert np.allclose(el, expected, atol=1e-9)


def test_sphere_el_equals_geodesic_rhs() -> None:
    _el_matches_lowered_geodesic(_sphere_curve, _sphere, ("theta", "phi"))


def test_hyperbolic_el_equals_geodesic_rhs() -> None:
    _el_matches_lowered_geodesic(_hyperbolic_curve, _hyperbolic, ("x", "y"))


def test_equator_is_a_geodesic() -> None:
    manifold = _sphere(torch)
    state = torch_state(_equator, 0.0, T)
    lag = tv.metric_lagrangian(manifold, dof=("theta", "phi"))
    el = to_np(tv.euler_lagrange_residual(state, lag))
    assert np.allclose(el, 0.0, atol=1e-10)


def test_geodesic_action_equator_arc_length() -> None:
    # On the equator ds = dphi, so the arc length over t in [0, 1.2] is 1.2.
    manifold = _sphere(torch)
    rule = gauss_legendre([(0.0, 1.2)], 16)
    nodes = quadrature_nodes(rule, like=torch.zeros(1, dtype=torch.float64))
    state = torch_state(_equator, 0.0, nodes[:, 0].numpy())
    length = float(to_np(tv.geodesic_action(state, manifold, rule=rule, dof=("theta", "phi"))))
    assert abs(length - 1.2) < 1e-10


def test_geodesic_bridge_cross_backend() -> None:
    import jax.numpy as jnp

    tm, jm = _sphere(torch), _sphere(jnp)
    ts = torch_state(_sphere_curve, 0.0, T)
    js = jax_state(_sphere_curve, 0.0, T)
    t_el = to_np(tv.euler_lagrange_residual(ts, tv.metric_lagrangian(tm, dof=("theta", "phi"))))
    j_el = to_np(jv.euler_lagrange_residual(js, jv.metric_lagrangian(jm, dof=("theta", "phi"))))
    assert np.allclose(t_el, j_el, rtol=1e-12, atol=1e-12)
