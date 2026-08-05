# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Phase 2 ops: integration, inner products, norms, tensor divergence.

Validated three ways: (1) analytic / manufactured solution via the
separable-polynomial field, (2) torch vs jax cross-backend bit-parity, and
(3) pinned numeric regression. All in float64.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import torch
from omnibias.fields._core.quadrature import gauss_legendre

BOUNDS = [(0.0, 1.0), (-1.0, 1.0)]


def _rule():  # type: ignore[no-untyped-def]
    # 6 nodes per axis integrates our degree<=2 polynomials exactly.
    return gauss_legendre(BOUNDS, (6, 6))


def _np(x):  # type: ignore[no-untyped-def]
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


# ----------------------------------------------------------------------
# Integration: analytic + cross-backend
# ----------------------------------------------------------------------


def test_integrate_matches_analytic(make_torch_field):  # type: ignore[no-untyped-def]
    rule = _rule()
    field = make_torch_field()
    nodes = torch.as_tensor(rule.nodes, dtype=torch.float64)
    state = field(nodes)
    got_u = float(_np(state.u.integrate(rule=rule)))
    got_v = float(_np(state.v.integrate(rule=rule)))
    exp_u = field.integral_component("u", BOUNDS)
    exp_v = field.integral_component("v", BOUNDS)
    assert np.isclose(got_u, exp_u, rtol=1e-12, atol=1e-12)
    assert np.isclose(got_v, exp_v, rtol=1e-12, atol=1e-12)


def test_integrate_cross_backend(make_torch_field, make_jax_field):  # type: ignore[no-untyped-def]
    rule = _rule()
    tf, jf = make_torch_field(), make_jax_field()
    ts = tf(torch.as_tensor(rule.nodes, dtype=torch.float64))
    js = jf(jnp.asarray(rule.nodes, dtype=jnp.float64))
    for name in ("u", "v"):
        t = _np(ts[name].integrate(rule=rule))
        j = _np(js[name].integrate(rule=rule))
        assert np.allclose(t, j, rtol=1e-12, atol=1e-12), name


def test_integrate_regression_pin(make_torch_field):  # type: ignore[no-untyped-def]
    rule = _rule()
    state = make_torch_field()(torch.as_tensor(rule.nodes, dtype=torch.float64))
    # int_0^1 (1+2x+x^2) dx * int_-1^1 (1-y) dy = (1+1+1/3) * (2) = 14/3.
    assert np.isclose(float(_np(state.u.integrate(rule=rule))), 14.0 / 3.0,
                      rtol=1e-12, atol=1e-14)


# ----------------------------------------------------------------------
# Inner product & norms
# ----------------------------------------------------------------------


def test_l2_norm_matches_analytic(make_torch_field):  # type: ignore[no-untyped-def]
    rule = gauss_legendre(BOUNDS, (8, 8))  # u^2 is degree 4 -> need >=3 nodes
    field = make_torch_field()
    state = field(torch.as_tensor(rule.nodes, dtype=torch.float64))
    got = float(_np(state.u.l2_norm(rule=rule)))
    # ||u||^2 = (int_0^1 (1+2x+x^2)^2 dx)(int_-1^1 (1-y)^2 dy).
    # (1+2x+x^2)^2 = (x+1)^4 -> int_0^1 = (2^5-1)/5 = 31/5.
    # (1-y)^2 -> int_-1^1 = 8/3.
    exp = np.sqrt((31.0 / 5.0) * (8.0 / 3.0))
    assert np.isclose(got, exp, rtol=1e-12, atol=1e-12)


def test_inner_product_cross_backend(make_torch_field, make_jax_field):  # type: ignore[no-untyped-def]
    rule = gauss_legendre(BOUNDS, (8, 8))
    ts = make_torch_field()(torch.as_tensor(rule.nodes, dtype=torch.float64))
    js = make_jax_field()(jnp.asarray(rule.nodes, dtype=jnp.float64))
    t = _np(ts.ops.inner_product(ts, "u", "v", rule=rule))
    j = _np(js.ops.inner_product(js, "u", "v", rule=rule))
    assert np.allclose(t, j, rtol=1e-12, atol=1e-12)


def test_sobolev_norm_h1_matches_analytic(make_torch_field):  # type: ignore[no-untyped-def]
    rule = gauss_legendre(BOUNDS, (8, 8))
    field = make_torch_field()
    state = field(torch.as_tensor(rule.nodes, dtype=torch.float64))
    got = float(_np(state.u.sobolev_norm(rule=rule, k=1)))
    # ||u||_{H1}^2 = int u^2 + int |grad u|^2.
    # u = (1+2x+x^2)(1-y); u_x = (2+2x)(1-y); u_y = -(1+2x+x^2).
    # int u^2 = (31/5)(8/3).
    # int u_x^2 = (int_0^1 (2+2x)^2)(int_-1^1 (1-y)^2) = (28/3)(8/3).
    #   (2+2x)^2 = 4(1+x)^2 -> int_0^1 = 4*(7/3) = 28/3.
    # int u_y^2 = (int_0^1 (1+2x+x^2)^2)(int_-1^1 1 dy) = (31/5)(2).
    exp_sq = (31.0 / 5.0) * (8.0 / 3.0) + (28.0 / 3.0) * (8.0 / 3.0) + (31.0 / 5.0) * 2.0
    assert np.isclose(got, np.sqrt(exp_sq), rtol=1e-12, atol=1e-12)


def test_sobolev_norm_cross_backend(make_torch_field, make_jax_field):  # type: ignore[no-untyped-def]
    rule = gauss_legendre(BOUNDS, (8, 8))
    ts = make_torch_field()(torch.as_tensor(rule.nodes, dtype=torch.float64))
    js = make_jax_field()(jnp.asarray(rule.nodes, dtype=jnp.float64))
    for k in (0, 1, 2):
        t = _np(ts.u.sobolev_norm(rule=rule, k=k))
        j = _np(js.u.sobolev_norm(rule=rule, k=k))
        assert np.allclose(t, j, rtol=1e-12, atol=1e-12), k


# Spacetime bounds (x, y, t); the t factor is degree 2 so d^2/dt^2 != 0.
BOUNDS_ST = [(0.0, 1.0), (-1.0, 1.0), (0.0, 1.0)]


def test_sobolev_norm_h2_excludes_time_axis(make_spacetime_torch_field):  # type: ignore[no-untyped-def]
    """Regression: ``H^2`` must use the *spatial* Hessian. On a spacetime field
    the second time derivative (and mixed space-time terms) must not leak into
    the norm, matching the spatial-only gradient term used at ``k=1``."""
    rule = gauss_legendre(BOUNDS_ST, (6, 6, 6))
    state = make_spacetime_torch_field()(torch.as_tensor(rule.nodes, dtype=torch.float64))
    w = np.asarray(rule.weights)
    # Isolate the order-2 term via weights (0, 0, 1) -> sqrt(int |Hess u|^2).
    got = float(_np(state.u.sobolev_norm(rule=rule, k=2, weights=(0.0, 0.0, 1.0))))
    h_spatial = _np(state.u.hess_spatial)   # (B, 2, 2): x, y only
    h_full = _np(state.u.hess)              # (B, 3, 3): x, y, t
    int_spatial = float(np.tensordot(w, (h_spatial**2).sum(axis=(-2, -1)), axes=([0], [0])))
    int_full = float(np.tensordot(w, (h_full**2).sum(axis=(-2, -1)), axes=([0], [0])))
    assert np.isclose(got, np.sqrt(int_spatial), rtol=1e-12, atol=1e-12)
    # The full Hessian (with d^2/dt^2 + mixed terms) is strictly larger, so the
    # old full-Hessian behaviour would have changed the answer -> the fix bites.
    assert int_full > int_spatial + 1e-6


def test_sobolev_norm_spacetime_cross_backend(  # type: ignore[no-untyped-def]
    make_spacetime_torch_field, make_spacetime_jax_field
):
    rule = gauss_legendre(BOUNDS_ST, (6, 6, 6))
    ts = make_spacetime_torch_field()(torch.as_tensor(rule.nodes, dtype=torch.float64))
    js = make_spacetime_jax_field()(jnp.asarray(rule.nodes, dtype=jnp.float64))
    for k in (0, 1, 2):
        t = _np(ts.u.sobolev_norm(rule=rule, k=k))
        j = _np(js.u.sobolev_norm(rule=rule, k=k))
        assert np.allclose(t, j, rtol=1e-12, atol=1e-12), k


# ----------------------------------------------------------------------
# Tensor divergence
# ----------------------------------------------------------------------


def test_tensor_divergence_matches_analytic(make_torch_field, grid_nodes):  # type: ignore[no-untyped-def]
    field = make_torch_field()
    state = field(torch.as_tensor(grid_nodes, dtype=torch.float64))
    # sigma = [[u, v], [v, u]]; (div sigma)_0 = du/dx + dv/dy,
    #                            (div sigma)_1 = dv/dx + du/dy.
    div = _np(state.ops.tensor_divergence(state, (("u", "v"), ("v", "u"))))
    x = grid_nodes[:, 0]
    y = grid_nodes[:, 1]
    # u = (1+2x+x^2)(1-y): u_x = (2+2x)(1-y); u_y = -(1+2x+x^2).
    # v = x(1+y+y^2):      v_x = (1+y+y^2);  v_y = x(1+2y).
    ux = (2 + 2 * x) * (1 - y)
    uy = -(1 + 2 * x + x**2)
    vx = 1 + y + y**2
    vy = x * (1 + 2 * y)
    exp0 = ux + vy
    exp1 = vx + uy
    assert np.allclose(div[:, 0], exp0, rtol=1e-12, atol=1e-12)
    assert np.allclose(div[:, 1], exp1, rtol=1e-12, atol=1e-12)


def test_tensor_divergence_cross_backend(make_torch_field, make_jax_field, grid_nodes):  # type: ignore[no-untyped-def]
    ts = make_torch_field()(torch.as_tensor(grid_nodes, dtype=torch.float64))
    js = make_jax_field()(jnp.asarray(grid_nodes, dtype=jnp.float64))
    layout = (("u", "v"), ("v", "u"))
    t = _np(ts.ops.tensor_divergence(ts, layout))
    j = _np(js.ops.tensor_divergence(js, layout))
    assert np.allclose(t, j, rtol=1e-12, atol=1e-12)
