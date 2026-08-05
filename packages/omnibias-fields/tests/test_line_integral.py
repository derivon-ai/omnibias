# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Gradient theorem (multivariate FTC) for the field ``line_integral`` op.

For a scalar potential ``u`` the line integral of its gradient along any curve
equals the endpoint potential difference::

    int_C grad u . dr = u(curve(t1)) - u(curve(t0)).

Validated three ways: the identity itself (polynomial curve -> Gauss-Legendre is
exact), a closed loop integrating to ~0, and torch<->jax bit-parity.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import torch
from omnibias.fields._core.quadrature import gauss_legendre

# u(x, y) = (1 + 2x + x^2)(1 - y) is the conftest analytic potential; its gradient
# is closed form, so the only approximation is the quadrature (made exact here).
_P0 = np.array([0.1, -0.3])
_P1 = np.array([0.7, 0.5])


def _np(x):  # type: ignore[no-untyped-def]
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _torch_linear(p0, p1):  # type: ignore[no-untyped-def]
    a = torch.as_tensor(p0, dtype=torch.float64)
    b = torch.as_tensor(p1, dtype=torch.float64)

    def curve(t: torch.Tensor) -> torch.Tensor:
        return a + t[0] * (b - a)

    return curve


def _jax_linear(p0, p1):  # type: ignore[no-untyped-def]
    a = jnp.asarray(p0, dtype=jnp.float64)
    b = jnp.asarray(p1, dtype=jnp.float64)

    def curve(t):  # type: ignore[no-untyped-def]
        return a + t[0] * (b - a)

    return curve


def _torch_curve_points(field, curve, rule):  # type: ignore[no-untyped-def]
    from omnibias.fields.torch.ops.integral import quadrature_nodes

    nodes = quadrature_nodes(rule, like=torch.zeros(1, dtype=torch.float64))
    return field(torch.func.vmap(curve)(nodes))


def _jax_curve_points(field, curve, rule):  # type: ignore[no-untyped-def]
    import jax
    from omnibias.fields.jax.ops.integral import quadrature_nodes

    nodes = quadrature_nodes(rule, like=jnp.zeros(1, dtype=jnp.float64))
    return field(jax.vmap(curve)(nodes))


def _endpoint_difference(field, p0, p1, backend):  # type: ignore[no-untyped-def]
    if backend == "torch":
        ends = field(torch.as_tensor(np.stack([p0, p1]), dtype=torch.float64))
    else:
        ends = field(jnp.asarray(np.stack([p0, p1]), dtype=jnp.float64))
    phi = _np(ends.u.value)
    return float(phi[1] - phi[0])


# ----------------------------------------------------------------------
# Gradient theorem: line integral == endpoint potential difference
# ----------------------------------------------------------------------


def test_gradient_theorem_torch(make_torch_field):  # type: ignore[no-untyped-def]
    field = make_torch_field()
    curve = _torch_linear(_P0, _P1)
    rule = gauss_legendre([(0.0, 1.0)], 8)  # exact for the degree-2 integrand
    state = _torch_curve_points(field, curve, rule)
    got = float(_np(state.u.line_integral(curve, rule=rule)))
    exp = _endpoint_difference(field, _P0, _P1, "torch")
    assert np.isclose(got, exp, rtol=1e-12, atol=1e-12)


def test_gradient_theorem_jax(make_jax_field):  # type: ignore[no-untyped-def]
    field = make_jax_field()
    curve = _jax_linear(_P0, _P1)
    rule = gauss_legendre([(0.0, 1.0)], 8)
    state = _jax_curve_points(field, curve, rule)
    got = float(_np(state.u.line_integral(curve, rule=rule)))
    exp = _endpoint_difference(field, _P0, _P1, "jax")
    assert np.isclose(got, exp, rtol=1e-12, atol=1e-12)


def test_line_integral_is_path_independent_torch(make_torch_field):  # type: ignore[no-untyped-def]
    """A different (curved) path between the same endpoints gives the same value."""
    field = make_torch_field()
    rule = gauss_legendre([(0.0, 1.0)], 10)
    a = torch.as_tensor(_P0, dtype=torch.float64)
    b = torch.as_tensor(_P1, dtype=torch.float64)

    def curved(t: torch.Tensor) -> torch.Tensor:
        s = t[0]
        # a quadratic detour that still starts at P0 (s=0) and ends at P1 (s=1)
        bump = 0.4 * s * (1 - s)
        return a + s * (b - a) + torch.stack([bump, -bump])

    state = _torch_curve_points(field, curved, rule)
    got = float(_np(state.u.line_integral(curved, rule=rule)))
    exp = _endpoint_difference(field, _P0, _P1, "torch")
    assert np.isclose(got, exp, rtol=1e-10, atol=1e-10)


# ----------------------------------------------------------------------
# Closed loop integrates to ~0
# ----------------------------------------------------------------------


def test_closed_loop_is_zero_torch(make_torch_field):  # type: ignore[no-untyped-def]
    field = make_torch_field()
    rule = gauss_legendre([(0.0, 1.0)], 10)  # exact for the polynomial integrand

    def loop(t: torch.Tensor) -> torch.Tensor:
        s = t[0]  # r(0) = r(1) = (0, 0): a closed teardrop
        return torch.stack([s * (1 - s), s * s * (1 - s)])

    state = _torch_curve_points(field, loop, rule)
    got = float(_np(state.u.line_integral(loop, rule=rule)))
    assert abs(got) < 1e-10


# ----------------------------------------------------------------------
# Cross-backend parity + dispatch/view equivalence
# ----------------------------------------------------------------------


def test_line_integral_cross_backend(make_torch_field, make_jax_field):  # type: ignore[no-untyped-def]
    rule = gauss_legendre([(0.0, 1.0)], 8)
    tf, jf = make_torch_field(), make_jax_field()
    tcurve, jcurve = _torch_linear(_P0, _P1), _jax_linear(_P0, _P1)
    ts = _torch_curve_points(tf, tcurve, rule)
    js = _jax_curve_points(jf, jcurve, rule)
    t = _np(ts.u.line_integral(tcurve, rule=rule))
    j = _np(js.u.line_integral(jcurve, rule=rule))
    assert np.allclose(t, j, rtol=1e-9, atol=1e-12)


def test_dispatch_matches_view_torch(make_torch_field):  # type: ignore[no-untyped-def]
    field = make_torch_field()
    curve = _torch_linear(_P0, _P1)
    rule = gauss_legendre([(0.0, 1.0)], 8)
    state = _torch_curve_points(field, curve, rule)
    via_view = _np(state.u.line_integral(curve, rule=rule))
    via_ops = _np(state.ops.line_integral(state, "u", curve, rule=rule))
    assert np.array_equal(via_view, via_ops)
