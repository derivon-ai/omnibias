# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Surface integration of differential forms: analytic + cross-backend parity.

The integrand is exact (closed-form field derivatives + exact forward-mode chart
Jacobian); the integral is Gauss-Legendre quadrature. Locked down with:

1. the pure-Python pullback core (a hand-checked minor determinant);
2. elementary ``integrate_form_values``: ``dx^dy`` -> 1 and ``x dx^dy`` -> 1/2 on
   the unit square; the closed 1-form ``-y dx + x dy`` line integral -> ``2 pi R``;
3. ``surface_area`` of the unit sphere -> ``4 pi`` and ``surface_integral(f=1) ==
   surface_area``;
4. Green's theorem as a Stokes self-test: the boundary line integral equals the
   interior area integral of ``d omega`` (both the analytic low-level form and the
   closed-form ``exterior_derivative`` path);
5. torch vs jax parity at ``rtol=1e-9`` in float64.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from omnibias.fields._core.coords import CoordinateSpec
from omnibias.fields._core.quadrature import gauss_legendre
from omnibias.fields.jax.ops.integral import quadrature_nodes as qnodes_j
from omnibias.fields.torch.ops.integral import quadrature_nodes as qnodes_t
from omnibias.geometry import ChartSpec, DifferentialForm
from omnibias.geometry._core.integration_core import pullback_form_components
from omnibias.geometry.jax import ops as jgeo
from omnibias.geometry.torch import ops as tgeo

R_DISK = 1.5


# ---------------------------------------------------------------------- #
# chart builders (backend-parametrised)
# ---------------------------------------------------------------------- #
def _sphere_phi(xp):  # type: ignore[no-untyped-def]
    def phi(x):  # (theta, phi) -> unit S^2 in R^3
        th, ph = x[0], x[1]
        return xp.stack([xp.sin(th) * xp.cos(ph), xp.sin(th) * xp.sin(ph), xp.cos(th)])

    return phi


def _sphere_chart(xp):  # type: ignore[no-untyped-def]
    return ChartSpec(phi=_sphere_phi(xp), domain_dim=2, ambient_dim=3, name="sphere_S2")


def _ident2_chart(xp):  # type: ignore[no-untyped-def]
    def phi(x):
        return x

    return ChartSpec(phi=phi, domain_dim=2, ambient_dim=2, name="id2")


def _circle_chart(xp, radius):  # type: ignore[no-untyped-def]
    def phi(x):  # theta -> (R cos, R sin) in R^2
        th = x[0]
        return xp.stack([radius * xp.cos(th), radius * xp.sin(th)])

    return ChartSpec(phi=phi, domain_dim=1, ambient_dim=2, name="circle")


def _polar_chart(xp):  # type: ignore[no-untyped-def]
    def phi(x):  # (r, theta) -> (r cos, r sin) in R^2
        r, th = x[0], x[1]
        return xp.stack([r * xp.cos(th), r * xp.sin(th)])

    return ChartSpec(phi=phi, domain_dim=2, ambient_dim=2, name="polar")


# ---------------------------------------------------------------------- #
# small helpers
# ---------------------------------------------------------------------- #
def _t0():  # type: ignore[no-untyped-def]
    return torch.zeros(1, dtype=torch.float64)


def _j0():  # type: ignore[no-untyped-def]
    return jnp.zeros(1, dtype=jnp.float64)


def _np(x):  # type: ignore[no-untyped-def]
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def _jac_t(phi, x):  # type: ignore[no-untyped-def]
    return torch.vmap(torch.func.jacfwd(phi))(x)


def _jac_j(phi, x):  # type: ignore[no-untyped-def]
    return jax.vmap(jax.jacfwd(phi))(x)


# ---------------------------------------------------------------------- #
# 1. Pure-Python pullback core
# ---------------------------------------------------------------------- #
def test_pullback_core_minor_determinant():  # type: ignore[no-untyped-def]
    # ambient R^3 -> domain R^2, a single 2-form component omega_{01} = 1.
    jac = np.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])  # (Q=1, n=3, d=2)
    out = pullback_form_components({(0, 1): np.array([1.0])}, jac, 2, 2, 3)
    # minor rows (0,1) cols (0,1): det([[1,2],[3,4]]) = -2
    assert np.allclose(out[(0, 1)], -2.0)


def test_pullback_core_one_form_columns():  # type: ignore[no-untyped-def]
    jac = np.array([[[0.5, -1.0], [2.0, 3.0]]])  # (1, 2, 2)
    out = pullback_form_components(
        {(0,): np.array([10.0]), (1,): np.array([100.0])}, jac, 1, 2, 2,
    )
    assert np.allclose(out[(0,)], 10.0 * 0.5 + 100.0 * 2.0)  # 205
    assert np.allclose(out[(1,)], 10.0 * -1.0 + 100.0 * 3.0)  # 290


def test_pullback_core_higher_degree_is_empty():  # type: ignore[no-untyped-def]
    # a 2-form pulls back to zero on a 1-dimensional domain.
    jac = np.zeros((1, 2, 1))
    assert pullback_form_components({(0, 1): np.array([1.0])}, jac, 2, 1, 2) == {}


# ---------------------------------------------------------------------- #
# 2. Elementary integrate_form_values
# ---------------------------------------------------------------------- #
def test_unit_square_two_form_torch():  # type: ignore[no-untyped-def]
    rule = gauss_legendre([(0.0, 1.0), (0.0, 1.0)], 8)
    x = qnodes_t(rule, like=_t0())
    jac = _jac_t(lambda y: y, x)
    one = torch.ones(x.shape[0], dtype=torch.float64)
    i_one = tgeo.integrate_form_values({(0, 1): one}, 2, jac, rule=rule)
    i_x = tgeo.integrate_form_values({(0, 1): x[:, 0]}, 2, jac, rule=rule)
    assert math.isclose(float(i_one), 1.0, abs_tol=1e-12)
    assert math.isclose(float(i_x), 0.5, abs_tol=1e-12)


def test_unit_square_two_form_jax():  # type: ignore[no-untyped-def]
    rule = gauss_legendre([(0.0, 1.0), (0.0, 1.0)], 8)
    x = qnodes_j(rule, like=_j0())
    jac = _jac_j(lambda y: y, x)
    one = jnp.ones(x.shape[0], dtype=jnp.float64)
    i_one = jgeo.integrate_form_values({(0, 1): one}, 2, jac, rule=rule)
    i_x = jgeo.integrate_form_values({(0, 1): x[:, 0]}, 2, jac, rule=rule)
    assert math.isclose(float(i_one), 1.0, abs_tol=1e-12)
    assert math.isclose(float(i_x), 0.5, abs_tol=1e-12)


def test_closed_one_form_line_integral_torch():  # type: ignore[no-untyped-def]
    # circulation of omega = -y dx + x dy around the circle of radius R is 2 pi R^2.
    r = R_DISK
    chart = _circle_chart(torch, r)
    rule = gauss_legendre([(0.0, 2.0 * math.pi)], 128)
    x = qnodes_t(rule, like=_t0())
    jac = _jac_t(chart.phi, x)
    theta = x[:, 0]
    px = -r * torch.sin(theta)  # P = -y on the circle
    qy = r * torch.cos(theta)   # Q =  x on the circle
    circ = tgeo.integrate_form_values({(0,): px, (1,): qy}, 1, jac, rule=rule)
    assert math.isclose(float(circ), 2.0 * math.pi * r**2, rel_tol=1e-10)


# ---------------------------------------------------------------------- #
# 3. Sphere area + surface integral of a scalar
# ---------------------------------------------------------------------- #
def _sphere_rule():  # type: ignore[no-untyped-def]
    return gauss_legendre([(0.0, math.pi), (0.0, 2.0 * math.pi)], 64)


def test_sphere_surface_area_torch():  # type: ignore[no-untyped-def]
    area = tgeo.surface_area(_sphere_chart(torch), _sphere_rule(), like=_t0())
    assert math.isclose(float(area), 4.0 * math.pi, rel_tol=1e-10)


def test_sphere_surface_area_jax():  # type: ignore[no-untyped-def]
    area = jgeo.surface_area(_sphere_chart(jnp), _sphere_rule(), like=_j0())
    assert math.isclose(float(area), 4.0 * math.pi, rel_tol=1e-10)


def _const_field_r3(builders, ops_module):  # type: ignore[no-untyped-def]
    Const1D = builders["Const1D"]
    coord3 = CoordinateSpec(("x", "y", "z"), time_axis=None)
    comp_axes = {"one": (Const1D(1.0), Const1D(1.0), Const1D(1.0))}
    return builders["AnalyticField"](
        coord3, builders["ComponentSpec"](("one",)), comp_axes, ops_module,
    )


def test_surface_integral_of_one_equals_area_torch(builders):  # type: ignore[no-untyped-def]
    from omnibias.fields.torch import _ops_dispatch

    chart = _sphere_chart(torch)
    rule = _sphere_rule()
    x = qnodes_t(rule, like=_t0())
    image = torch.vmap(chart.phi)(x)
    state = _const_field_r3(builders, _ops_dispatch)(image)
    si = tgeo.surface_integral(state, "one", chart, rule)
    area = tgeo.surface_area(chart, rule, like=_t0())
    assert math.isclose(float(si), float(area), rel_tol=1e-12)
    assert math.isclose(float(si), 4.0 * math.pi, rel_tol=1e-10)


def test_surface_integral_of_one_equals_area_jax(builders):  # type: ignore[no-untyped-def]
    from omnibias.fields.jax import _ops_dispatch

    chart = _sphere_chart(jnp)
    rule = _sphere_rule()
    x = qnodes_j(rule, like=_j0())
    image = jax.vmap(chart.phi)(x)
    state = _const_field_r3(builders, _ops_dispatch)(image)
    si = jgeo.surface_integral(state, "one", chart, rule)
    assert math.isclose(float(si), 4.0 * math.pi, rel_tol=1e-10)


# ---------------------------------------------------------------------- #
# 4. Green's theorem (Stokes self-test) on a disk of radius R
# ---------------------------------------------------------------------- #
def _omega_form():  # type: ignore[no-untyped-def]
    # omega = P dx0 + Q dx1  with P = -x1, Q = x0  ->  d omega = 2 dx0 ^ dx1
    return DifferentialForm(degree=1, dim=2, comps={(0,): "P", (1,): "Q"})


def _green_field(builders, ops_module):  # type: ignore[no-untyped-def]
    Poly1D, Const1D = builders["Poly1D"], builders["Const1D"]
    comp_axes = {
        "P": (Const1D(1.0), Poly1D((0.0, -1.0))),  # 1 * (-x1) = -x1
        "Q": (Poly1D((0.0, 1.0)), Const1D(1.0)),   # x0 * 1  =  x0
    }
    return builders["AnalyticField"](
        builders["coord_spec_2d"](), builders["ComponentSpec"](("P", "Q")),
        comp_axes, ops_module,
    )


def test_green_theorem_low_level_torch():  # type: ignore[no-untyped-def]
    r = R_DISK
    expected = 2.0 * math.pi * r**2

    # boundary: integral over the circle of the analytic 1-form omega.
    bchart = _circle_chart(torch, r)
    brule = gauss_legendre([(0.0, 2.0 * math.pi)], 128)
    bx = qnodes_t(brule, like=_t0())
    bjac = _jac_t(bchart.phi, bx)
    theta = bx[:, 0]
    boundary = tgeo.integrate_form_values(
        {(0,): -r * torch.sin(theta), (1,): r * torch.cos(theta)}, 1, bjac, rule=brule,
    )

    # interior: integral over the disk of d omega = 2 dx0 ^ dx1.
    ichart = _polar_chart(torch)
    irule = gauss_legendre([(0.0, r), (0.0, 2.0 * math.pi)], 32)
    ix = qnodes_t(irule, like=_t0())
    ijac = _jac_t(ichart.phi, ix)
    two = 2.0 * torch.ones(ix.shape[0], dtype=torch.float64)
    interior = tgeo.integrate_form_values({(0, 1): two}, 2, ijac, rule=irule)

    assert math.isclose(float(boundary), expected, rel_tol=1e-10)
    assert math.isclose(float(interior), expected, rel_tol=1e-10)
    assert math.isclose(float(boundary), float(interior), rel_tol=1e-10)


def test_green_theorem_exterior_derivative_torch(builders):  # type: ignore[no-untyped-def]
    from omnibias.fields.torch import _ops_dispatch

    r = R_DISK
    form = _omega_form()

    # boundary via the name-form integrate_form (field evaluated at circle image).
    bchart = _circle_chart(torch, r)
    brule = gauss_legendre([(0.0, 2.0 * math.pi)], 128)
    bx = qnodes_t(brule, like=_t0())
    bimg = torch.vmap(bchart.phi)(bx)
    bstate = _green_field(builders, _ops_dispatch)(bimg)
    boundary = tgeo.integrate_form(bstate, form, bchart, brule)

    # interior via closed-form exterior_derivative(omega) integrated over the disk.
    ichart = _polar_chart(torch)
    irule = gauss_legendre([(0.0, r), (0.0, 2.0 * math.pi)], 32)
    ix = qnodes_t(irule, like=_t0())
    iimg = torch.vmap(ichart.phi)(ix)
    istate = _green_field(builders, _ops_dispatch)(iimg)
    domega = tgeo.exterior_derivative(istate, form)  # {(0,1): ~2}
    ijac = _jac_t(ichart.phi, ix)
    interior = tgeo.integrate_form_values(domega, 2, ijac, rule=irule)

    assert math.isclose(float(boundary), 2.0 * math.pi * r**2, rel_tol=1e-10)
    assert math.isclose(float(boundary), float(interior), rel_tol=1e-9)


# ---------------------------------------------------------------------- #
# 5. Cross-backend parity (rtol = 1e-9, float64)
# ---------------------------------------------------------------------- #
def test_parity_surface_area():  # type: ignore[no-untyped-def]
    a_t = float(tgeo.surface_area(_sphere_chart(torch), _sphere_rule(), like=_t0()))
    a_j = float(jgeo.surface_area(_sphere_chart(jnp), _sphere_rule(), like=_j0()))
    assert np.allclose(a_t, a_j, rtol=1e-9, atol=1e-9)


def test_parity_volume_element():  # type: ignore[no-untyped-def]
    rule = _sphere_rule()
    v_t = _np(tgeo.volume_element(_sphere_chart(torch), rule, like=_t0()))
    v_j = _np(jgeo.volume_element(_sphere_chart(jnp), rule, like=_j0()))
    assert np.allclose(v_t, v_j, rtol=1e-9, atol=1e-9)


def test_parity_integrate_form_values_square():  # type: ignore[no-untyped-def]
    rule = gauss_legendre([(0.0, 1.0), (0.0, 1.0)], 8)
    xt = qnodes_t(rule, like=_t0())
    xj = qnodes_j(rule, like=_j0())
    it = float(
        tgeo.integrate_form_values({(0, 1): xt[:, 0]}, 2, _jac_t(lambda y: y, xt), rule=rule),
    )
    ij = float(
        jgeo.integrate_form_values({(0, 1): xj[:, 0]}, 2, _jac_j(lambda y: y, xj), rule=rule),
    )
    assert np.allclose(it, ij, rtol=1e-9, atol=1e-9)


# ---------------------------------------------------------------------- #
# 6. Error guards
# ---------------------------------------------------------------------- #
def test_integrate_form_values_degree_mismatch_raises():  # type: ignore[no-untyped-def]
    rule = gauss_legendre([(0.0, 1.0), (0.0, 1.0)], 4)
    x = qnodes_t(rule, like=_t0())
    jac = _jac_t(lambda y: y, x)  # (Q, 2, 2) -> domain dim 2
    with pytest.raises(ValueError, match="must equal the submanifold"):
        tgeo.integrate_form_values({(0,): x[:, 0]}, 1, jac, rule=rule)


def test_integrate_form_degree_mismatch_raises(builders):  # type: ignore[no-untyped-def]
    from omnibias.fields.torch import _ops_dispatch

    # a 1-form over a 2-dimensional chart is not top-degree -> reject.
    chart = _ident2_chart(torch)
    rule = gauss_legendre([(0.0, 1.0), (0.0, 1.0)], 4)
    x = qnodes_t(rule, like=_t0())
    state = _green_field(builders, _ops_dispatch)(torch.vmap(chart.phi)(x))
    with pytest.raises(ValueError, match="top-degree"):
        tgeo.integrate_form(state, _omega_form(), chart, rule)
