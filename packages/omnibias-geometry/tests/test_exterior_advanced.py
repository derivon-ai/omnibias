# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Phase 4 exterior calculus: interior product, Lie derivative, general codifferential.

Validated with closed-form field derivatives plus the analytic metric, against
independent coordinate formulas / Cartan identities, and torch<->jax parity.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
import torch
from omnibias.fields._core.coords import CoordinateSpec
from omnibias.geometry import DifferentialForm, ManifoldSpec, MetricSpec
from omnibias.geometry.jax import ops as jgeo
from omnibias.geometry.torch import ops as tgeo

COORDS2 = np.array([[0.7, 0.3], [1.1, 1.5], [1.9, 2.2], [2.4, 4.0]], dtype=np.float64)
COORDS3 = np.array(
    [[0.4, -0.2, 0.5], [0.9, 0.3, -0.7], [-0.6, 1.1, 0.2], [1.3, -0.8, 0.6]],
    dtype=np.float64,
)


def _np(x):  # type: ignore[no-untyped-def]
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def _coords(backend, arr):  # type: ignore[no-untyped-def]
    return (
        torch.as_tensor(arr, dtype=torch.float64) if backend == "torch"
        else jnp.asarray(arr, dtype=jnp.float64)
    )


def _ops(backend):  # type: ignore[no-untyped-def]
    if backend == "torch":
        from omnibias.fields.torch import _ops_dispatch as o
    else:
        from omnibias.fields.jax import _ops_dispatch as o
    return o


def _field_2d(builders, backend, comp_axes):  # type: ignore[no-untyped-def]
    return builders["AnalyticField"](
        builders["coord_spec_2d"](), builders["ComponentSpec"](tuple(comp_axes)),
        comp_axes, _ops(backend),
    )


def _field_3d(builders, backend, comp_axes):  # type: ignore[no-untyped-def]
    coord = CoordinateSpec(("x", "y", "z"), time_axis=None)
    return builders["AnalyticField"](
        coord, builders["ComponentSpec"](tuple(comp_axes)), comp_axes, _ops(backend),
    )


def _sphere(builders, xp, stack):  # type: ignore[no-untyped-def]
    g = builders["sphere_metric_factory"](xp, stack)
    return ManifoldSpec("S2", 2, MetricSpec(g, dim=2))


def _flat3(builders, xp, stack):  # type: ignore[no-untyped-def]
    g = builders["flat_metric_factory"](xp, stack, 3)
    return ManifoldSpec("E3", 3, MetricSpec(g, dim=3))


# ============================ interior product ============================


def test_interior_product_value_one_form():
    from omnibias.geometry._core.forms import interior_product

    rng = np.random.default_rng(0)
    b = 5
    omega = {(i,): torch.as_tensor(rng.normal(size=b)) for i in range(3)}
    x = torch.as_tensor(rng.normal(size=(b, 3)))
    got = interior_product(x, omega, 1, 3)[()]
    exp = sum(x[:, j] * omega[(j,)] for j in range(3))
    assert torch.allclose(got, exp, atol=1e-12)


def test_interior_product_zero_form_is_empty():
    from omnibias.geometry._core.forms import interior_product

    x = torch.ones((4, 3), dtype=torch.float64)
    assert interior_product(x, {(): torch.ones(4)}, 0, 3) == {}


def test_interior_product_nilpotent():
    """iota_X iota_X omega = 0 for any form (anti-derivation, degree -1)."""
    from omnibias.geometry._core.forms import interior_product

    rng = np.random.default_rng(2)
    b = 6
    omega2 = {idx: torch.as_tensor(rng.normal(size=b)) for idx in [(0, 1), (0, 2), (1, 2)]}
    x = torch.as_tensor(rng.normal(size=(b, 3)))
    once = interior_product(x, omega2, 2, 3)
    twice = interior_product(x, once, 1, 3)
    assert torch.allclose(twice[()], torch.zeros(b, dtype=torch.float64), atol=1e-12)


# ============================== Lie derivative ============================


def _lie_field(builders, backend):  # type: ignore[no-untyped-def]
    P = builders["Poly1D"]
    comp_axes = {
        "f": (P((0.2, 1.0, -0.5)), P((1.0, 0.4, -0.7))),
        "X0": (P((0.3, -0.6)), P((1.0, 0.2))),
        "X1": (P((1.0, 0.5)), P((0.1, -0.4))),
        "w0": (P((0.5, 1.0, 0.2)), P((1.0, -0.3))),
        "w1": (P((1.0, 0.1)), P((0.4, 0.7, -0.2))),
    }
    return _field_2d(builders, backend, comp_axes)


def test_lie_derivative_zero_form_is_directional(builders):  # type: ignore[no-untyped-def]
    st = _lie_field(builders, "torch")(_coords("torch", COORDS2))
    form = DifferentialForm(0, 2, {(): "f"})
    got = _np(tgeo.lie_derivative(st, ("X0", "X1"), form)[()])
    exp = sum(
        _np(st.ops.value(st, f"X{m}")) * _np(st.ops.derivative(st, "f", axis=m, order=1))
        for m in range(2)
    )
    assert np.allclose(got, exp, rtol=1e-12, atol=1e-12)


def test_lie_derivative_zero_form_matches_cartan(builders):  # type: ignore[no-untyped-def]
    """0-form Cartan: L_X f = iota_X(df) (since iota_X f = 0)."""
    from omnibias.geometry._core.forms import interior_product

    st = _lie_field(builders, "torch")(_coords("torch", COORDS2))
    form = DifferentialForm(0, 2, {(): "f"})
    lx = _np(tgeo.lie_derivative(st, ("X0", "X1"), form)[()])
    df = tgeo.exterior_derivative(st, form)  # evaluated 1-form
    xvec = torch.stack([st.ops.value(st, "X0"), st.ops.value(st, "X1")], dim=-1)
    cartan = _np(interior_product(xvec, df, 1, 2)[()])
    assert np.allclose(lx, cartan, rtol=1e-12, atol=1e-12)


def test_lie_derivative_one_form_matches_coordinate_formula(builders):  # type: ignore[no-untyped-def]
    st = _lie_field(builders, "torch")(_coords("torch", COORDS2))
    form = DifferentialForm(1, 2, {(0,): "w0", (1,): "w1"})
    got = tgeo.lie_derivative(st, ("X0", "X1"), form)
    names = {0: "w0", 1: "w1"}
    xn = ("X0", "X1")
    for i in range(2):
        # (L_X w)_i = X^m d_m w_i + (d_i X^m) w_m
        exp = sum(
            _np(st.ops.value(st, f"X{m}"))
            * _np(st.ops.derivative(st, names[i], axis=m, order=1))
            for m in range(2)
        )
        exp = exp + sum(
            _np(st.ops.derivative(st, xn[m], axis=i, order=1))
            * _np(st.ops.value(st, names[m]))
            for m in range(2)
        )
        assert np.allclose(_np(got[(i,)]), exp, rtol=1e-12, atol=1e-12)


# ============================== codifferential ===========================


def _codiff_1form_field(builders, backend):  # type: ignore[no-untyped-def]
    P = builders["Poly1D"]
    C = builders["Cos1D"]
    xp = torch if backend == "torch" else jnp
    comp_axes = {
        "w0": (C(0.7, 1.0, xp), P((1.0, 0.3, -0.2))),
        "w1": (P((0.5, 1.0, 0.1)), C(0.9, 1.0, xp)),
    }
    return _field_2d(builders, backend, comp_axes)


def test_codifferential_one_form_curved_matches_compact_formula(builders):  # type: ignore[no-untyped-def]
    """delta(omega) for a 1-form via the covariant formula equals the compact
    -(A^nu w_nu + g^{mu nu} d_mu w_nu), computed through metric_density_divergence."""
    st = _codiff_1form_field(builders, "torch")(_coords("torch", COORDS2))
    m = _sphere(builders, torch, torch.stack)
    got = _np(tgeo.codifferential(st, DifferentialForm(1, 2, {(0,): "w0", (1,): "w1"}), m)[()])
    coords = _coords("torch", COORDS2)
    a_vec = _np(tgeo.metric_density_divergence(coords, m))  # (B, 2)
    ginv = _np(tgeo.inverse_metric(coords, m))  # (B, 2, 2)
    names = {0: "w0", 1: "w1"}
    term_a = sum(a_vec[:, nu] * _np(st.ops.value(st, names[nu])) for nu in range(2))
    term_g = np.zeros(COORDS2.shape[0])
    for mu in range(2):
        for nu in range(2):
            term_g = term_g + ginv[:, mu, nu] * _np(
                st.ops.derivative(st, names[nu], axis=mu, order=1)
            )
    assert np.allclose(got, -(term_a + term_g), rtol=1e-9, atol=1e-9)


def _codiff_2form_field(builders, backend):  # type: ignore[no-untyped-def]
    P = builders["Poly1D"]
    comp_axes = {
        "w01": (P((0.5, 1.0)), P((1.0, 0.3)), P((0.2, -0.4, 0.1))),
        "w02": (P((1.0, 0.2, 0.1)), P((0.5, 1.0)), P((1.0, -0.3))),
        "w12": (P((0.3, -0.7)), P((1.0, 0.4, 0.2)), P((0.6, 1.0))),
    }
    return _field_3d(builders, backend, comp_axes)


def test_codifferential_two_form_flat_matches_divergence(builders):  # type: ignore[no-untyped-def]
    """On flat E^3, (delta omega)_i = -sum_j d_j omega_{ji} for a 2-form."""
    st = _codiff_2form_field(builders, "torch")(_coords("torch", COORDS3))
    m = _flat3(builders, torch, torch.stack)
    form = DifferentialForm(2, 3, {(0, 1): "w01", (0, 2): "w02", (1, 2): "w12"})
    got = tgeo.codifferential(st, form, m)
    # delta_0 = d1 w01 + d2 w02 ; delta_1 = -d0 w01 + d2 w12 ; delta_2 = -d0 w02 - d1 w12
    d = st.ops.derivative
    exp0 = _np(d(st, "w01", axis=1, order=1)) + _np(d(st, "w02", axis=2, order=1))
    exp1 = -_np(d(st, "w01", axis=0, order=1)) + _np(d(st, "w12", axis=2, order=1))
    exp2 = -_np(d(st, "w02", axis=0, order=1)) - _np(d(st, "w12", axis=1, order=1))
    assert np.allclose(_np(got[(0,)]), exp0, rtol=1e-11, atol=1e-11)
    assert np.allclose(_np(got[(1,)]), exp1, rtol=1e-11, atol=1e-11)
    assert np.allclose(_np(got[(2,)]), exp2, rtol=1e-11, atol=1e-11)


# ============================ cross-backend parity =======================


def test_lie_derivative_parity(builders):  # type: ignore[no-untyped-def]
    ts = _lie_field(builders, "torch")(_coords("torch", COORDS2))
    js = _lie_field(builders, "jax")(_coords("jax", COORDS2))
    form = DifferentialForm(1, 2, {(0,): "w0", (1,): "w1"})
    t = tgeo.lie_derivative(ts, ("X0", "X1"), form)
    j = jgeo.lie_derivative(js, ("X0", "X1"), form)
    for idx in t:
        assert np.allclose(_np(t[idx]), _np(j[idx]), rtol=1e-11, atol=1e-11)


def test_codifferential_parity(builders):  # type: ignore[no-untyped-def]
    ts = _codiff_2form_field(builders, "torch")(_coords("torch", COORDS3))
    js = _codiff_2form_field(builders, "jax")(_coords("jax", COORDS3))
    mt = _flat3(builders, torch, torch.stack)
    mj = _flat3(builders, jnp, jnp.stack)
    form = DifferentialForm(2, 3, {(0, 1): "w01", (0, 2): "w02", (1, 2): "w12"})
    t = tgeo.codifferential(ts, form, mt)
    j = jgeo.codifferential(js, form, mj)
    for idx in t:
        assert np.allclose(_np(t[idx]), _np(j[idx]), rtol=1e-11, atol=1e-11)
