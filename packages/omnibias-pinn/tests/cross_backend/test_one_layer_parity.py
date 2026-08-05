# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Bit-parity: torch ``OneLayerVectorField`` vs JAX ``OneLayerVectorField``.

Both backends share the same closed-form math; the field is parameterised
by identical numpy arrays, so the per-op outputs must match to
``rtol=1e-12, atol=1e-12`` in float64. We sweep over all five Riccati-
class activations.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import torch
from omnibias.jax.activations import get_activation as jax_get_activation
from omnibias.pinn import ComponentSpec, CoordinateSpec
from omnibias.pinn.jax import ops as jops
from omnibias.pinn.jax.fields.one_layer import OneLayerVectorField as JaxField
from omnibias.pinn.torch import ops as tops
from omnibias.pinn.torch.fields import OneLayerVectorField as TorchField


def _make_torch(shared, activation: str):
    coord_spec = CoordinateSpec(("x", "y", "t"))
    comp_spec = ComponentSpec(
        ("u", "v", "p"), groups={"velocity": ("u", "v")},
    )
    field = TorchField(
        coordinate_spec=coord_spec,
        components=comp_spec,
        hidden=shared["H"],
        base=activation,
        dtype=torch.float64,
    )
    with torch.no_grad():
        field.W.weight.copy_(torch.from_numpy(shared["W"]))
        field.W.bias.copy_(torch.from_numpy(shared["beta"]))
        field.c.weight.copy_(torch.from_numpy(shared["c"]))
        field.c.bias.copy_(torch.from_numpy(shared["b"]))
    coords = torch.from_numpy(shared["coords"])
    return field, coords


def _make_jax(shared, activation: str):
    coord_spec = CoordinateSpec(("x", "y", "t"))
    comp_spec = ComponentSpec(
        ("u", "v", "p"), groups={"velocity": ("u", "v")},
    )
    spec = jax_get_activation(activation)
    field = JaxField(
        coordinate_spec=coord_spec,
        components=comp_spec,
        spec=spec,
        W=jnp.asarray(shared["W"]),
        beta=jnp.asarray(shared["beta"]),
        c=jnp.asarray(shared["c"]),
        b=jnp.asarray(shared["b"]),
        hidden=shared["H"],
    )
    coords = jnp.asarray(shared["coords"])
    return field, coords


def _allclose(t: torch.Tensor, j, *, rtol: float = 1e-12, atol: float = 1e-12) -> bool:
    return np.allclose(
        t.detach().cpu().numpy(), np.asarray(j), rtol=rtol, atol=atol,
    )


def test_parity_value(riccati, shared_params):
    tf, tc = _make_torch(shared_params, riccati)
    jf, jc = _make_jax(shared_params, riccati)
    ts = tf(tc)
    js = jf(jc)
    for n in ("u", "v", "p"):
        assert _allclose(ts[n].value, js[n].value), f"value mismatch for {n!r}"


def test_parity_first_partials(riccati, shared_params):
    tf, tc = _make_torch(shared_params, riccati)
    jf, jc = _make_jax(shared_params, riccati)
    ts = tf(tc)
    js = jf(jc)
    for n in ("u", "v", "p"):
        for axis in ("x", "y", "t"):
            t_val = tops.derivative(ts, n, axis=axis)
            j_val = jops.derivative(js, n, axis=axis)
            assert _allclose(t_val, j_val), (
                f"derivative({n}, {axis}) mismatch under {riccati}"
            )


def test_parity_higher_order_pure(riccati, shared_params):
    tf, tc = _make_torch(shared_params, riccati)
    jf, jc = _make_jax(shared_params, riccati)
    ts = tf(tc)
    js = jf(jc)
    for order in (2, 3, 4):
        t_val = tops.derivative(ts, "u", axis="x", order=order)
        j_val = jops.derivative(js, "u", axis="x", order=order)
        assert _allclose(t_val, j_val), (
            f"d^{order} u / dx^{order} mismatch under {riccati}"
        )


def test_parity_mixed_partials(riccati, shared_params):
    tf, tc = _make_torch(shared_params, riccati)
    jf, jc = _make_jax(shared_params, riccati)
    ts = tf(tc)
    js = jf(jc)
    for axes, orders in [
        (("x", "y"), (1, 1)),
        (("x", "y"), (2, 1)),
        (("x", "t"), (2, 1)),
        (("x", "y", "t"), (1, 1, 1)),
    ]:
        t_val = tops.mixed_partial(ts, "u", axes, orders)
        j_val = jops.mixed_partial(js, "u", axes, orders)
        assert _allclose(t_val, j_val), (
            f"mixed_partial({axes}, {orders}) mismatch under {riccati}"
        )


def test_parity_gradient(riccati, shared_params):
    tf, tc = _make_torch(shared_params, riccati)
    jf, jc = _make_jax(shared_params, riccati)
    ts = tf(tc)
    js = jf(jc)
    for n in ("u", "v"):
        assert _allclose(tops.gradient(ts, n), jops.gradient(js, n))
        assert _allclose(
            tops.gradient(ts, n, axes=("x", "y", "t")),
            jops.gradient(js, n, axes=("x", "y", "t")),
        )


def test_parity_laplacian_and_biharmonic(riccati, shared_params):
    tf, tc = _make_torch(shared_params, riccati)
    jf, jc = _make_jax(shared_params, riccati)
    ts = tf(tc)
    js = jf(jc)
    for n in ("u", "v"):
        assert _allclose(tops.laplacian(ts, n), jops.laplacian(js, n))
        assert _allclose(tops.biharmonic(ts, n), jops.biharmonic(js, n))


def test_parity_polylaplacian(riccati, shared_params):
    tf, tc = _make_torch(shared_params, riccati)
    jf, jc = _make_jax(shared_params, riccati)
    ts = tf(tc)
    js = jf(jc)
    for k in (1, 2, 3, 4):
        t_val = tops.polylaplacian(ts, "u", k=k)
        j_val = jops.polylaplacian(js, "u", k=k)
        # Order 2k = 8 for k=4 is at the float64 edge of the activation
        # derivative recurrence; use a loose-but-still-tight tolerance for
        # the highest order.
        rtol = 1e-12 if k <= 2 else (1e-11 if k == 3 else 1e-9)
        atol = rtol
        assert _allclose(t_val, j_val, rtol=rtol, atol=atol), (
            f"polylap k={k} under {riccati}"
        )


def test_parity_hessian(riccati, shared_params):
    tf, tc = _make_torch(shared_params, riccati)
    jf, jc = _make_jax(shared_params, riccati)
    ts = tf(tc)
    js = jf(jc)
    assert _allclose(tops.hessian(ts, "u"), jops.hessian(js, "u"))
    assert _allclose(
        tops.hessian(ts, "u", axes=("x", "y")),
        jops.hessian(js, "u", axes=("x", "y")),
    )
    assert _allclose(tops.spatial_hessian(ts, "u"), jops.spatial_hessian(js, "u"))
    assert _allclose(
        tops.gradient_of_derivative(ts, "u", axis="x"),
        jops.gradient_of_derivative(js, "u", axis="x"),
    )


def test_parity_jacobian_divergence_curl(riccati, shared_params):
    tf, tc = _make_torch(shared_params, riccati)
    jf, jc = _make_jax(shared_params, riccati)
    ts = tf(tc)
    js = jf(jc)
    assert _allclose(tops.jacobian(ts, ("u", "v")), jops.jacobian(js, ("u", "v")))
    assert _allclose(
        tops.spatial_jacobian(ts, ("u", "v", "p")),
        jops.spatial_jacobian(js, ("u", "v", "p")),
    )
    assert _allclose(tops.divergence(ts, ("u", "v")), jops.divergence(js, ("u", "v")))
    assert _allclose(tops.curl(ts, ("u", "v")), jops.curl(js, ("u", "v")))


def test_parity_advection_and_material_derivative(riccati, shared_params):
    tf, tc = _make_torch(shared_params, riccati)
    jf, jc = _make_jax(shared_params, riccati)
    ts = tf(tc)
    js = jf(jc)
    assert _allclose(
        tops.advection(ts, velocity=("u", "v")),
        jops.advection(js, velocity=("u", "v")),
    )
    assert _allclose(
        tops.advection(ts, velocity=("u", "v"), scalar="p"),
        jops.advection(js, velocity=("u", "v"), scalar="p"),
    )
    assert _allclose(
        tops.material_derivative(ts, velocity=("u", "v")),
        jops.material_derivative(js, velocity=("u", "v")),
    )


def test_parity_p_laplacian(riccati, shared_params):
    tf, tc = _make_torch(shared_params, riccati)
    jf, jc = _make_jax(shared_params, riccati)
    ts = tf(tc)
    js = jf(jc)
    for p in (1.5, 2.0, 3.0, 4.0):
        t_val = tops.p_laplacian(ts, "u", p=p)
        j_val = jops.p_laplacian(js, "u", p=p)
        assert _allclose(t_val, j_val), f"p-Laplacian p={p} under {riccati}"


def test_parity_strain_rate_deformation_gradient(riccati, shared_params):
    tf, tc = _make_torch(shared_params, riccati)
    jf, jc = _make_jax(shared_params, riccati)
    ts = tf(tc)
    js = jf(jc)
    assert _allclose(
        tops.strain_rate(ts, ("u", "v")),
        jops.strain_rate(js, ("u", "v")),
    )
    assert _allclose(
        tops.deformation_gradient(ts, ("u", "v")),
        jops.deformation_gradient(js, ("u", "v")),
    )
    assert _allclose(
        tops.vector_hessian(ts, ("u", "v"), axes=("x", "y")),
        jops.vector_hessian(js, ("u", "v"), axes=("x", "y")),
    )
    assert _allclose(
        tops.vector_polylaplacian(ts, ("u", "v"), k=2),
        jops.vector_polylaplacian(js, ("u", "v"), k=2),
    )
