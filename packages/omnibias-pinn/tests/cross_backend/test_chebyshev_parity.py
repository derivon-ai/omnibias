# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Bit-parity: torch ``ChebyshevVectorField`` vs JAX ``ChebyshevVectorField``."""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest
import torch
from omnibias.jax.activations import get_activation as jax_get_activation
from omnibias.pinn._core.components import ComponentSpec
from omnibias.pinn._core.coords import CoordinateSpec
from omnibias.pinn.jax import ops as jops
from omnibias.pinn.jax.fields.chebyshev import ChebyshevVectorField as JaxField
from omnibias.pinn.jax.fields.chebyshev import (
    _chebyshev_differentiation_matrix as _jax_D,
)
from omnibias.pinn.torch import ops as tops
from omnibias.pinn.torch.fields.chebyshev import ChebyshevVectorField as TorchField

_RICCATI = ("tanh", "sigmoid", "softplus", "gaussian", "exp")


@pytest.fixture(params=_RICCATI)
def riccati(request) -> str:
    return request.param


@pytest.fixture(params=[1, 2, 3])
def D_spatial(request) -> int:
    return request.param


def _build_shared_params(D_spatial: int, K: int = 4, time_hidden: int = 6):
    rng = np.random.default_rng(2026)
    C = 2 if D_spatial == 1 else (3 if D_spatial == 2 else 4)
    modes = (K + 1) ** D_spatial
    out_dim = C * modes
    W_t = rng.normal(scale=0.5, size=(time_hidden, 1)).astype(np.float64)
    beta_t = rng.normal(scale=0.1, size=(time_hidden,)).astype(np.float64)
    V = rng.normal(scale=0.2, size=(out_dim, time_hidden)).astype(np.float64)
    b_t = rng.normal(scale=0.1, size=(out_dim,)).astype(np.float64)
    if D_spatial == 1:
        coords = np.zeros((6, 2), dtype=np.float64)
        coords[:, 0] = np.linspace(0.0, 1.0, 6)
        coords[:, 1] = np.linspace(-0.8, 0.7, 6)
        axes = ("t", "x")
        domain = ((0.0, 1.0), (-1.0, 1.0))
        names = ("u", "p")
        groups = {}
    elif D_spatial == 2:
        coords = np.zeros((6, 3), dtype=np.float64)
        coords[:, 0] = np.linspace(0.0, 1.0, 6)
        coords[:, 1] = np.linspace(-0.8, 0.7, 6)
        coords[:, 2] = np.linspace(-0.5, 1.4, 6)
        axes = ("t", "x", "y")
        domain = ((0.0, 1.0), (-1.0, 1.0), (-2.0, 2.0))
        names = ("u", "v", "p")
        groups = {"velocity": ("u", "v")}
    else:
        coords = np.zeros((5, 4), dtype=np.float64)
        coords[:, 0] = np.linspace(0.0, 1.0, 5)
        coords[:, 1] = np.linspace(-0.8, 0.7, 5)
        coords[:, 2] = np.linspace(-0.5, 0.4, 5)
        coords[:, 3] = np.linspace(-0.6, 0.3, 5)
        axes = ("t", "x", "y", "z")
        domain = ((0.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))
        names = ("u", "v", "w", "p")
        groups = {"velocity": ("u", "v", "w")}
    return dict(
        W_t=W_t, beta_t=beta_t, V=V, b_t=b_t,
        coords=coords, axes=axes, domain=domain,
        names=names, groups=groups, K=K, time_hidden=time_hidden,
    )


def _make_torch(shared, activation: str):
    cspec = CoordinateSpec(
        axes=shared["axes"], time_axis="t", domain=shared["domain"],
    )
    mspec = ComponentSpec(shared["names"], groups=shared["groups"] or None)
    field = TorchField(
        coordinate_spec=cspec, components=mspec,
        K=shared["K"], time_hidden=shared["time_hidden"],
        time_depth=1, activation=activation,
        dtype=torch.float64,
    )
    with torch.no_grad():
        field.W_t.copy_(torch.from_numpy(shared["W_t"]))
        field.beta_t.copy_(torch.from_numpy(shared["beta_t"]))
        field.V.copy_(torch.from_numpy(shared["V"]))
        field.b_t.copy_(torch.from_numpy(shared["b_t"]))
    coords = torch.from_numpy(shared["coords"])
    return field, coords


def _make_jax(shared, activation: str):
    cspec = CoordinateSpec(
        axes=shared["axes"], time_axis="t", domain=shared["domain"],
    )
    mspec = ComponentSpec(shared["names"], groups=shared["groups"] or None)
    spec = jax_get_activation(activation)
    K = shared["K"]
    spatial_domain = tuple(
        shared["domain"][cspec.axis_index(a)] for a in cspec.spatial_axes
    )
    field = JaxField(
        coordinate_spec=cspec, components=mspec, spec=spec,
        W_t=jnp.asarray(shared["W_t"]),
        beta_t=jnp.asarray(shared["beta_t"]),
        inner_W=tuple(),
        inner_b=tuple(),
        V=jnp.asarray(shared["V"]),
        b_t=jnp.asarray(shared["b_t"]),
        D_mat=_jax_D(K, dtype=jnp.float64),
        K=K,
        domain=spatial_domain,
        time_hidden=shared["time_hidden"], time_depth=1,
    )
    coords = jnp.asarray(shared["coords"])
    return field, coords


def _allclose(t, j, *, rtol: float = 1e-12, atol: float = 1e-12) -> bool:
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu().numpy()
    return np.allclose(np.asarray(t), np.asarray(j), rtol=rtol, atol=atol)


def test_parity_value(riccati, D_spatial):
    shared = _build_shared_params(D_spatial)
    tf, tc = _make_torch(shared, riccati)
    jf, jc = _make_jax(shared, riccati)
    ts = tf(tc)
    js = jf(jc)
    for n in shared["names"]:
        assert _allclose(ts[n].value, js[n].value), (
            f"value mismatch: {n!r} D={D_spatial} act={riccati}"
        )


def test_parity_first_partials(riccati, D_spatial):
    shared = _build_shared_params(D_spatial)
    tf, tc = _make_torch(shared, riccati)
    jf, jc = _make_jax(shared, riccati)
    ts = tf(tc)
    js = jf(jc)
    for n in shared["names"]:
        for axis in shared["axes"]:
            t_val = tops.derivative(ts, n, axis=axis)
            j_val = jops.derivative(js, n, axis=axis)
            assert _allclose(t_val, j_val), (
                f"d{axis} {n!r} D={D_spatial} act={riccati}"
            )


def test_parity_higher_order_pure(riccati, D_spatial):
    shared = _build_shared_params(D_spatial)
    tf, tc = _make_torch(shared, riccati)
    jf, jc = _make_jax(shared, riccati)
    ts = tf(tc)
    js = jf(jc)
    n = shared["names"][0]
    spatial = list(shared["axes"])[1:]
    if not spatial:
        return
    axis = spatial[0]
    for order in (2, 3):
        t_val = tops.derivative(ts, n, axis=axis, order=order)
        j_val = jops.derivative(js, n, axis=axis, order=order)
        assert _allclose(t_val, j_val, rtol=1e-11, atol=1e-11), (
            f"d^{order} {n}/d{axis}^{order} D={D_spatial} act={riccati}"
        )


def test_parity_mixed_partials(riccati, D_spatial):
    shared = _build_shared_params(D_spatial)
    if D_spatial == 1:
        return
    tf, tc = _make_torch(shared, riccati)
    jf, jc = _make_jax(shared, riccati)
    ts = tf(tc)
    js = jf(jc)
    n = shared["names"][0]
    spatial = list(shared["axes"])[1:]
    cases = [
        ((spatial[0], spatial[1]), (1, 1)),
        ((spatial[0], spatial[1]), (2, 1)),
        (("t", spatial[0]), (1, 1)),
    ]
    for axes, orders in cases:
        t_val = tops.mixed_partial(ts, n, axes, orders)
        j_val = jops.mixed_partial(js, n, axes, orders)
        assert _allclose(t_val, j_val), (
            f"mixed_partial({axes}, {orders}) D={D_spatial} act={riccati}"
        )


def test_parity_gradient(riccati, D_spatial):
    shared = _build_shared_params(D_spatial)
    tf, tc = _make_torch(shared, riccati)
    jf, jc = _make_jax(shared, riccati)
    ts = tf(tc)
    js = jf(jc)
    for n in shared["names"]:
        assert _allclose(tops.gradient(ts, n), jops.gradient(js, n)), n


def test_parity_laplacian_and_biharmonic(riccati, D_spatial):
    shared = _build_shared_params(D_spatial)
    tf, tc = _make_torch(shared, riccati)
    jf, jc = _make_jax(shared, riccati)
    ts = tf(tc)
    js = jf(jc)
    for n in shared["names"][:1]:
        assert _allclose(tops.laplacian(ts, n), jops.laplacian(js, n))
        assert _allclose(
            tops.biharmonic(ts, n), jops.biharmonic(js, n),
            rtol=1e-11, atol=1e-11,
        )


def test_parity_polylaplacian(riccati, D_spatial):
    shared = _build_shared_params(D_spatial)
    tf, tc = _make_torch(shared, riccati)
    jf, jc = _make_jax(shared, riccati)
    ts = tf(tc)
    js = jf(jc)
    n = shared["names"][0]
    for k in (1, 2):
        t_val = tops.polylaplacian(ts, n, k=k)
        j_val = jops.polylaplacian(js, n, k=k)
        assert _allclose(t_val, j_val, rtol=1e-11, atol=1e-11), (
            f"polylap k={k} D={D_spatial} act={riccati}"
        )


def test_parity_divergence(riccati, D_spatial):
    shared = _build_shared_params(D_spatial)
    if D_spatial == 1:
        return
    velocity = shared["groups"].get("velocity")
    if velocity is None:
        return
    tf, tc = _make_torch(shared, riccati)
    jf, jc = _make_jax(shared, riccati)
    ts = tf(tc)
    js = jf(jc)
    assert _allclose(
        tops.divergence(ts, velocity),
        jops.divergence(js, velocity),
    )
