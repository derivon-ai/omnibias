# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Bit-parity: torch ``SpectralVectorField`` vs JAX ``SpectralVectorField``.

Both backends share the same closed-form math. Parameters are matching
numpy arrays; per-op outputs match to ``rtol=1e-12, atol=1e-12`` in
float64 for orders that don't hit the activation-derivative recurrence
floor.

Sweep: 1D + 2D + 3D spatial layouts (with one time axis), Riccati
activations, the full op surface defined by the omnibias-pinn v0.1
plan.
"""

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
from omnibias.pinn.jax.fields.spectral import SpectralVectorField as JaxField
from omnibias.pinn.torch import ops as tops
from omnibias.pinn.torch.fields.spectral import SpectralVectorField as TorchField

_RICCATI = ("tanh", "sigmoid", "softplus", "gaussian", "exp")


@pytest.fixture(params=_RICCATI)
def riccati(request) -> str:
    return request.param


@pytest.fixture(params=[1, 2, 3])
def D_spatial(request) -> int:
    return request.param


def _build_shared_params(D_spatial: int, K: int = 3, time_hidden: int = 6):
    rng = np.random.default_rng(2026)
    C = 2 if D_spatial == 1 else (3 if D_spatial == 2 else 4)
    modes = (2 * K + 1) ** D_spatial
    out_dim = C * modes
    W_t = rng.normal(scale=0.5, size=(time_hidden, 1)).astype(np.float64)
    beta_t = rng.normal(scale=0.1, size=(time_hidden,)).astype(np.float64)
    V = rng.normal(scale=0.2, size=(out_dim, time_hidden)).astype(np.float64)
    b_t = rng.normal(scale=0.1, size=(out_dim,)).astype(np.float64)
    if D_spatial == 1:
        coords = np.zeros((6, 2), dtype=np.float64)
        coords[:, 0] = np.linspace(0.0, 1.0, 6)              # t
        coords[:, 1] = np.linspace(0.1, 1.7, 6)              # x
        axes = ("t", "x")
        periodicity = (False, True)
        names = ("u", "p")
        groups = {}
    elif D_spatial == 2:
        coords = np.zeros((6, 3), dtype=np.float64)
        coords[:, 0] = np.linspace(0.0, 1.0, 6)
        coords[:, 1] = np.linspace(0.1, 1.7, 6)
        coords[:, 2] = np.linspace(0.2, 1.5, 6)
        axes = ("t", "x", "y")
        periodicity = (False, True, True)
        names = ("u", "v", "p")
        groups = {"velocity": ("u", "v")}
    else:
        coords = np.zeros((5, 4), dtype=np.float64)
        coords[:, 0] = np.linspace(0.0, 1.0, 5)
        coords[:, 1] = np.linspace(0.1, 1.7, 5)
        coords[:, 2] = np.linspace(0.2, 1.5, 5)
        coords[:, 3] = np.linspace(-0.3, 1.1, 5)
        axes = ("t", "x", "y", "z")
        periodicity = (False, True, True, True)
        names = ("u", "v", "w", "p")
        groups = {"velocity": ("u", "v", "w")}
    return dict(
        W_t=W_t, beta_t=beta_t, V=V, b_t=b_t,
        coords=coords, axes=axes, periodicity=periodicity,
        names=names, groups=groups, K=K, time_hidden=time_hidden,
    )


def _make_torch(shared, activation: str):
    cspec = CoordinateSpec(
        axes=shared["axes"],
        periodicity=shared["periodicity"],
        time_axis="t",
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
    coords.requires_grad_(False)
    return field, coords


def _make_jax(shared, activation: str):
    cspec = CoordinateSpec(
        axes=shared["axes"],
        periodicity=shared["periodicity"],
        time_axis="t",
    )
    mspec = ComponentSpec(shared["names"], groups=shared["groups"] or None)
    spec = jax_get_activation(activation)
    L_value = tuple(2.0 * math.pi for _ in cspec.spatial_axes)
    field = JaxField(
        coordinate_spec=cspec, components=mspec, spec=spec,
        W_t=jnp.asarray(shared["W_t"]),
        beta_t=jnp.asarray(shared["beta_t"]),
        inner_W=tuple(),
        inner_b=tuple(),
        V=jnp.asarray(shared["V"]),
        b_t=jnp.asarray(shared["b_t"]),
        K=shared["K"], L=L_value,
        time_hidden=shared["time_hidden"], time_depth=1,
    )
    coords = jnp.asarray(shared["coords"])
    return field, coords


def _allclose(t, j, *, rtol: float = 1e-12, atol: float = 1e-12) -> bool:
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu().numpy()
    return np.allclose(np.asarray(t), np.asarray(j), rtol=rtol, atol=atol)


# -- value -----------------------------------------------------------


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
    for order in (2, 3, 4):
        t_val = tops.derivative(ts, n, axis=axis, order=order)
        j_val = jops.derivative(js, n, axis=axis, order=order)
        rtol = 1e-12 if order <= 3 else 1e-10
        atol = rtol
        assert _allclose(t_val, j_val, rtol=rtol, atol=atol), (
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
        (("t", spatial[0]), (2, 1)),
    ]
    if D_spatial == 3:
        cases.append((tuple(spatial), (1, 1, 1)))
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
        t_g = tops.gradient(ts, n)
        j_g = jops.gradient(js, n)
        assert _allclose(t_g, j_g), f"gradient {n!r} D={D_spatial} act={riccati}"
        # Include the time axis on a separate call
        full_axes = shared["axes"]
        t_full = tops.gradient(ts, n, axes=full_axes)
        j_full = jops.gradient(js, n, axes=full_axes)
        assert _allclose(t_full, j_full)


def test_parity_laplacian_and_biharmonic(riccati, D_spatial):
    shared = _build_shared_params(D_spatial)
    tf, tc = _make_torch(shared, riccati)
    jf, jc = _make_jax(shared, riccati)
    ts = tf(tc)
    js = jf(jc)
    for n in shared["names"][:1]:
        t_lap = tops.laplacian(ts, n)
        j_lap = jops.laplacian(js, n)
        assert _allclose(t_lap, j_lap)
        t_bih = tops.biharmonic(ts, n)
        j_bih = jops.biharmonic(js, n)
        assert _allclose(t_bih, j_bih, rtol=1e-11, atol=1e-11)


def test_parity_polylaplacian(riccati, D_spatial):
    shared = _build_shared_params(D_spatial)
    tf, tc = _make_torch(shared, riccati)
    jf, jc = _make_jax(shared, riccati)
    ts = tf(tc)
    js = jf(jc)
    n = shared["names"][0]
    for k in (1, 2, 3):
        t_val = tops.polylaplacian(ts, n, k=k)
        j_val = jops.polylaplacian(js, n, k=k)
        rtol = 1e-12 if k == 1 else (1e-11 if k == 2 else 1e-10)
        atol = rtol
        assert _allclose(t_val, j_val, rtol=rtol, atol=atol), (
            f"polylap k={k} D={D_spatial} act={riccati}"
        )


def test_parity_divergence_curl(riccati, D_spatial):
    shared = _build_shared_params(D_spatial)
    if D_spatial == 1:
        return
    tf, tc = _make_torch(shared, riccati)
    jf, jc = _make_jax(shared, riccati)
    ts = tf(tc)
    js = jf(jc)
    velocity = shared["groups"].get("velocity")
    if velocity is None:
        return
    t_div = tops.divergence(ts, velocity)
    j_div = jops.divergence(js, velocity)
    assert _allclose(t_div, j_div)
    t_curl = tops.curl(ts, velocity)
    j_curl = jops.curl(js, velocity)
    assert _allclose(t_curl, j_curl)


def test_parity_jacobian(riccati, D_spatial):
    shared = _build_shared_params(D_spatial)
    tf, tc = _make_torch(shared, riccati)
    jf, jc = _make_jax(shared, riccati)
    ts = tf(tc)
    js = jf(jc)
    if D_spatial == 1:
        names = shared["names"]
    else:
        names = shared["groups"]["velocity"]
    t_jac = tops.jacobian(ts, names)
    j_jac = jops.jacobian(js, names)
    assert _allclose(t_jac, j_jac)


# -- deep temporal head (time_depth > 1): bit-parity of closed-form jets ----


def _build_deep_shared(time_hidden: int = 6, time_depth: int = 3, K: int = 3) -> dict:
    """Shared float64 params for a 2D-spatial deep-temporal-head field."""
    rng = np.random.default_rng(99)
    C = 2
    out_dim = C * (2 * K + 1) ** 2
    shared = dict(
        W_t=rng.normal(scale=0.7, size=(time_hidden, 1)),
        beta_t=rng.normal(scale=0.2, size=(time_hidden,)),
        inner_W=[
            rng.normal(scale=1.0 / math.sqrt(time_hidden), size=(time_hidden, time_hidden))
            for _ in range(time_depth - 1)
        ],
        inner_b=[rng.normal(scale=0.2, size=(time_hidden,)) for _ in range(time_depth - 1)],
        V=rng.normal(scale=0.2, size=(out_dim, time_hidden)),
        b_t=rng.normal(scale=0.1, size=(out_dim,)),
        K=K, time_hidden=time_hidden, time_depth=time_depth,
    )
    coords = np.zeros((6, 3), dtype=np.float64)
    coords[:, 0] = np.linspace(0.0, 1.0, 6)
    coords[:, 1] = np.linspace(0.1, 1.7, 6)
    coords[:, 2] = np.linspace(0.2, 1.5, 6)
    shared["coords"] = coords
    return shared


def _make_torch_deep(shared, activation: str):
    cspec = CoordinateSpec(
        axes=("t", "x", "y"), periodicity=(False, True, True), time_axis="t",
    )
    mspec = ComponentSpec(("u", "v"), groups={"velocity": ("u", "v")})
    field = TorchField(
        coordinate_spec=cspec, components=mspec, K=shared["K"],
        time_hidden=shared["time_hidden"], time_depth=shared["time_depth"],
        activation=activation, dtype=torch.float64,
    )
    with torch.no_grad():
        field.W_t.copy_(torch.from_numpy(shared["W_t"]))
        field.beta_t.copy_(torch.from_numpy(shared["beta_t"]))
        for i, layer in enumerate(field._inner_layers):
            layer.weight.copy_(torch.from_numpy(shared["inner_W"][i]))
            layer.bias.copy_(torch.from_numpy(shared["inner_b"][i]))
        field.V.copy_(torch.from_numpy(shared["V"]))
        field.b_t.copy_(torch.from_numpy(shared["b_t"]))
    return field, torch.from_numpy(shared["coords"])


def _make_jax_deep(shared, activation: str):
    cspec = CoordinateSpec(
        axes=("t", "x", "y"), periodicity=(False, True, True), time_axis="t",
    )
    mspec = ComponentSpec(("u", "v"), groups={"velocity": ("u", "v")})
    spec = jax_get_activation(activation)
    L_value = tuple(2.0 * math.pi for _ in cspec.spatial_axes)
    field = JaxField(
        coordinate_spec=cspec, components=mspec, spec=spec,
        W_t=jnp.asarray(shared["W_t"]), beta_t=jnp.asarray(shared["beta_t"]),
        inner_W=tuple(jnp.asarray(w) for w in shared["inner_W"]),
        inner_b=tuple(jnp.asarray(b) for b in shared["inner_b"]),
        V=jnp.asarray(shared["V"]), b_t=jnp.asarray(shared["b_t"]),
        K=shared["K"], L=L_value,
        time_hidden=shared["time_hidden"], time_depth=shared["time_depth"],
    )
    return field, jnp.asarray(shared["coords"])


@pytest.mark.parametrize("activation", ["tanh", "sigmoid", "softplus"])
@pytest.mark.parametrize("time_depth", [2, 3])
def test_parity_deep_time_head(activation: str, time_depth: int) -> None:
    """torch and jax deep temporal heads share the ``mlp_jet`` kernel, so their
    closed-form time derivatives must be bit-identical (no autograd anywhere)."""
    shared = _build_deep_shared(time_depth=time_depth)
    tf, tc = _make_torch_deep(shared, activation)
    jf, jc = _make_jax_deep(shared, activation)
    ts = tf(tc)
    js = jf(jc)
    for order in (1, 2, 3):
        t_val = tops.derivative(ts, "u", axis="t", order=order)
        j_val = jops.derivative(js, "u", axis="t", order=order)
        rtol = 1e-11 if order <= 2 else 1e-9
        assert _allclose(t_val, j_val, rtol=rtol, atol=rtol), (
            f"d^{order}/dt^{order} depth={time_depth} act={activation}"
        )
    t_mp = tops.mixed_partial(ts, "u", ("x", "t"), (1, 2))
    j_mp = jops.mixed_partial(js, "u", ("x", "t"), (1, 2))
    assert _allclose(t_mp, j_mp, rtol=1e-10, atol=1e-10), (
        f"mixed d^3/dx dt^2 depth={time_depth} act={activation}"
    )
