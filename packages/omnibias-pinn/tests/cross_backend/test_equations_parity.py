# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Cross-backend bit-parity tests for the equation registry.

For each equation in :mod:`omnibias.pinn.{torch,jax}.equations`, build a
matching pair of fields with shared parameters and assert the residuals
match to ``rtol=1e-9, atol=1e-12`` (the plan's bit-parity contract for
equations).

We reuse the same parameter scaffolding as
:mod:`tests.cross_backend.test_spectral_parity` for stability.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

jax.config.update("jax_enable_x64", True)

from omnibias.jax.activations import get_activation as jax_get_activation
from omnibias.pinn._core.components import ComponentSpec
from omnibias.pinn._core.coords import CoordinateSpec
from omnibias.pinn.jax import equations as jeq
from omnibias.pinn.jax.fields.spectral import SpectralVectorField as JaxField
from omnibias.pinn.torch import equations as teq
from omnibias.pinn.torch.fields.spectral import SpectralVectorField as TorchField

_RICCATI = ("tanh", "sigmoid")


@pytest.fixture(params=_RICCATI)
def riccati(request) -> str:
    return request.param


def _shared_params(*, axes, periodicity, names, groups, K, time_hidden, n_pts, seed):
    """Build a matching parameter dict + collocation coords for a
    SpectralVectorField on the given axis spec."""
    rng = np.random.default_rng(seed)
    D_spatial = sum(1 for a in axes if a != "t")
    C = len(names)
    modes = (2 * K + 1) ** D_spatial
    out_dim = C * modes
    W_t = rng.normal(scale=0.5, size=(time_hidden, 1)).astype(np.float64)
    beta_t = rng.normal(scale=0.1, size=(time_hidden,)).astype(np.float64)
    V = rng.normal(scale=0.2, size=(out_dim, time_hidden)).astype(np.float64)
    b_t = rng.normal(scale=0.1, size=(out_dim,)).astype(np.float64)
    coords = rng.normal(size=(n_pts, len(axes))).astype(np.float64)
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
    coords = torch.from_numpy(shared["coords"]).clone()
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
        inner_W=tuple(), inner_b=tuple(),
        V=jnp.asarray(shared["V"]),
        b_t=jnp.asarray(shared["b_t"]),
        K=shared["K"], L=L_value,
        time_hidden=shared["time_hidden"], time_depth=1,
    )
    coords = jnp.asarray(shared["coords"])
    return field, coords


def _allclose(t, j, *, rtol: float = 1e-9, atol: float = 1e-12) -> bool:
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu().numpy()
    return np.allclose(np.asarray(t), np.asarray(j), rtol=rtol, atol=atol)


# ---------------- Heat ---------------------------------------------


def test_heat_residual_parity(riccati):
    shared = _shared_params(
        axes=("t", "x"), periodicity=(False, True),
        names=("u",), groups={}, K=3, time_hidden=4, n_pts=6, seed=10,
    )
    tf, tc = _make_torch(shared, riccati)
    jf, jc = _make_jax(shared, riccati)
    ts = tf(tc)
    js = jf(jc)
    t_out = teq.heat(ts, alpha=0.7)
    j_out = jeq.heat(js, alpha=0.7)
    assert _allclose(t_out.residual, j_out.residual)


# ---------------- Burgers ------------------------------------------


def test_burgers_scalar_parity(riccati):
    shared = _shared_params(
        axes=("t", "x"), periodicity=(False, True),
        names=("u",), groups={}, K=3, time_hidden=4, n_pts=6, seed=11,
    )
    tf, tc = _make_torch(shared, riccati)
    jf, jc = _make_jax(shared, riccati)
    ts = tf(tc)
    js = jf(jc)
    t_out = teq.burgers(ts, nu=0.05, form="scalar")
    j_out = jeq.burgers(js, nu=0.05, form="scalar")
    assert _allclose(t_out.residual, j_out.residual)


def test_burgers_vector_2d_parity(riccati):
    shared = _shared_params(
        axes=("t", "x", "y"), periodicity=(False, True, True),
        names=("u", "v"), groups={"velocity": ("u", "v")},
        K=3, time_hidden=4, n_pts=5, seed=12,
    )
    tf, tc = _make_torch(shared, riccati)
    jf, jc = _make_jax(shared, riccati)
    ts = tf(tc)
    js = jf(jc)
    t_out = teq.burgers(ts, nu=0.01, form="vector",
                        velocity=("u", "v"))
    j_out = jeq.burgers(js, nu=0.01, form="vector",
                         velocity=("u", "v"))
    assert _allclose(t_out.residual, j_out.residual)


# ---------------- KS -----------------------------------------------


def test_ks_1d_residual_parity(riccati):
    shared = _shared_params(
        axes=("t", "x"), periodicity=(False, True),
        names=("u",), groups={}, K=4, time_hidden=4, n_pts=7, seed=13,
    )
    tf, tc = _make_torch(shared, riccati)
    jf, jc = _make_jax(shared, riccati)
    ts = tf(tc)
    js = jf(jc)
    t_out = teq.kuramoto_sivashinsky(ts, form="1d")
    j_out = jeq.kuramoto_sivashinsky(js, form="1d")
    # 4th-order derivative drift: relax slightly.
    assert _allclose(t_out.residual, j_out.residual, rtol=1e-8, atol=1e-12)


# ---------------- Cahn-Hilliard ------------------------------------


def test_cahn_hilliard_residual_parity(riccati):
    shared = _shared_params(
        axes=("t", "x", "y"), periodicity=(False, True, True),
        names=("c",), groups={}, K=3, time_hidden=4, n_pts=6, seed=14,
    )
    tf, tc = _make_torch(shared, riccati)
    jf, jc = _make_jax(shared, riccati)
    ts = tf(tc)
    js = jf(jc)
    t_out = teq.cahn_hilliard(ts, M=1.5, kappa=2e-3,
                                potential=teq.GinzburgLandauPotential(W=1.0))
    j_out = jeq.cahn_hilliard(js, M=1.5, kappa=2e-3,
                                potential=jeq.GinzburgLandauPotential(W=1.0))
    assert _allclose(t_out.residual, j_out.residual)


# ---------------- Biharmonic ---------------------------------------


def test_biharmonic_residual_parity(riccati):
    shared = _shared_params(
        axes=("t", "x", "y"), periodicity=(False, True, True),
        names=("u",), groups={}, K=3, time_hidden=4, n_pts=5, seed=15,
    )
    tf, tc = _make_torch(shared, riccati)
    jf, jc = _make_jax(shared, riccati)
    ts = tf(tc)
    js = jf(jc)
    t_out = teq.biharmonic(ts, component="u", include_time=False)
    j_out = jeq.biharmonic(js, component="u", include_time=False)
    assert _allclose(t_out.residual, j_out.residual)


# ---------------- Navier-Stokes ------------------------------------


def test_ns_primitive_2d_residual_parity(riccati):
    shared = _shared_params(
        axes=("t", "x", "y"), periodicity=(False, True, True),
        names=("u", "v", "p"), groups={"velocity": ("u", "v")},
        K=2, time_hidden=4, n_pts=4, seed=16,
    )
    tf, tc = _make_torch(shared, riccati)
    jf, jc = _make_jax(shared, riccati)
    ts = tf(tc)
    js = jf(jc)
    t_out = teq.navier_stokes(ts, viscosity=0.1, form="primitive_2d",
                                velocity=("u", "v"))
    j_out = jeq.navier_stokes(js, viscosity=0.1, form="primitive_2d",
                                velocity=("u", "v"))
    assert _allclose(t_out.residual, j_out.residual)
    assert _allclose(t_out.continuity, j_out.continuity)


def test_ns_primitive_3d_residual_parity(riccati):
    shared = _shared_params(
        axes=("t", "x", "y", "z"), periodicity=(False, True, True, True),
        names=("u", "v", "w", "p"),
        groups={"velocity": ("u", "v", "w")},
        K=2, time_hidden=4, n_pts=3, seed=17,
    )
    tf, tc = _make_torch(shared, riccati)
    jf, jc = _make_jax(shared, riccati)
    ts = tf(tc)
    js = jf(jc)
    t_out = teq.navier_stokes(ts, viscosity=0.1, form="primitive_3d",
                                velocity=("u", "v", "w"))
    j_out = jeq.navier_stokes(js, viscosity=0.1, form="primitive_3d",
                                velocity=("u", "v", "w"))
    assert _allclose(t_out.residual, j_out.residual)
    assert _allclose(t_out.continuity, j_out.continuity)


def test_ns_vorticity_stream_2d_residual_parity(riccati):
    shared = _shared_params(
        axes=("t", "x", "y"), periodicity=(False, True, True),
        names=("psi",), groups={}, K=3, time_hidden=4, n_pts=6, seed=18,
    )
    tf, tc = _make_torch(shared, riccati)
    jf, jc = _make_jax(shared, riccati)
    ts = tf(tc)
    js = jf(jc)
    t_out = teq.navier_stokes(ts, viscosity=0.05,
                                form="vorticity_stream_2d",
                                streamfunction="psi")
    j_out = jeq.navier_stokes(js, viscosity=0.05,
                                form="vorticity_stream_2d",
                                streamfunction="psi")
    # Mixed partials of order 4 (psi_xxx, psi_yyy etc) hit the same
    # numerical drift floor as the 4th-order spectral parity test.
    assert _allclose(t_out.residual, j_out.residual,
                     rtol=1e-8, atol=1e-12)
