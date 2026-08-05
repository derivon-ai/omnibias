# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Cross-backend bit-parity tests for Helmholtz / Klein-Gordon / Dirac
residuals + Bloch / Hermitian cage helpers."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

jax.config.update("jax_enable_x64", True)

from omnibias.jax.activations import get_activation as jax_get_activation
from omnibias.pinn._core.components import ComponentSpec
from omnibias.pinn._core.coords import CoordinateSpec
from omnibias.pinn.jax.fields.one_layer import OneLayerVectorField as JOne
from omnibias.pinn.torch.fields.one_layer import OneLayerVectorField as TOne
from omnibias.qpinn import (
    make_psi_components,
    make_spinor_components,
)
from omnibias.qpinn.jax import cage as jcage
from omnibias.qpinn.jax import equations as jeq
from omnibias.qpinn.torch import cage as tcage
from omnibias.qpinn.torch import equations as teq

from .conftest import _allclose


def _make_pair(axes, riccati, components, hidden=8, seed=0xC0FFEE):
    """Build matching torch / jax OneLayerVectorField with shared params."""
    rng = np.random.default_rng(seed)
    D = len(axes)
    C = components.n_components
    W = rng.normal(scale=0.5, size=(hidden, D)).astype(np.float64)
    beta = rng.normal(scale=0.1, size=(hidden,)).astype(np.float64)
    c = rng.normal(scale=0.5, size=(C, hidden)).astype(np.float64)
    b = rng.normal(scale=0.1, size=(C,)).astype(np.float64)
    coord = CoordinateSpec(axes=axes)
    t_field = TOne(
        coordinate_spec=coord, components=components,
        hidden=hidden, base=riccati, dtype=torch.float64,
    )
    with torch.no_grad():
        t_field.W.weight.copy_(torch.from_numpy(W))
        t_field.W.bias.copy_(torch.from_numpy(beta))
        t_field.c.weight.copy_(torch.from_numpy(c))
        t_field.c.bias.copy_(torch.from_numpy(b))
    j_field = JOne(
        coordinate_spec=coord, components=components,
        spec=jax_get_activation(riccati),
        W=jnp.asarray(W), beta=jnp.asarray(beta),
        c=jnp.asarray(c), b=jnp.asarray(b),
        hidden=hidden,
    )
    coords_np = rng.normal(size=(13, D)).astype(np.float64)
    return t_field, j_field, coords_np


def test_helmholtz_parity(riccati):
    spec = make_psi_components(name="psi")
    t_field, j_field, coords_np = _make_pair(("x", "y"), riccati, spec)
    t_state = t_field(torch.from_numpy(coords_np))
    j_state = j_field(jnp.asarray(coords_np))

    def k_callable(s):
        # 1.0 + 0.5 * x^2  -- index-of-refraction style
        return 1.0 + 0.5 * s.coords[..., 0] ** 2

    t_res = teq.helmholtz(t_state, k=k_callable).residual
    j_res = jeq.helmholtz(j_state, k=k_callable).residual
    assert _allclose(t_res, j_res)


def test_klein_gordon_parity(riccati):
    spec = ComponentSpec(("phi",))
    t_field, j_field, coords_np = _make_pair(("x", "t"), riccati, spec)
    t_state = t_field(torch.from_numpy(coords_np))
    j_state = j_field(jnp.asarray(coords_np))
    t_res = teq.klein_gordon(t_state, mass=1.0, lambda_phi4=0.25).residual
    j_res = jeq.klein_gordon(j_state, mass=1.0, lambda_phi4=0.25).residual
    assert _allclose(t_res, j_res)


def test_dirac_parity(riccati):
    spec = make_spinor_components(name="spinor", n_components=4)
    t_field, j_field, coords_np = _make_pair(("x", "y", "z", "t"), riccati, spec)
    t_state = t_field(torch.from_numpy(coords_np))
    j_state = j_field(jnp.asarray(coords_np))
    t_res = teq.dirac(t_state, mass=1.0, representation="dirac").residual
    j_res = jeq.dirac(j_state, mass=1.0, representation="dirac").residual
    assert _allclose(t_res, j_res)


def test_dirac_weyl_parity(riccati):
    spec = make_spinor_components(name="spinor", n_components=4)
    t_field, j_field, coords_np = _make_pair(("x", "y", "z", "t"), riccati, spec)
    t_state = t_field(torch.from_numpy(coords_np))
    j_state = j_field(jnp.asarray(coords_np))
    t_res = teq.dirac(t_state, mass=1.0, representation="weyl").residual
    j_res = jeq.dirac(j_state, mass=1.0, representation="weyl").residual
    assert _allclose(t_res, j_res)


def test_bloch_value_parity(riccati):
    """The caged psi values should match across backends."""
    spec = make_psi_components(name="u")
    t_field, j_field, _ = _make_pair(("x",), riccati, spec)
    coords_np = np.linspace(-2.0, 2.0, 11).reshape(-1, 1).astype(np.float64)
    t_cage = tcage.make_bloch_periodic_field(base=t_field, k=[1.3])
    j_cage = jcage.make_bloch_periodic_field(base=j_field, k=[1.3])
    t_state = t_cage(torch.from_numpy(coords_np))
    j_state = j_cage(jnp.asarray(coords_np))
    t_re = t_state.ops.value(t_state, "psi_re")
    j_re = j_state.ops.value(j_state, "psi_re")
    assert _allclose(t_re, j_re)
    t_im = t_state.ops.value(t_state, "psi_im")
    j_im = j_state.ops.value(j_state, "psi_im")
    assert _allclose(t_im, j_im)


def test_bloch_laplacian_parity(riccati):
    spec = make_psi_components(name="u")
    t_field, j_field, _ = _make_pair(("x",), riccati, spec)
    coords_np = np.linspace(-2.0, 2.0, 11).reshape(-1, 1).astype(np.float64)
    t_cage = tcage.make_bloch_periodic_field(base=t_field, k=[1.3])
    j_cage = jcage.make_bloch_periodic_field(base=j_field, k=[1.3])
    t_state = t_cage(torch.from_numpy(coords_np))
    j_state = j_cage(jnp.asarray(coords_np))
    t_d2 = t_state.ops.derivative(t_state, "psi_re", axis=0, order=2)
    j_d2 = j_state.ops.derivative(j_state, "psi_re", axis=0, order=2)
    assert _allclose(t_d2, j_d2)


def test_hermitian_projection_parity():
    rng = np.random.default_rng(7)
    M_np = rng.normal(size=(4, 4)).astype(np.float64)
    t_H = tcage.hermitian_projection(torch.from_numpy(M_np))
    j_H = jcage.hermitian_projection(jnp.asarray(M_np))
    assert _allclose(t_H, j_H, rtol=1e-12, atol=1e-14)


def test_hermiticity_loss_parity():
    rng = np.random.default_rng(7)
    M_np = rng.normal(size=(4, 4)).astype(np.float64)
    t_L = tcage.hermiticity_loss(torch.from_numpy(M_np))
    j_L = jcage.hermiticity_loss(jnp.asarray(M_np))
    assert _allclose(t_L, j_L, rtol=1e-12, atol=1e-14)
