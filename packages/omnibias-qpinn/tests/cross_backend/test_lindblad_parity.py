# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Cross-backend parity for the GKSL residual and density-matrix cage."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

jax.config.update("jax_enable_x64", True)

from omnibias.jax.activations import get_activation as jax_get_activation
from omnibias.pinn._core.coords import CoordinateSpec
from omnibias.pinn.jax.fields.one_layer import OneLayerVectorField as JOne
from omnibias.pinn.torch.fields.one_layer import OneLayerVectorField as TOne
from omnibias.qpinn import make_rho_components, rho_entry_names
from omnibias.qpinn.jax import equations as jeq
from omnibias.qpinn.jax.cage import make_density_matrix_field as make_jax_cage
from omnibias.qpinn.torch import equations as teq
from omnibias.qpinn.torch.cage import make_density_matrix_field as make_torch_cage

from .conftest import _allclose


@pytest.fixture
def shared_rho_params():
    rng = np.random.default_rng(20260909)
    H, D, C = 8, 2, 8
    return dict(
        W=rng.normal(scale=0.5, size=(H, D)).astype(np.float64),
        beta=rng.normal(scale=0.1, size=(H,)).astype(np.float64),
        c=rng.normal(scale=0.5, size=(C, H)).astype(np.float64),
        b=rng.normal(scale=0.1, size=(C,)).astype(np.float64),
        coords=rng.normal(size=(6, D)).astype(np.float64),
        H=H,
        D=D,
        C=C,
    )


def _build_pair(shared, riccati: str):
    coord = CoordinateSpec(axes=("x", "t"))
    components = make_rho_components(dim=2)
    t_field = TOne(
        coordinate_spec=coord,
        components=components,
        hidden=shared["H"],
        base=riccati,
        dtype=torch.float64,
    )
    with torch.no_grad():
        t_field.W.weight.copy_(torch.from_numpy(shared["W"]))
        t_field.W.bias.copy_(torch.from_numpy(shared["beta"]))
        t_field.c.weight.copy_(torch.from_numpy(shared["c"]))
        t_field.c.bias.copy_(torch.from_numpy(shared["b"]))
    spec = jax_get_activation(riccati)
    j_field = JOne(
        coordinate_spec=coord,
        components=components,
        spec=spec,
        W=jnp.asarray(shared["W"]),
        beta=jnp.asarray(shared["beta"]),
        c=jnp.asarray(shared["c"]),
        b=jnp.asarray(shared["b"]),
        hidden=shared["H"],
    )
    return t_field, torch.from_numpy(shared["coords"]).clone(), j_field, jnp.asarray(shared["coords"])


def _ops():
    ham = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
    jumps = np.array([[[0.0, 1.0], [0.0, 0.0]]], dtype=np.complex128)
    rates = np.array([0.4], dtype=np.float64)
    return ham, jumps, rates


def test_lindblad_residual_parity(riccati, shared_rho_params) -> None:
    t_field, t_coords, j_field, j_coords = _build_pair(shared_rho_params, riccati)
    ham, jumps, rates = _ops()
    t_out = teq.lindblad(
        t_field(t_coords),
        torch.tensor(ham),
        torch.tensor(jumps),
        torch.tensor(rates),
    )
    j_out = jeq.lindblad(
        j_field(j_coords),
        jnp.asarray(ham),
        jnp.asarray(jumps),
        jnp.asarray(rates),
    )
    assert _allclose(t_out.residual, j_out.residual)


def test_density_cage_parity(riccati, shared_rho_params) -> None:
    t_field, t_coords, j_field, j_coords = _build_pair(shared_rho_params, riccati)
    t_cage = make_torch_cage(base=t_field, dim=2)
    j_cage = make_jax_cage(base=j_field, dim=2)
    t_state = t_cage(t_coords)
    j_state = j_cage(j_coords)
    for i in range(2):
        for j in range(2):
            re_n, im_n = rho_entry_names("rho", i, j)
            assert _allclose(t_state.ops.value(t_state, re_n), j_state.ops.value(j_state, re_n))
            assert _allclose(t_state.ops.value(t_state, im_n), j_state.ops.value(j_state, im_n))
            assert _allclose(
                t_state.ops.derivative(t_state, re_n, axis="t", order=1),
                j_state.ops.derivative(j_state, re_n, axis="t", order=1),
            )
