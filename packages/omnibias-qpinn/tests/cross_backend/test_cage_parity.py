# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Cross-backend bit-parity for the NormConservation cage + diagnostics."""

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
from omnibias.qpinn import make_psi_components
from omnibias.qpinn.jax.cage import make_norm_conservation_field as make_jax_cage
from omnibias.qpinn.jax.diagnostics import norm_squared as jax_norm_squared
from omnibias.qpinn.torch.cage import make_norm_conservation_field as make_torch_cage
from omnibias.qpinn.torch.diagnostics import norm_squared as torch_norm_squared

from .conftest import _allclose


def _build_pair(shared, axes, riccati):
    coord = CoordinateSpec(axes=axes)
    components = make_psi_components(name="psi")
    t_field = TOne(
        coordinate_spec=coord, components=components,
        hidden=shared["H"], base=riccati, dtype=torch.float64,
    )
    with torch.no_grad():
        t_field.W.weight.copy_(torch.from_numpy(shared["W"]))
        t_field.W.bias.copy_(torch.from_numpy(shared["beta"]))
        t_field.c.weight.copy_(torch.from_numpy(shared["c"]))
        t_field.c.bias.copy_(torch.from_numpy(shared["b"]))
    j_field = JOne(
        coordinate_spec=coord, components=components,
        spec=jax_get_activation(riccati),
        W=jnp.asarray(shared["W"]),
        beta=jnp.asarray(shared["beta"]),
        c=jnp.asarray(shared["c"]),
        b=jnp.asarray(shared["b"]),
        hidden=shared["H"],
    )
    return t_field, j_field


def test_norm_cage_values_parity(riccati, shared_psi_params_1d):
    """The caged ``psi_re`` / ``psi_im`` values must match across backends."""
    t_field, j_field = _build_pair(shared_psi_params_1d, ("x",), riccati)
    quad_np = np.linspace(-3.0, 3.0, 401).reshape(-1, 1).astype(np.float64)
    weights_np = np.full((401,), 6.0 / 401, dtype=np.float64)
    t_cage = make_torch_cage(
        base=t_field,
        quadrature_coords=torch.from_numpy(quad_np),
        quadrature_weights=torch.from_numpy(weights_np),
    )
    j_cage = make_jax_cage(
        base=j_field,
        quadrature_coords=jnp.asarray(quad_np),
        quadrature_weights=jnp.asarray(weights_np),
    )
    query_np = shared_psi_params_1d["coords"]
    t_state = t_cage(torch.from_numpy(query_np))
    j_state = j_cage(jnp.asarray(query_np))
    t_re = t_state.ops.value(t_state, "psi_re")
    j_re = j_state.ops.value(j_state, "psi_re")
    assert _allclose(t_re, j_re)
    t_im = t_state.ops.value(t_state, "psi_im")
    j_im = j_state.ops.value(j_state, "psi_im")
    assert _allclose(t_im, j_im)


def test_norm_cage_laplacian_parity(riccati, shared_psi_params_1d):
    """Laplacian of the caged field (used by every Schrodinger residual)
    must match across backends."""
    t_field, j_field = _build_pair(shared_psi_params_1d, ("x",), riccati)
    quad_np = np.linspace(-3.0, 3.0, 401).reshape(-1, 1).astype(np.float64)
    weights_np = np.full((401,), 6.0 / 401, dtype=np.float64)
    t_cage = make_torch_cage(
        base=t_field,
        quadrature_coords=torch.from_numpy(quad_np),
        quadrature_weights=torch.from_numpy(weights_np),
    )
    j_cage = make_jax_cage(
        base=j_field,
        quadrature_coords=jnp.asarray(quad_np),
        quadrature_weights=jnp.asarray(weights_np),
    )
    query_np = shared_psi_params_1d["coords"]
    t_state = t_cage(torch.from_numpy(query_np))
    j_state = j_cage(jnp.asarray(query_np))
    t_lap = t_state.ops.laplacian(t_state, "psi_re")
    j_lap = j_state.ops.laplacian(j_state, "psi_re")
    assert _allclose(t_lap, j_lap)


def test_norm_diagnostic_parity(riccati, shared_psi_params_1d):
    """The post-cage norm should be 1.0 (up to quadrature precision) in
    both backends."""
    t_field, j_field = _build_pair(shared_psi_params_1d, ("x",), riccati)
    quad_np = np.linspace(-3.0, 3.0, 401).reshape(-1, 1).astype(np.float64)
    weights_np = np.full((401,), 6.0 / 401, dtype=np.float64)
    t_cage = make_torch_cage(
        base=t_field,
        quadrature_coords=torch.from_numpy(quad_np),
        quadrature_weights=torch.from_numpy(weights_np),
    )
    j_cage = make_jax_cage(
        base=j_field,
        quadrature_coords=jnp.asarray(quad_np),
        quadrature_weights=jnp.asarray(weights_np),
    )
    t_state = t_cage(torch.from_numpy(quad_np))
    j_state = j_cage(jnp.asarray(quad_np))
    t_nrm = torch_norm_squared(
        t_state, quadrature_weights=torch.from_numpy(weights_np),
    )
    j_nrm = jax_norm_squared(
        j_state, quadrature_weights=jnp.asarray(weights_np),
    )
    assert abs(float(t_nrm.detach()) - 1.0) < 1e-12
    assert abs(float(j_nrm) - 1.0) < 1e-12
    assert _allclose(t_nrm, j_nrm)
