# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Tests for multi-head operator conditioning and FNO2d."""

from __future__ import annotations

import torch
from omnibias.pinn import ComponentSpec, CoordinateSpec
from omnibias.pinn.domain import Sphere
from omnibias.pinn.operator import ConditioningSpec
from omnibias.pinn.operator.torch import (
    build_deeponet,
    build_fno2d,
    encode_geometry,
    make_parametric_heat_slab,
    probe_grid,
)
from omnibias.pinn.torch import ops as tops

DTYPE = torch.float64


def test_conditioning_spec_total_dim():
    c = ConditioningSpec(
        n_function_sensors=8, n_parameters=1, n_boundary_sensors=4, n_geometry_probes=16
    )
    assert c.total_dim == 8 + 1 + 4 + 16
    assert c.has_parameters and c.has_boundary and c.has_geometry


def test_parameter_head_distinguishes_scalar_values():
    """Width-1 LayerNorm would map every scalar to 0; coeffs must still vary."""
    cond = ConditioningSpec(n_function_sensors=8, n_parameters=1)
    op = build_deeponet(
        coordinate_spec=CoordinateSpec(
            ("x", "t"), domain=((0.0, 1.0), (0.0, 1.0)), time_axis="t"
        ),
        components=ComponentSpec(("u",)),
        n_sensors=8,
        trunk_width=4,
        trunk_hidden=8,
        trunk_depth=2,
        branch_hidden=8,
        branch_depth=2,
        conditioning=cond,
        dtype=DTYPE,
    )
    sensors = torch.randn(1, 8, dtype=DTYPE).expand(3, -1).contiguous()
    params = torch.tensor([[0.05], [0.15], [0.35]], dtype=DTYPE)
    coeffs, _bias = op.branch(sensors=sensors, parameters=params)
    # Identical sensors, distinct parameters => distinct branch coeffs.
    d01 = float((coeffs[0] - coeffs[1]).detach().norm())
    d02 = float((coeffs[0] - coeffs[2]).detach().norm())
    assert d01 > 1e-8, d01
    assert d02 > 1e-8, d02


def test_parameter_head_distinguishes_scalar_values_jax():
    """JAX twin of the width-1 LayerNorm identity guard."""
    import jax
    import jax.numpy as jnp
    from omnibias.pinn.operator.jax import make_deeponet as make_deeponet_jax

    cond = ConditioningSpec(n_function_sensors=8, n_parameters=1)
    op = make_deeponet_jax(
        coordinate_spec=CoordinateSpec(
            ("x", "t"), domain=((0.0, 1.0), (0.0, 1.0)), time_axis="t"
        ),
        components=ComponentSpec(("u",)),
        n_sensors=8,
        trunk_width=4,
        trunk_hidden=8,
        trunk_depth=2,
        branch_hidden=8,
        branch_depth=2,
        conditioning=cond,
        seed=0,
    )
    sensors = jnp.broadcast_to(jax.random.normal(jax.random.PRNGKey(1), (1, 8)), (3, 8))
    params = jnp.asarray([[0.05], [0.15], [0.35]])
    coeffs, _bias = op.branch(sensors=sensors, parameters=params)
    d01 = float(jnp.linalg.norm(coeffs[0] - coeffs[1]))
    d02 = float(jnp.linalg.norm(coeffs[0] - coeffs[2]))
    assert d01 > 1e-8, d01
    assert d02 > 1e-8, d02


def test_geometry_encoding_length():
    probes = probe_grid([(-1.5, 1.5), (-1.5, 1.5)], n_per_axis=3)
    code = encode_geometry(Sphere(center=(0.0, 0.0), radius=1.0), probes)
    assert code.shape == (9,)


def test_parametric_heat_slab_shapes():
    slab = make_parametric_heat_slab(
        n_samples=4, n_grid=32, n_sensors=8, n_modes=2, n_times=5, seed=0
    )
    assert slab.parameters.shape == (4, 1)
    assert slab.sensors.shape == (4, 8)
    assert slab.values.shape[0] == 4


def test_fno2d_forward_shape():
    fno = build_fno2d(modes_x=4, modes_y=4, width=8, n_layers=2, dtype=DTYPE)
    u0 = torch.randn(2, 16, 16, dtype=DTYPE)
    out = fno(u0)
    assert out.shape == (2, 16, 16, 1)
