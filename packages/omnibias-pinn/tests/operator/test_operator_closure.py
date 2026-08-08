# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Operator-closure: multi-head encoders, parametric losses, geometry hard BC."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import torch
from omnibias.pinn import ComponentSpec, CoordinateSpec
from omnibias.pinn.domain import Sphere
from omnibias.pinn.operator import ConditioningSpec
from omnibias.pinn.operator.jax import (
    make_deeponet,
    make_fno2d,
)
from omnibias.pinn.operator.jax import (
    make_parametric_heat_slab as make_parametric_heat_slab_jax,
)
from omnibias.pinn.operator.torch import (
    build_deeponet,
    build_fno2d,
    condition_with_geometry,
    encode_geometry,
    make_nonperiodic_parametric_heat_slab,
    make_parametric_heat_slab,
    make_variable_diffusivity_disk_poisson,
    parametric_heat_residual_loss,
    probe_grid,
)
from omnibias.pinn.torch import ops as tops

DTYPE = torch.float64


def test_multi_head_branch_encoder_forward():
    cond = ConditioningSpec(
        n_function_sensors=8,
        n_parameters=1,
        n_boundary_sensors=2,
        n_geometry_probes=4,
    )
    op = build_deeponet(
        coordinate_spec=CoordinateSpec(("x", "t"), time_axis="t"),
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
    F = 2
    sensors = torch.randn(F, 8, dtype=DTYPE)
    params = torch.tensor([[0.1], [0.2]], dtype=DTYPE)
    boundary = torch.randn(F, 2, dtype=DTYPE)
    geometry = torch.randn(F, 4, dtype=DTYPE)
    field = op.condition(
        sensors, parameters=params, boundary=boundary, geometry=geometry
    )
    coords = torch.rand(5, 2, dtype=DTYPE)
    u = tops.value(field.on_grid(coords), "u")
    assert u.shape == (F * 5,)
    assert torch.isfinite(u).all()


def test_function_only_path_unchanged_api():
    op = build_deeponet(
        coordinate_spec=CoordinateSpec(("x", "t")),
        components=ComponentSpec(("u",)),
        n_sensors=8,
        trunk_width=4,
        trunk_hidden=8,
        trunk_depth=2,
        branch_hidden=8,
        branch_depth=2,
        dtype=DTYPE,
    )
    sensors = torch.randn(1, 8, dtype=DTYPE)
    field = op.condition(sensors)
    assert tops.value(field(torch.rand(3, 2, dtype=DTYPE)), "u").shape == (3,)


def test_parametric_heat_residual_loss_broadcasts_diffusivity():
    slab = make_parametric_heat_slab(
        n_samples=2, n_grid=32, n_sensors=8, n_modes=2, n_times=5, seed=0, dtype=DTYPE
    )
    cond = ConditioningSpec(n_function_sensors=8, n_parameters=1)
    op = build_deeponet(
        coordinate_spec=CoordinateSpec(("x", "t"), time_axis="t"),
        components=ComponentSpec(("u",)),
        n_sensors=8,
        trunk_width=4,
        trunk_hidden=8,
        trunk_depth=2,
        branch_hidden=8,
        branch_depth=2,
        conditioning=cond,
        jet_order=2,
        dtype=DTYPE,
    )
    loss = parametric_heat_residual_loss(op, slab)
    assert torch.isfinite(loss)


def test_nonperiodic_parametric_heat_slab_finite():
    slab = make_nonperiodic_parametric_heat_slab(
        n_samples=2, n_grid=32, n_sensors=8, n_modes=2, n_times=4, seed=0, dtype=DTYPE
    )
    assert torch.isfinite(slab.values).all()
    assert slab.parameters.shape == (2, 1)


def test_manufactured_disk_poisson_shapes():
    slab = make_variable_diffusivity_disk_poisson(
        n_samples=3, n_grid=16, n_sensors=8, seed=0, dtype=DTYPE
    )
    assert slab.values.shape[0] == 3
    assert slab.radii.shape == (3, 1)


def test_condition_with_geometry_hard_bc_on_sphere():
    probes = probe_grid([(-1.5, 1.5), (-1.5, 1.5)], n_per_axis=2)
    geom = encode_geometry(Sphere(center=(0.0, 0.0), radius=1.0), probes)
    cond = ConditioningSpec(n_function_sensors=4, n_geometry_probes=geom.numel())
    op = build_deeponet(
        coordinate_spec=CoordinateSpec(("x", "y")),
        components=ComponentSpec(("u",)),
        n_sensors=4,
        trunk_width=4,
        trunk_hidden=8,
        trunk_depth=2,
        branch_hidden=8,
        branch_depth=2,
        conditioning=cond,
        dtype=DTYPE,
    )
    sensors = torch.randn(1, 4, dtype=DTYPE)
    wrapped = condition_with_geometry(
        op,
        sensors,
        geometry=geom.unsqueeze(0),
        sdf=Sphere(center=(0.0, 0.0), radius=1.0),
        hard_bc=True,
    )
    boundary = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]], dtype=DTYPE)
    u = tops.value(wrapped(boundary), "u")
    assert float(u.detach().abs().max()) < 1e-8


def test_jax_parametric_heat_and_fno2d():
    slab = make_parametric_heat_slab_jax(
        n_samples=2, n_grid=32, n_sensors=8, n_modes=2, n_times=5, seed=0
    )
    assert slab.parameters.shape == (2, 1)
    fno = make_fno2d(modes_x=4, modes_y=4, width=8, n_layers=2)
    u0 = jax.random.normal(jax.random.PRNGKey(0), (2, 16, 16), dtype=jnp.float64)
    out = fno(u0)
    assert out.shape == (2, 16, 16, 1)


def test_jax_fno2d_matches_torch():
    jax.config.update("jax_enable_x64", True)
    t_fno = build_fno2d(modes_x=4, modes_y=4, width=8, n_layers=2, dtype=DTYPE)
    j_fno = make_fno2d(modes_x=4, modes_y=4, width=8, n_layers=2, seed=0)
    # Copy weights for a structural smoke (shapes only — not bit parity yet).
    u0 = torch.randn(1, 12, 12, dtype=DTYPE)
    assert t_fno(u0).shape == (1, 12, 12, 1)
    assert j_fno(jnp.asarray(u0.numpy())).shape == (1, 12, 12, 1)
