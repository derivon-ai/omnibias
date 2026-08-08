# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Torch ↔ JAX DeepONet parity: value, derivatives, laplacian, shared grid."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import torch
from omnibias.jax.activations import get_activation as jax_get_activation
from omnibias.jax.architectures.pinn import JetMLP
from omnibias.pinn import ComponentSpec, CoordinateSpec
from omnibias.pinn.jax import ops as jops
from omnibias.pinn.operator._core.spec import OperatorSpec
from omnibias.pinn.operator.jax.deeponet import (
    DeepONetOperator as JaxOp,
)
from omnibias.pinn.operator.jax.deeponet import (
    _BranchNet,
    _HeadEncoder,
)
from omnibias.pinn.operator.torch import build_deeponet
from omnibias.pinn.torch import ops as tops
from omnibias.torch.architectures.pinn import JetMLP as TorchJetMLP

jax.config.update("jax_enable_x64", True)

RTOL = 1e-11


def _copy_trunk_torch_to_jax(torch_trunk: TorchJetMLP) -> JetMLP:
    weights = []
    biases = []
    for lin in torch_trunk.linears:
        weights.append(jnp.asarray(lin.weight.detach().cpu().numpy()))
        biases.append(jnp.asarray(lin.bias.detach().cpu().numpy()))
    return JetMLP(
        weights=tuple(weights),
        biases=tuple(biases),
        spec=jax_get_activation(torch_trunk.spec.name),
        in_dim=torch_trunk.in_dim,
        out_dim=torch_trunk.out_dim,
    )


def _copy_head_torch_to_jax(torch_enc) -> _HeadEncoder:
    weights = []
    biases = []
    for lin in torch_enc.linears:
        weights.append(jnp.asarray(lin.weight.detach().cpu().numpy()))
        biases.append(jnp.asarray(lin.bias.detach().cpu().numpy()))
    return _HeadEncoder(
        norm_gamma=jnp.asarray(torch_enc.norm.weight.detach().cpu().numpy()),
        norm_beta=jnp.asarray(torch_enc.norm.bias.detach().cpu().numpy()),
        weights=tuple(weights),
        biases=tuple(biases),
        spec=jax_get_activation(torch_enc.spec.name),
        n_input=torch_enc.n_input,
        encoder_dim=torch_enc.encoder_dim,
    )


def _copy_branch_torch_to_jax(torch_branch) -> _BranchNet:
    function_encoder = _copy_head_torch_to_jax(torch_branch.encoders["function"])
    parameter_encoder = None
    boundary_encoder = None
    geometry_encoder = None
    if "pde_params" in torch_branch.encoders:
        parameter_encoder = _copy_head_torch_to_jax(torch_branch.encoders["pde_params"])
    if "boundary" in torch_branch.encoders:
        boundary_encoder = _copy_head_torch_to_jax(torch_branch.encoders["boundary"])
    if "geometry" in torch_branch.encoders:
        geometry_encoder = _copy_head_torch_to_jax(torch_branch.encoders["geometry"])
    fusion_weights = []
    fusion_biases = []
    for lin in torch_branch.fusion:
        fusion_weights.append(jnp.asarray(lin.weight.detach().cpu().numpy()))
        fusion_biases.append(jnp.asarray(lin.bias.detach().cpu().numpy()))
    return _BranchNet(
        function_encoder=function_encoder,
        parameter_encoder=parameter_encoder,
        boundary_encoder=boundary_encoder,
        geometry_encoder=geometry_encoder,
        fusion_weights=tuple(fusion_weights),
        fusion_biases=tuple(fusion_biases),
        spec=jax_get_activation(torch_branch.spec.name),
        layout=torch_branch.layout,
        n_components=torch_branch.n_components,
        trunk_width=torch_branch.trunk_width,
        per_sample_bias=torch_branch.per_sample_bias,
    )


def _matched_pair():
    cs = CoordinateSpec(("x", "t"))
    comps = ComponentSpec(("u",))
    torch.manual_seed(11)
    top = build_deeponet(
        coordinate_spec=cs,
        components=comps,
        n_sensors=16,
        trunk_width=8,
        trunk_hidden=12,
        trunk_depth=2,
        branch_hidden=12,
        branch_depth=2,
        base="tanh",
        jet_order=3,
    )
    jop = JaxOp(
        spec=OperatorSpec(cs, comps, n_sensors=16, trunk_width=8),
        trunk=_copy_trunk_torch_to_jax(top.core.trunk),
        branch=_copy_branch_torch_to_jax(top.branch),
        shared_bias=None,
        jet_order=3,
    )
    return top, jop


def test_parity_value_and_derivatives() -> None:
    top, jop = _matched_pair()
    rng = np.random.default_rng(0)
    sensors_np = rng.standard_normal((3, 16))
    coords_np = rng.standard_normal((3, 2))
    t_sensors = torch.tensor(sensors_np, dtype=torch.float64)
    t_coords = torch.tensor(coords_np, dtype=torch.float64)
    j_sensors = jnp.asarray(sensors_np)
    j_coords = jnp.asarray(coords_np)

    t_field = top.condition(t_sensors)
    j_field = jop.condition(j_sensors)
    t_state = t_field(t_coords)
    j_state = j_field(j_coords)

    np.testing.assert_allclose(
        tops.value(t_state, "u").detach().numpy(),
        np.asarray(jops.value(j_state, "u")),
        rtol=RTOL,
        atol=0.0,
    )
    for order in (1, 2, 3):
        for axis in (0, 1):
            np.testing.assert_allclose(
                tops.derivative(t_state, "u", axis=axis, order=order).detach().numpy(),
                np.asarray(jops.derivative(j_state, "u", axis=axis, order=order)),
                rtol=RTOL,
                atol=0.0,
            )
    np.testing.assert_allclose(
        tops.laplacian(t_state, "u").detach().numpy(),
        np.asarray(jops.laplacian(j_state, "u")),
        rtol=RTOL,
        atol=0.0,
    )


def test_parity_shared_grid() -> None:
    top, jop = _matched_pair()
    rng = np.random.default_rng(1)
    sensors_np = rng.standard_normal((4, 16))
    query_np = rng.standard_normal((6, 2))
    t_state = top.condition(torch.tensor(sensors_np, dtype=torch.float64)).on_grid(
        torch.tensor(query_np, dtype=torch.float64)
    )
    j_state = jop.condition(jnp.asarray(sensors_np)).on_grid(jnp.asarray(query_np))
    np.testing.assert_allclose(
        tops.value(t_state, "u").detach().numpy(),
        np.asarray(jops.value(j_state, "u")),
        rtol=RTOL,
        atol=0.0,
    )
    np.testing.assert_allclose(
        tops.derivative(t_state, "u", axis=0, order=1).detach().numpy(),
        np.asarray(jops.derivative(j_state, "u", axis=0, order=1)),
        rtol=RTOL,
        atol=0.0,
    )
