# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""PyTorch neural-operator drivers for omnibias.pinn.operator."""

from __future__ import annotations

from omnibias.pinn.operator.torch.data import (
    OperatorSlab,
    ParametricOperatorSlab,
    make_burgers_slab,
    make_heat_slab,
    make_ks_slab,
    make_nonperiodic_parametric_burgers_slab,
    make_nonperiodic_parametric_heat_slab,
    make_parametric_burgers_slab,
    make_parametric_heat_slab,
    make_variable_diffusivity_disk_poisson,
)
from omnibias.pinn.operator.torch.deeponet import (
    DeepONetField,
    DeepONetOperator,
    build_deeponet,
)
from omnibias.pinn.operator.torch.fno import (
    FNO1d,
    FNO2d,
    SpectralConv1d,
    SpectralConv2d,
    build_fno1d,
    build_fno2d,
)
from omnibias.pinn.operator.torch.geometry import (
    encode_geometry,
    encode_geometry_batch,
    probe_grid,
)
from omnibias.pinn.operator.torch.geometry_field import (
    condition_with_geometry,
    evaluate_geometry_batch,
)
from omnibias.pinn.operator.torch.losses import (
    burgers_residual_loss,
    causal_operator_loss,
    data_loss,
    heat_residual_loss,
    heat_residual_loss_fd,
    ks_residual_loss,
    ks_residual_loss_fd,
    parametric_burgers_residual_loss,
    parametric_heat_residual_loss,
)

__all__ = [
    "DeepONetField",
    "DeepONetOperator",
    "FNO1d",
    "FNO2d",
    "OperatorSlab",
    "ParametricOperatorSlab",
    "SpectralConv1d",
    "SpectralConv2d",
    "build_deeponet",
    "build_fno1d",
    "build_fno2d",
    "burgers_residual_loss",
    "causal_operator_loss",
    "condition_with_geometry",
    "data_loss",
    "encode_geometry",
    "encode_geometry_batch",
    "evaluate_geometry_batch",
    "heat_residual_loss",
    "heat_residual_loss_fd",
    "ks_residual_loss",
    "ks_residual_loss_fd",
    "make_burgers_slab",
    "make_heat_slab",
    "make_ks_slab",
    "make_nonperiodic_parametric_burgers_slab",
    "make_nonperiodic_parametric_heat_slab",
    "make_parametric_burgers_slab",
    "make_parametric_heat_slab",
    "make_variable_diffusivity_disk_poisson",
    "parametric_burgers_residual_loss",
    "parametric_heat_residual_loss",
    "probe_grid",
]
