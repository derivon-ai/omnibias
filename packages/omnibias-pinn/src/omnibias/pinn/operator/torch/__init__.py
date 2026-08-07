# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""PyTorch neural-operator drivers for omnibias.pinn.operator."""

from __future__ import annotations

from omnibias.pinn.operator.torch.data import (
    OperatorSlab,
    make_burgers_slab,
    make_heat_slab,
    make_ks_slab,
)
from omnibias.pinn.operator.torch.deeponet import (
    DeepONetField,
    DeepONetOperator,
    build_deeponet,
)
from omnibias.pinn.operator.torch.fno import FNO1d, SpectralConv1d, build_fno1d
from omnibias.pinn.operator.torch.losses import (
    burgers_residual_loss,
    data_loss,
    heat_residual_loss,
    heat_residual_loss_fd,
    ks_residual_loss,
    ks_residual_loss_fd,
)

__all__ = [
    "DeepONetField",
    "DeepONetOperator",
    "FNO1d",
    "OperatorSlab",
    "SpectralConv1d",
    "build_deeponet",
    "build_fno1d",
    "burgers_residual_loss",
    "data_loss",
    "heat_residual_loss",
    "heat_residual_loss_fd",
    "ks_residual_loss",
    "ks_residual_loss_fd",
    "make_burgers_slab",
    "make_heat_slab",
    "make_ks_slab",
]
