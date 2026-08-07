# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""JAX neural-operator drivers for omnibias.pinn.operator."""

from __future__ import annotations

from omnibias.pinn.operator.jax.data import (
    OperatorSlab,
    make_burgers_slab,
    make_heat_slab,
    make_ks_slab,
)
from omnibias.pinn.operator.jax.deeponet import (
    DeepONetField,
    DeepONetOperator,
    make_deeponet,
)
from omnibias.pinn.operator.jax.fno import FNO1d, SpectralConv1d, make_fno1d
from omnibias.pinn.operator.jax.losses import (
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
    "burgers_residual_loss",
    "data_loss",
    "heat_residual_loss",
    "heat_residual_loss_fd",
    "ks_residual_loss",
    "ks_residual_loss_fd",
    "make_burgers_slab",
    "make_deeponet",
    "make_fno1d",
    "make_heat_slab",
    "make_ks_slab",
]
