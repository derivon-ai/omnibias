# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Operator-typed composite blocks built on :class:`OperatorMultiBiasUnit`."""

from omnibias.torch.blocks.conv import (
    AnalyticGaussianConv1d,
    AnalyticGaussianConv2d,
    analytic_gaussian_taps,
    cmbConv1d,
    cmbConv2d,
)
from omnibias.torch.blocks.linear import cmbLinear
from omnibias.torch.blocks.operator import OperatorBlock

__all__ = [
    "AnalyticGaussianConv1d",
    "AnalyticGaussianConv2d",
    "OperatorBlock",
    "analytic_gaussian_taps",
    "cmbConv1d",
    "cmbConv2d",
    "cmbLinear",
]
