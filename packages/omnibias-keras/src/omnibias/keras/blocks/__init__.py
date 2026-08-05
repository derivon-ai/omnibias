# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Operator-typed Keras layers."""

from omnibias.keras.blocks.conv import (
    AnalyticGaussianConv1D,
    AnalyticGaussianConv2D,
    analytic_gaussian_taps,
    cmbConv1D,
    cmbConv2D,
)
from omnibias.keras.blocks.linear import cmbDense
from omnibias.keras.blocks.operator import OperatorBlock, OpName

__all__ = [
    "AnalyticGaussianConv1D",
    "AnalyticGaussianConv2D",
    "OpName",
    "OperatorBlock",
    "analytic_gaussian_taps",
    "cmbConv1D",
    "cmbConv2D",
    "cmbDense",
]
