# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Backend-free schemas for omnibias.pinn.operator (no torch / jax imports)."""

from __future__ import annotations

from omnibias.pinn.operator._core.sensors import SensorGrid, sample_fourier_ics
from omnibias.pinn.operator._core.spec import OperatorSpec
from omnibias.pinn.operator._core.verified import (
    branch_coefficient_box,
    certify_heat_residual,
    enclose_heat_residual,
)

__all__ = [
    "OperatorSpec",
    "SensorGrid",
    "branch_coefficient_box",
    "certify_heat_residual",
    "enclose_heat_residual",
    "sample_fourier_ics",
]
