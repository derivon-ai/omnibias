# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""PyTorch backend for omnibias-control."""

from __future__ import annotations

from omnibias.control.torch.builders import (
    actuator_box,
    control_affine_cbf_rows,
    lagrangian_cbf_rows,
)
from omnibias.control.torch.filter import cbf_filter, cbf_residual, filter_action
from omnibias.control.torch.rollout import barrier_trace, min_barrier, safe_rollout

__all__ = [
    "actuator_box",
    "barrier_trace",
    "cbf_filter",
    "cbf_residual",
    "control_affine_cbf_rows",
    "filter_action",
    "lagrangian_cbf_rows",
    "min_barrier",
    "safe_rollout",
]
