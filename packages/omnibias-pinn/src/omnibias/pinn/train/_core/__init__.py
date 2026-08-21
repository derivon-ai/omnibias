# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Backend-free training diagnostics and schedules for omnibias.pinn.train."""

from __future__ import annotations

from omnibias.pinn.train._core.bands import SpectralBandScheduler
from omnibias.pinn.train._core.causality import (
    CausalityReport,
    causality_index,
    report_causality,
    unlocked_fraction,
)
from omnibias.pinn.train._core.depth_residual import (
    DepthResidualConfig,
    DepthResidualForbidden,
    DepthResidualReport,
    apply_hard_bc_tower,
    hard_bc_mask_tower,
    honesty_payload,
    layer_is_unlocked,
    leibniz_product_tower,
    reject_depth_residual_flood,
)
from omnibias.pinn.train._core.guards import (
    TrivialSolutionVerdict,
    trivial_solution_guard,
)

__all__ = [
    "CausalityReport",
    "DepthResidualConfig",
    "DepthResidualForbidden",
    "DepthResidualReport",
    "SpectralBandScheduler",
    "TrivialSolutionVerdict",
    "apply_hard_bc_tower",
    "causality_index",
    "hard_bc_mask_tower",
    "honesty_payload",
    "layer_is_unlocked",
    "leibniz_product_tower",
    "reject_depth_residual_flood",
    "report_causality",
    "trivial_solution_guard",
    "unlocked_fraction",
]
