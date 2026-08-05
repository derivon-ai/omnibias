# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""PyTorch backend for omnibias-submodular (the differentiable continuous-greedy twin)."""

from __future__ import annotations

from omnibias.submodular.torch.relaxation import (
    budget_multilinear,
    budget_relaxation,
    continuous_greedy,
    coverage_multilinear,
    coverage_relaxation,
    facility_multilinear,
    facility_relaxation,
    graphcut_multilinear,
    submodular_relaxation,
)

__all__ = [
    "budget_multilinear",
    "budget_relaxation",
    "continuous_greedy",
    "coverage_multilinear",
    "coverage_relaxation",
    "facility_multilinear",
    "facility_relaxation",
    "graphcut_multilinear",
    "submodular_relaxation",
]
