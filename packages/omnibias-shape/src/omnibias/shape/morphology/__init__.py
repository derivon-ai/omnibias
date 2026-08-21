# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Differentiable mathematical morphology (theory 03-05).

``beta -> inf`` is temperature collapse (feasibility), not the
founding bias collapse (``delta -> 0``). Do not conflate the two.
Pack structuring elements use the founding tower. Soft dilation is
a conservative upper bound. Not a seventh OperatorBlock role.
"""

from __future__ import annotations

from omnibias.shape.morphology._core import (
    MorphResult,
    StructuringElement,
    closing,
    dilate,
    erode,
    exact_distance_transform,
    hard_dilate,
    hard_erode,
    honesty_payload,
    morphological_gradient,
    morphology_gap_bound,
    named_worked_signal,
    named_worked_soft_center,
    opening,
    soft_distance_transform,
    soft_max_pool,
    soft_support,
    top_hat,
)

__all__ = [
    "MorphResult",
    "StructuringElement",
    "closing",
    "dilate",
    "erode",
    "exact_distance_transform",
    "hard_dilate",
    "hard_erode",
    "honesty_payload",
    "morphological_gradient",
    "morphology_gap_bound",
    "named_worked_signal",
    "named_worked_soft_center",
    "opening",
    "soft_distance_transform",
    "soft_max_pool",
    "soft_support",
    "top_hat",
]
