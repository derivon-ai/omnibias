# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Collapse-Net on the 01-04 stencil (theory 09-11).

Train uses founding stencil weights from
:mod:`omnibias.difference._core.stencil`. Inference is founding
bias collapse (``delta -> 0``) to ``sigma^(K-1)``. Temperature
collapse (``beta -> inf``, feasibility) does not appear. Do not
conflate the two.
"""

from __future__ import annotations

from omnibias.core.collapse_net import (
    DEFAULT_CONFIG,
    DISCLAIMER,
    CollapseNetConfig,
    collapse_net_forward,
    collapse_remainder,
    collapsed_value,
    founding_spacing,
    honesty_payload,
    sin_skill,
    stencil_value,
    worked_example,
)
from omnibias.core.ftc import sigmoid
from omnibias.difference._core.stencil import stencil_offsets, stencil_signs


def difference_stencil_value(x: float, *, config: CollapseNetConfig | None = None) -> float:
    """Apply the 01-04 central stencil to the named family at ``x``."""
    cfg = DEFAULT_CONFIG if config is None else config
    spacing = founding_spacing(cfg.order, cfg.delta)
    signs = stencil_signs(cfg.order, spacing, "central")
    offsets = stencil_offsets(cfg.order, spacing, "central")
    if cfg.family == "sigmoid":
        field = sigmoid
    elif cfg.family == "sin":
        import math

        field = math.sin
    else:
        raise ValueError(f"unknown family {cfg.family!r}")
    total = 0.0
    for sign, offset in zip(signs, offsets, strict=True):
        total += sign * field(x + offset)
    return total


__all__ = [
    "CollapseNetConfig",
    "DEFAULT_CONFIG",
    "DISCLAIMER",
    "collapse_net_forward",
    "collapse_remainder",
    "collapsed_value",
    "difference_stencil_value",
    "founding_spacing",
    "honesty_payload",
    "sin_skill",
    "stencil_value",
    "worked_example",
]
