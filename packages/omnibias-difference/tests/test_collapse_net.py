# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Collapse-Net stencil matches the 01-04 difference signs."""

from __future__ import annotations

from omnibias.core.collapse_net import CollapseNetConfig, stencil_value, worked_example
from omnibias.difference._core.collapse_net import difference_stencil_value


def test_g1_matches_difference_stencil() -> None:
    ex = worked_example()
    cfg = CollapseNetConfig(order=1, delta=0.1, family="sigmoid")
    assert difference_stencil_value(0.0, config=cfg) == stencil_value(0.0, config=cfg)
    assert abs(difference_stencil_value(0.0, config=cfg) - ex["expected_stencil"]) < 1e-15


def test_sin_matches() -> None:
    cfg = CollapseNetConfig(order=1, delta=0.1, family="sin")
    assert difference_stencil_value(0.3, config=cfg) == stencil_value(0.3, config=cfg)
