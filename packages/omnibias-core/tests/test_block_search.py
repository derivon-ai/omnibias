# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Block / coordinate exact search algebra (theory 08-07)."""

from __future__ import annotations

import pytest
from omnibias.core.block_search import (
    BlockSpec,
    apply_block_step,
    arrangement_w_block,
    last_linear_block,
    ombu_bias_block,
    resolve_block_mask,
    unit_direction_from_mask,
)


def test_last_linear_mask_is_the_tail() -> None:
    spec = last_linear_block(2)
    assert resolve_block_mask(5, spec) == (False, False, False, True, True)


def test_ombu_bias_is_one_slot() -> None:
    spec = ombu_bias_block(n_channels=2, k_biases=3, channel=1, slot=1)
    flags = resolve_block_mask(6, spec)
    assert flags == (False, False, False, False, True, False)


def test_arrangement_w_is_one_row() -> None:
    spec = arrangement_w_block(n_hyperplanes=2, n_features=3, row=1)
    flags = resolve_block_mask(6, spec)
    assert flags == (False, False, False, True, True, True)


def test_unit_direction_uses_probe_then_basis() -> None:
    mask = (True, False, True)
    d = unit_direction_from_mask(mask, probe=(-3.0, 9.0, 0.0))
    assert d[1] == 0.0
    assert d[0] == pytest.approx(-1.0)
    assert d[2] == pytest.approx(0.0)
    d0 = unit_direction_from_mask(mask, probe=(0.0, 1.0, 0.0))
    assert d0 == (1.0, 0.0, 0.0)


def test_apply_block_step() -> None:
    assert apply_block_step((0.0, 0.0), (1.0, 0.0), 1.0) == (1.0, 0.0)


def test_empty_mask_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        unit_direction_from_mask((False, False))
    with pytest.raises(ValueError):
        BlockSpec(kind="last_linear", width=0)
