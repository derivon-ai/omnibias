# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Tests for omnibias.core.scan (theory 01-02 bank algebra)."""

from __future__ import annotations

import math

import pytest
from omnibias.core.scan import BankSpec


def test_uniform_spacing() -> None:
    bank = BankSpec.uniform(-1.0, 1.0, 5)
    assert bank.offsets == pytest.approx((-1.0, -0.5, 0.0, 0.5, 1.0))
    assert bank.spacing == pytest.approx(0.5)
    assert bank.n_offsets == 5
    assert bank.n_scales == 1
    assert bank.scales == (1.0,)
    assert bank.min_separation() == pytest.approx(0.5)


def test_non_uniform_spacing_is_none() -> None:
    bank = BankSpec(offsets=(0.0, 0.1, 0.4))
    assert bank.spacing is None
    assert bank.min_separation() == pytest.approx(0.1)


def test_single_offset_spacing_none() -> None:
    bank = BankSpec(offsets=(0.25,))
    assert bank.spacing is None
    assert math.isinf(bank.min_separation())


def test_uniform_rejects_degenerate() -> None:
    with pytest.raises(ValueError, match="n >= 2"):
        BankSpec.uniform(-1.0, 1.0, 1)
    with pytest.raises(ValueError, match="differ"):
        BankSpec.uniform(0.0, 0.0, 3)


def test_rejects_empty_and_non_finite() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        BankSpec(offsets=())
    with pytest.raises(ValueError, match="finite"):
        BankSpec(offsets=(0.0, float("nan")))
    with pytest.raises(ValueError, match="positive"):
        BankSpec(offsets=(0.0, 1.0), scales=(0.0,))
