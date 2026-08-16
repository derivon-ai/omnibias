# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Finite multi-spacing a² fit. Not a YM continuum certificate."""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.geometry.gauge._core.continuum_fit import (
    extrapolate_in_a2,
    scale_from_string_tension,
)


def test_planted_a2_series_earns_fit_claim() -> None:
    a = np.array([0.20, 0.18, 0.16, 0.14, 0.12, 0.10])
    y = 1.2 + 0.3 * a * a
    out = extrapolate_in_a2(a, y)
    assert out.intercept == pytest.approx(1.2, rel=1e-6)
    assert out.slope == pytest.approx(0.3, rel=1e-6)
    assert out.multi_beta_gate_passed is True
    assert out.continuum_claim is True
    assert out.yang_mills_claim is False
    assert "not a→0" in out.scope


def test_constant_series_does_not_beat_constant_baseline() -> None:
    a = np.array([0.20, 0.18, 0.16, 0.14, 0.12, 0.10])
    y = np.full(a.shape, 1.7)
    out = extrapolate_in_a2(a, y)
    assert out.yang_mills_claim is False
    assert out.skill == pytest.approx(0.0, abs=1e-12) or out.continuum_claim is False


def test_scale_from_string_tension() -> None:
    assert scale_from_string_tension(0.04, spacing=0.5) == pytest.approx(0.1)
