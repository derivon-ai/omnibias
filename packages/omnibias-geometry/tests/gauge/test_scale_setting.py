# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Sommer r0 and planted Wilson-flow t0/w0."""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.geometry.gauge._core.scale_setting import (
    sommer_r0,
    t0_from_energy_curve,
    w0_from_energy_curve,
)


def test_sommer_r0_from_linear_potential() -> None:
    sigma = 0.2
    radii = np.linspace(0.5, 5.0, 20)
    force = np.full_like(radii, sigma)
    out = sommer_r0(radii, force)
    assert out.value == pytest.approx((1.65 / sigma) ** 0.5, rel=1e-3)
    assert out.yang_mills_claim is False
    assert out.continuum_claim is False


def test_planted_energy_curve_t0() -> None:
    t = np.linspace(0.05, 3.0, 40)
    energy = 2.0 * np.exp(-t)
    t0 = t0_from_energy_curve(t, energy)
    assert t0.value > 0.0
    assert (t0.value**2) * (2.0 * np.exp(-t0.value)) == pytest.approx(0.3, rel=5e-2)
    w0 = w0_from_energy_curve(t, energy)
    assert w0.value > 0.0
    assert t0.yang_mills_claim is False
