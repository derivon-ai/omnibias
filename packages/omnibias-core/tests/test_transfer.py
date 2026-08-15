# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Layered transfer G1–G3 (theory 02-11). Distinct from geometry.gauge.transfer."""

from __future__ import annotations

import pytest
from omnibias.core.transfer import (
    Layer,
    bloch_dispersion,
    certified_band_gap,
    quarter_wave_stack,
    reflection_transmission,
    stack_matrix,
    unitarity_residual,
)


def test_g1_structural_identities() -> None:
    layers = (
        Layer(1.5 + 0.0j, 0.3),
        Layer(2.3 + 0.0j, 0.2),
        Layer(1.2 + 0.0j, 0.4),
    )
    for omega in (0.4, 1.0, 2.2):
        m = stack_matrix(layers, omega)
        assert unitarity_residual(m) <= 1e-12
        r, t = reflection_transmission(m)
        cons = abs(r) ** 2 + abs(t) ** 2
        assert abs(cons - 1.0) <= 1e-13


def test_unitarity_refused_when_lossy() -> None:
    with pytest.raises(ValueError, match="lossless"):
        unitarity_residual(stack_matrix((Layer(1.5 + 0.1j, 0.2),), 1.0), lossless=False)


def test_g2_quarter_wave_midgap() -> None:
    cell = quarter_wave_stack(2.0, 1.0, n_periods=1, omega0=1.0)
    mid = abs(bloch_dispersion(cell, 1.0))
    assert mid > 1.0


def test_g3_certified_gap_sound() -> None:
    cell = quarter_wave_stack(2.3, 1.0, n_periods=1, omega0=1.0)
    cert = certified_band_gap(cell, omega_range=(0.85, 1.15), n_grid=48)
    assert cert.continuum_claim is False
    xs = [0.85 + (1.15 - 0.85) * i / 80 for i in range(81)]
    vals = [abs(bloch_dispersion(cell, w)) for w in xs]
    if cert.is_gap:
        assert min(vals) >= 1.0 - 1e-9
    assert cert.trace_half_lo <= min(vals) + 1e-9
    assert cert.trace_half_hi + 1e-9 >= max(vals)
