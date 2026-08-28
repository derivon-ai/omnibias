# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Fixed-box Gaussian de Bruijn-style heat-kernel enclosure checks."""

from __future__ import annotations

import random

import pytest

mp = pytest.importorskip("mpmath")

from omnibias.core.collapse.winding import winding_enclosure_function
from omnibias.core.verified.complex_interval import ComplexInterval
from omnibias.core.verified.heat_kernel import (
    GaussianHeatKernelContract,
    count_gaussian_heat_kernel_zeros,
    gaussian_heat_kernel_enclosure,
)


def _reference(z: complex, t: float) -> complex:
    """Closed form of the Gaussian fixture's full half-line integral."""
    a = mp.mpf(1) - mp.mpf(t)
    zz = mp.mpc(z.real, z.imag)
    return complex(mp.sqrt(mp.pi) / (2 * mp.sqrt(a)) * mp.exp(-(zz * zz) / (4 * a)))


def test_generic_winding_counts_exactly_countable_identity_fixture() -> None:
    winding = winding_enclosure_function(
        lambda z: z,
        0j,
        1.0,
        contour="rectangle",
        half_width=2.0,
        half_height=1.0,
        segments=64,
    )
    assert winding is not None
    assert winding.contains(1.0)


def test_gaussian_kernel_enclosure_contains_grid_and_random_references() -> None:
    contract = GaussianHeatKernelContract(
        t=0.0,
        center=0j,
        half_width=0.5,
        half_height=0.5,
        truncation=6.0,
        panels=64,
    )
    grid = [complex(x, y) for x in (-0.5, -0.1, 0.25) for y in (-0.4, 0.0, 0.5)]
    rng = random.Random(20260827)
    points = grid + [
        complex(rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5)) for _ in range(16)
    ]
    for z in points:
        enclosure = gaussian_heat_kernel_enclosure(ComplexInterval.point(z), contract)
        reference = _reference(z, contract.t)
        assert enclosure.contains(reference)


def test_gaussian_kernel_certifies_zero_count_on_one_fixed_rectangle() -> None:
    contract = GaussianHeatKernelContract(
        t=0.0,
        center=0j,
        half_width=0.25,
        half_height=0.25,
        truncation=6.0,
        panels=64,
        contour_segments=16,
    )
    result = count_gaussian_heat_kernel_zeros(contract)
    assert result.certified
    assert result.count == 0
    assert result.winding is not None
    assert result.winding.contains(0.0)
    assert "Lambda" not in result.detail


def test_contract_rejects_unproved_tail_regime() -> None:
    with pytest.raises(ValueError, match="2\\(1-t\\)U"):
        GaussianHeatKernelContract(
            t=0.9,
            center=0j,
            half_width=1.0,
            half_height=1.0,
            truncation=1.0,
        )
