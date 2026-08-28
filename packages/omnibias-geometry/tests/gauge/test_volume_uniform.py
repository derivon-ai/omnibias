# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Volume-uniform strong-coupling polymer family (not continuum YM)."""

from __future__ import annotations

from omnibias.geometry.gauge.transfer.strong_coupling import (
    volume_uniform_strong_coupling_family,
)


def test_volume_uniform_family_at_locked_beta() -> None:
    family = volume_uniform_strong_coupling_family(0.1, spacetime_dims=(2, 3, 4))
    assert family.yang_mills_claim is False
    assert family.continuum_claim is False
    assert family.spacetime_dims == (2, 3, 4)
    assert len(family.gaps) == 3
    assert family.min_gap == min(family.gaps)
    # Small beta is inside the polymer domain for these dimensions.
    assert family.certified
    assert family.min_gap > 0.0
