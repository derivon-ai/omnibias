# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Cone-field hyperbolicity on finite orbit segments."""

from __future__ import annotations

from omnibias.dynamics._core.cone import (
    cat_map_jacobian,
    cat_map_unstable_generator,
    certified_cone_hyperbolicity,
    doubling_map_jacobian,
    rotation_jacobian,
)


def test_doubling_map_is_cone_hyperbolic() -> None:
    result = certified_cone_hyperbolicity(
        [doubling_map_jacobian()],
        [[1.0]],
        eta=1.5,
    )
    assert result.certified
    assert result.entropy_lower > 0.0
    assert result.continuum_claim is False
    assert result.anosov_claim is False


def test_cat_map_expands_unstable_generator() -> None:
    result = certified_cone_hyperbolicity(
        [cat_map_jacobian()],
        [cat_map_unstable_generator()],
        eta=2.0,
    )
    assert result.certified
    assert result.expansions[0] >= 2.0


def test_rotation_is_refused() -> None:
    result = certified_cone_hyperbolicity(
        [rotation_jacobian()],
        [[1.0, 0.0]],
        eta=1.1,
    )
    assert result.certified is False
    assert result.entropy_lower == 0.0
