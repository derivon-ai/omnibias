# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Finite Ramsey colouring and saturation smokes."""

from __future__ import annotations

import pytest
from omnibias.combinatorics.ramsey import (
    is_saturated,
    is_triangle_free_edge_colouring,
    pentagon_two_colouring,
    verify_pentagon_colouring,
    verify_saturated_smoke,
)


def test_pentagon_is_triangle_free() -> None:
    colours = pentagon_two_colouring()
    assert is_triangle_free_edge_colouring(5, 2, colours)
    cert = verify_pentagon_colouring()
    assert cert["replay_ok"] is True
    assert cert["honesty"]["erdos_183_claim"] is False
    assert cert["honesty"]["ramsey_colouring_replay"] is True


def test_monochrome_triangle_is_rejected() -> None:
    colours = {(i, j): 0 for i in range(3) for j in range(i + 1, 3)}
    assert is_triangle_free_edge_colouring(3, 1, colours) is False


def test_n_cap() -> None:
    with pytest.raises(ValueError, match="n_cap"):
        is_triangle_free_edge_colouring(9, 2, {}, n_cap=8)


def test_saturated_smoke() -> None:
    assert is_saturated(2, 2, 1, ((0, 1),))
    assert is_saturated(3, 2, 1, ((0, 1, 2),))
    assert is_saturated(3, 2, 1, ((0, 0, 0),)) is False
    cert = verify_saturated_smoke()
    assert cert["replay_ok"] is True
    assert cert["honesty"]["erdos_183_claim"] is False
