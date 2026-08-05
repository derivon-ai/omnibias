# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Index bookkeeping for the antisymmetric flux potential (backend-free)."""

from __future__ import annotations

import pytest
from omnibias.pinn._core.fluxform import antisymmetric_pairs, potential_table


def test_pair_count_is_the_antisymmetric_dimension() -> None:
    for d in range(2, 7):
        assert len(antisymmetric_pairs(d)) == d * (d - 1) // 2


def test_pairs_are_sorted_strictly_increasing() -> None:
    pairs = antisymmetric_pairs(4)
    assert pairs == ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    assert all(i < j for i, j in pairs)
    assert list(pairs) == sorted(pairs)


def test_two_axes_is_the_streamfunction_and_three_the_vector_potential() -> None:
    assert antisymmetric_pairs(2) == ((0, 1),)
    assert len(antisymmetric_pairs(3)) == 3


def test_fewer_than_two_axes_rejected() -> None:
    for d in (-1, 0, 1):
        with pytest.raises(ValueError, match="at least 2 axes"):
            antisymmetric_pairs(d)


def test_table_covers_both_orders_with_opposite_signs() -> None:
    table = potential_table(3, ("a", "b", "c"))
    # 3 pairs x 2 orders; the diagonal is absent, which is A^{ii} = 0.
    assert len(table) == 6
    for i, j in antisymmetric_pairs(3):
        name_ij, sign_ij = table[(i, j)]
        name_ji, sign_ji = table[(j, i)]
        assert name_ij == name_ji
        assert sign_ij == 1.0
        assert sign_ji == -1.0
    for i in range(3):
        assert (i, i) not in table


def test_table_assigns_names_in_pair_order() -> None:
    table = potential_table(3, ("a", "b", "c"))
    assert table[(0, 1)] == ("a", 1.0)
    assert table[(0, 2)] == ("b", 1.0)
    assert table[(1, 2)] == ("c", 1.0)


def test_wrong_number_of_names_rejected() -> None:
    with pytest.raises(ValueError, match="independent components"):
        potential_table(3, ("a", "b"))
    with pytest.raises(ValueError, match="independent components"):
        potential_table(2, ("a", "b"))
