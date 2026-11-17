# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Exact combinatorics, serialization and refusal tests for selected jets."""

from dataclasses import FrozenInstanceError
from math import factorial, prod

import pytest
from omnibias.core.jet_algebra import JetBudget, JetSpec
from omnibias.core.multi_index import get_multi_index_budget, set_multi_index_budget


def test_directional_normalization_and_config() -> None:
    spec = JetSpec.directional(8)
    assert spec.indices == tuple((i,) for i in range(9))
    assert spec.factorials == tuple(factorial(i) for i in range(9))
    assert spec.degrees == tuple(range(9))
    assert spec.multiplication_terms == 45
    assert JetSpec.from_config(spec.to_config()) == spec
    assert spec.explain()["coefficient_convention"] == "D^alpha f / alpha!"
    with pytest.raises(FrozenInstanceError):
        spec.mode = "other"  # type: ignore[misc]


def test_mixed_union_is_exact_downward_closure() -> None:
    spec = JetSpec.mixed([(4, 0), (2, 2), (0, 4)])
    expected = {(a, b) for a in range(5) for b in range(5)
                if (a <= 4 and b == 0) or (a <= 2 and b <= 2) or (a == 0 and b <= 4)}
    assert spec.indices == tuple(sorted(expected, key=lambda a: (sum(a), a)))
    assert spec.size == 13
    assert spec.index((2, 2)) >= 0
    with pytest.raises(ValueError, match="outside"):
        spec.index((3, 1))
    for k, alpha in enumerate(spec.indices):
        assert len(spec.pairs[k]) == prod(a + 1 for a in alpha)
        assert spec.factorials[k] == prod(factorial(a) for a in alpha)
        for i, j in spec.pairs[k]:
            assert tuple(a + b for a, b in zip(spec.indices[i], spec.indices[j], strict=True)) == alpha
    assert JetSpec.from_config(spec.to_config()) == spec


def test_mixed_rectangular_count_and_zero_order() -> None:
    assert JetSpec.mixed([(3, 2, 1)]).size == 24
    assert JetSpec.mixed([(0, 0)]).indices == ((0, 0),)
    assert JetSpec.directional(0).pairs == (((0, 0),),)
    # Many redundant targets must not enumerate their overlapping boxes.
    assert JetSpec.mixed([(k, 0) for k in range(100)]).size == 100


def test_budgets_reject_before_schedule_and_tensor_allocation() -> None:
    with pytest.raises(ValueError, match="coefficients"):
        JetSpec.directional(10**9)
    with pytest.raises(ValueError, match="multiplication terms"):
        JetSpec.directional(20, JetBudget(max_multiplication_terms=10))
    with pytest.raises(ValueError, match="coefficients"):
        JetSpec.mixed([(10**9, 10**9)])
    with pytest.raises(ValueError, match="coefficients"):
        JetSpec.mixed([(4, 0), (0, 4)], JetBudget(max_coefficients=8))
    spec = JetSpec.directional(2, JetBudget(max_tensor_bytes=48))
    assert spec.check_tensor((2,), 8) == 48
    with pytest.raises(ValueError, match="bytes"):
        spec.check_tensor((3,), 8)


def test_default_budget_is_read_at_construction() -> None:
    previous = get_multi_index_budget()
    try:
        set_multi_index_budget(2)
        assert JetBudget().max_coefficients == 2
        with pytest.raises(ValueError, match="coefficients"):
            JetSpec.directional(2)
    finally:
        set_multi_index_budget(previous)


@pytest.mark.parametrize("order", [-1, 1.5, True])
def test_invalid_directional_order(order: object) -> None:
    with pytest.raises(ValueError):
        JetSpec.directional(order)  # type: ignore[arg-type]


@pytest.mark.parametrize("targets", [[], [()], [(1,), (1, 2)], [(-1, 0)], [(1.2, 0)]])
def test_invalid_mixed_targets(targets: list) -> None:
    with pytest.raises(ValueError):
        JetSpec.mixed(targets)
