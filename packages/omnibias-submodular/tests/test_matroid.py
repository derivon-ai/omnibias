# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Matroids: independence, rank, the LP oracle, and the soft basis hardening."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from omnibias.submodular import PartitionMatroid, UniformMatroid


def test_uniform_independence_and_rank() -> None:
    m = UniformMatroid(5, 2)
    assert m.n == 5
    assert m.rank() == 2
    assert m.is_independent(np.array([1.0, 1.0, 0.0, 0.0, 0.0]))
    assert not m.is_independent(np.array([1.0, 1.0, 1.0, 0.0, 0.0]))


def test_uniform_max_weight_basis_is_top_k_positive() -> None:
    m = UniformMatroid(5, 2)
    w = np.array([0.1, -3.0, 5.0, 2.0, 0.0])
    y = m.max_weight_basis(w)
    assert list(y) == [0.0, 0.0, 1.0, 1.0, 0.0]  # picks 5.0 and 2.0
    # never selects a nonpositive weight even if capacity remains
    y2 = m.max_weight_basis(np.array([1.0, -1.0, -1.0, -1.0, -1.0]))
    assert float(y2.sum()) == 1.0


def test_partition_validation_and_caps() -> None:
    m = PartitionMatroid([[0, 1, 2], [3, 4]], [2, 1])
    assert m.n == 5
    assert m.rank() == 3
    assert m.is_independent(np.array([1.0, 1.0, 0.0, 1.0, 0.0]))
    assert not m.is_independent(np.array([1.0, 1.0, 1.0, 0.0, 0.0]))  # group 0 over cap
    with pytest.raises(ValueError, match="partition"):
        PartitionMatroid([[0, 1], [1, 2]], [1, 1])  # overlap
    with pytest.raises(ValueError, match="cap"):
        PartitionMatroid([[0, 1]], [5])


def test_partition_max_weight_basis_respects_group_caps() -> None:
    m = PartitionMatroid([[0, 1, 2], [3, 4]], [1, 2])
    w = np.array([1.0, 3.0, 2.0, 0.5, 0.7])
    y = m.max_weight_basis(w)
    assert float(y[[0, 1, 2]].sum()) == 1.0  # one from group 0 (the 3.0)
    assert y[1] == 1.0
    assert float(y[[3, 4]].sum()) == 2.0  # both from group 1


@pytest.mark.parametrize("matroid", [UniformMatroid(6, 3), PartitionMatroid([[0, 1, 2], [3, 4, 5]], [1, 2])])
def test_soft_basis_hardens_to_hard_basis(matroid) -> None:
    # The soft oracle mirrors the LP oracle in its operating regime: nonnegative weights
    # (a monotone-submodular gradient). Its 0.5 level set matches the hard basis at any
    # beta, and the values saturate onto it as beta -> inf (the feasibility collapse).
    rng = np.random.default_rng(0)
    for _ in range(10):
        w = rng.permutation(matroid.n).astype(float) + 1.0  # distinct, well-separated, > 0
        hard = matroid.max_weight_basis(w)
        assert np.array_equal((matroid.soft_basis(w, 30.0) > 0.5).astype(float), hard)
        assert np.max(np.abs(matroid.soft_basis(w, 2000.0) - hard)) < 1e-3
        # monotone hardening: larger beta is never farther from the hard basis
        far = np.max(np.abs(matroid.soft_basis(w, 5.0) - hard))
        near = np.max(np.abs(matroid.soft_basis(w, 200.0) - hard))
        assert near <= far + 1e-12


def test_max_weight_basis_matches_brute_force() -> None:
    m = PartitionMatroid([[0, 1, 2], [3, 4]], [2, 1])
    rng = np.random.default_rng(1)
    for _ in range(20):
        w = rng.standard_normal(5)
        y = m.max_weight_basis(w)
        best_val = -np.inf
        for bits in itertools.product([0, 1], repeat=5):
            x = np.array(bits, dtype=float)
            if m.is_independent(x):
                best_val = max(best_val, float(w @ x))
        assert abs(float(w @ y) - best_val) < 1e-9
