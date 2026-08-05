# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Exact (weighted) model count: manual checks + independent enumeration oracle."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from omnibias.logic import exact_model_count, model_count


def _brute_force_count(problem: object) -> float:
    """Independent oracle: enumerate the cube and count zero-energy (model) points."""
    n = problem.n  # type: ignore[attr-defined]
    rows = np.array(list(itertools.product([0.0, 1.0], repeat=n)))
    energies = np.asarray(problem.energy(rows))  # type: ignore[attr-defined]
    return float(np.count_nonzero(energies == 0.0))


def test_exact_count_manual() -> None:
    # models of (x0 or ~x1) and (x1 or x2) and (~x0 or ~x2): exactly {001, 110}.
    mc = model_count([[1, -2], [2, 3], [-1, -3]])
    assert exact_model_count(mc) == 2.0
    assert _brute_force_count(mc) == 2.0


def test_unsatisfiable_has_zero_models() -> None:
    mc = model_count([[1], [-1]])  # x0 and ~x0
    assert exact_model_count(mc) == 0.0


def test_single_clause_counts_all_but_the_falsifier() -> None:
    mc = model_count([[1, 2]], n_vars=2)  # only 00 falsifies -> 3 models
    assert exact_model_count(mc) == 3.0
    assert _brute_force_count(mc) == 3.0


def test_weighted_model_count() -> None:
    # weights [w0, w1] = [1, 2] per variable; models {001, 110}
    # 001 -> 1*1*2 = 2, 110 -> 2*2*1 = 4, total weighted count 6.
    mc = model_count([[1, -2], [2, 3], [-1, -3]], weights=np.array([[1.0, 2.0]] * 3))
    assert mc.is_weighted
    assert exact_model_count(mc) == pytest.approx(6.0)


def test_energy_is_zero_iff_model() -> None:
    mc = model_count([[1, -2], [2, 3], [-1, -3]])
    assert mc.is_model(np.array([0.0, 0.0, 1.0]))  # a model
    assert float(mc.energy(np.array([0.0, 0.0, 1.0]))) == 0.0
    assert not mc.is_model(np.array([0.0, 1.0, 0.0]))  # falsifies clause 1
    assert float(mc.energy(np.array([0.0, 1.0, 0.0]))) > 0.0


def test_independent_enumeration_agreement_across_seeds() -> None:
    for seed in range(12):
        rng = np.random.default_rng(seed)
        n = int(rng.integers(3, 7))
        m = int(rng.integers(2, 8))
        clauses = []
        for _ in range(m):
            k = int(rng.integers(1, 4))
            variables = rng.choice(np.arange(1, n + 1), size=min(k, n), replace=False)
            signs = rng.choice([-1, 1], size=len(variables))
            clauses.append([int(s * v) for s, v in zip(signs, variables, strict=True)])
        mc = model_count(clauses, n_vars=n)
        assert exact_model_count(mc) == _brute_force_count(mc)


def test_exact_count_rejects_large_n() -> None:
    mc = model_count([[1, 2]], n_vars=6)
    with pytest.raises(ValueError, match="exponential"):
        exact_model_count(mc, max_n=4)


def test_weights_shape_is_validated() -> None:
    with pytest.raises(ValueError, match="shape"):
        model_count([[1, 2]], weights=np.array([1.0, 2.0]), n_vars=2)
    with pytest.raises(ValueError, match="nonnegative"):
        model_count([[1, 2]], weights=np.array([[1.0, -0.5], [1.0, 1.0]]), n_vars=2)
