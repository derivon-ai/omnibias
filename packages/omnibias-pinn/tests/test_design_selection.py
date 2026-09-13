# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""D/A/E selection is checked against independent complete subset oracles."""

from itertools import combinations

import numpy as np
import pytest
from omnibias.pinn.inverse.design import design_score, select_design


def independent_score(matrix, criterion):
    eigenvalues = np.linalg.eigvalsh(matrix)
    if criterion == "D":
        return float(np.log(eigenvalues).sum())
    if criterion == "A":
        return -float((1 / eigenvalues).sum())
    return float(eigenvalues[0])


@pytest.mark.parametrize("criterion", ["D", "A", "E"])
def test_exhaustive_and_greedy_selection_against_complete_independent_oracle(criterion):
    factors = np.random.default_rng(612).normal(size=(7, 2, 3))
    blocks = np.einsum("cmi,cmj->cij", factors, factors)
    prior = np.diag([0.7, 1.1, 1.6])
    entries = [
        (independent_score(prior + blocks[list(indices)].sum(axis=0), criterion), indices)
        for indices in combinations(range(7), 3)
    ]
    expected_score, expected_indices = max(entries, key=lambda row: row[0])
    exhaustive = select_design(
        blocks, prior, 3, criterion=criterion, method="exhaustive", max_subsets=35
    )
    assert exhaustive.status == "selected" and exhaustive.enumeration_complete
    assert exhaustive.indices == expected_indices
    assert exhaustive.evaluated_subsets == exhaustive.required_subsets == 35
    assert exhaustive.score == pytest.approx(expected_score, abs=1e-11)
    assert not exhaustive.certified
    greedy = select_design(blocks, prior, 3, criterion=criterion)
    assert greedy.status == "selected" and not greedy.enumeration_complete
    assert len(set(greedy.indices)) == 3 and greedy.weights.sum() == 3
    assert greedy.evaluated_subsets == greedy.required_subsets == 7 + 6 + 5
    assert greedy.score <= expected_score + 1e-11
    expected_greedy = independent_score(prior + blocks[list(greedy.indices)].sum(axis=0), criterion)
    assert greedy.score == pytest.approx(expected_greedy, abs=1e-11)


@pytest.mark.parametrize("criterion", ["D", "A", "E"])
@pytest.mark.parametrize("method", ["greedy", "exhaustive"])
def test_ties_and_empty_budget_are_deterministic(criterion, method):
    blocks = np.repeat(np.eye(2)[None], 5, axis=0)
    prior = np.eye(2)
    result = select_design(blocks, prior, 2, criterion=criterion, method=method)
    assert result.indices == (0, 1)
    empty = select_design(blocks, prior, 0, criterion=criterion, method=method, max_subsets=1)
    assert empty.status == "selected" and empty.indices == ()
    assert empty.weights.sum() == 0 and empty.evaluated_subsets == 1
    assert empty.score == pytest.approx(independent_score(prior, criterion))


@pytest.mark.parametrize("method,required", [("exhaustive", 252), ("greedy", 40)])
def test_evaluation_budget_refusal_is_explicit_and_has_no_partial_design(method, required):
    blocks = np.repeat(np.eye(2)[None], 10, axis=0)
    result = select_design(blocks, np.eye(2), 5, method=method, max_subsets=required - 1)
    assert result.status == "budget_exceeded" and not result.enumeration_complete
    assert result.indices == () and result.weights is None and result.score is None
    assert result.evaluated_subsets == 0 and result.required_subsets == required


def test_selector_reuses_information_validation_and_rejects_invalid_options():
    blocks = np.repeat(np.eye(2)[None], 3, axis=0)
    with pytest.raises(ValueError, match="positive definite"):
        select_design(blocks, np.zeros((2, 2)), 2)
    with pytest.raises(ValueError, match="semidefinite"):
        select_design(-blocks, np.eye(2), 2)
    for options in ({"criterion": "unknown"}, {"method": "unknown"}, {"max_subsets": -1}):
        with pytest.raises(ValueError):
            select_design(blocks, np.eye(2), 2, **options)
    for budget in (-1, 4, 1.5, True):
        with pytest.raises(ValueError, match="budget"):
            select_design(blocks, np.eye(2), budget)


@pytest.mark.parametrize("criterion", ["D", "A", "E"])
@pytest.mark.parametrize("method", ["greedy", "exhaustive"])
def test_finite_inputs_whose_total_information_overflows_are_refused(criterion, method):
    with pytest.raises(ValueError, match="finite total information"):
        select_design(
            np.array([[[1e308]]]), np.array([[1e308]]), 1, criterion=criterion, method=method
        )


def test_nonfinite_objective_is_refused_even_when_information_is_finite():
    with pytest.raises(ValueError, match="finite design scores"):
        design_score(np.zeros((1, 1, 1)), np.ones(1), np.array([[1e-320]]), criterion="A")
