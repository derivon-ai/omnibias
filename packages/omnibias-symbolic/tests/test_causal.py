# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Tests for causal parent-ranking (MI screening + NOTEARS-lite acyclicity)."""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.symbolic.causal import (
    causal_discovery_report,
    mutual_information_matrix,
    notears_acyclicity,
    notears_lite,
    term_parent_ranking,
)


def _chain_sem(
    *, n: int = 4000, noise: float = 0.3, seed: int = 0
) -> np.ndarray:
    r"""Linear-Gaussian chain ``x0 -> x1 -> x2`` with equal noise variances.

    Equal-noise + raw (un-standardised) scales is the direction-identifiable
    case for linear SEMs, so the learner should recover the *directed* chain.
    """
    rng = np.random.default_rng(seed)
    x0 = rng.standard_normal(n)
    x1 = 2.0 * x0 + noise * rng.standard_normal(n)
    x2 = -1.5 * x1 + noise * rng.standard_normal(n)
    return np.stack([x0, x1, x2], axis=1)


# --------------------------------------------------------------------------- #
# acyclicity functional                                                       #
# --------------------------------------------------------------------------- #
def test_acyclicity_zero_for_dag_positive_for_cycle() -> None:
    dag = np.triu(np.ones((3, 3)), 1)  # strictly upper triangular = acyclic
    cycle = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    h_dag, _ = notears_acyclicity(dag)
    h_cycle, _ = notears_acyclicity(cycle)
    assert h_dag == pytest.approx(0.0, abs=1e-9)
    assert h_cycle > 1e-3
    # a self-loop is also a cycle
    h_self, _ = notears_acyclicity(np.diag([0.0, 1.0, 0.0]))
    assert h_self > 1e-3


def test_acyclicity_gradient_matches_finite_differences() -> None:
    rng = np.random.default_rng(3)
    w = rng.standard_normal((4, 4)) * 0.4
    _, grad = notears_acyclicity(w)
    eps = 1e-6
    for i in range(4):
        for j in range(4):
            wp = w.copy()
            wm = w.copy()
            wp[i, j] += eps
            wm[i, j] -= eps
            fd = (notears_acyclicity(wp)[0] - notears_acyclicity(wm)[0]) / (2 * eps)
            assert fd == pytest.approx(grad[i, j], abs=1e-4)


def test_acyclicity_rejects_non_square() -> None:
    with pytest.raises(ValueError):
        notears_acyclicity(np.zeros((2, 3)))


# --------------------------------------------------------------------------- #
# mutual-information backbone                                                  #
# --------------------------------------------------------------------------- #
def test_mutual_information_matrix_symmetric_zero_diag_and_detects_dependence() -> None:
    rng = np.random.default_rng(1)
    n = 4000
    a = rng.standard_normal(n)
    b = a + 0.05 * rng.standard_normal(n)  # strongly dependent on a
    c = rng.standard_normal(n)  # independent
    mat = mutual_information_matrix(np.stack([a, b, c], axis=1))
    assert np.allclose(np.diag(mat), 0.0)
    assert np.allclose(mat, mat.T)
    assert mat[0, 1] > mat[0, 2]
    assert mat[0, 1] > mat[1, 2]
    # independent pair near zero (bias-corrected)
    assert mat[0, 2] < 0.05


# --------------------------------------------------------------------------- #
# NOTEARS-lite structure learning                                             #
# --------------------------------------------------------------------------- #
def test_notears_lite_recovers_directed_chain() -> None:
    res = notears_lite(_chain_sem(), w_threshold=0.5)
    assert res["acyclicity"] < 1e-6
    support = res["support"]
    weights = res["weights"]
    # exactly the chain edges 0->1 and 1->2 survive a 0.5 threshold
    expected = np.zeros((3, 3), dtype=bool)
    expected[0, 1] = True
    expected[1, 2] = True
    assert np.array_equal(support, expected)
    # and with the right signs / magnitudes
    assert weights[0, 1] == pytest.approx(2.0, abs=0.2)
    assert weights[1, 2] == pytest.approx(-1.5, abs=0.2)
    # no self loops
    assert np.allclose(np.diag(weights), 0.0)


def test_notears_lite_top_ranked_edges_are_the_chain() -> None:
    res = notears_lite(_chain_sem())
    w = res["weights"]
    ranked = sorted(
        ((i, j) for i in range(3) for j in range(3)),
        key=lambda ij: -abs(w[ij[0], ij[1]]),
    )
    assert set(ranked[:2]) == {(0, 1), (1, 2)}


def test_notears_lite_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError):
        notears_lite(np.zeros((5,)))
    with pytest.raises(ValueError):
        notears_lite(np.zeros((1, 3)))  # too few samples


# --------------------------------------------------------------------------- #
# parent ranking + report                                                     #
# --------------------------------------------------------------------------- #
def test_term_parent_ranking_puts_true_parents_on_top() -> None:
    rng = np.random.default_rng(2)
    n = 4000
    a = rng.standard_normal(n)
    b = rng.standard_normal(n)
    c = rng.standard_normal(n)  # irrelevant term
    y = 2.0 * a - 1.5 * b + 0.2 * rng.standard_normal(n)
    out = term_parent_ranking(np.stack([a, b, c], axis=1), y, ["a", "b", "c"])

    # the irrelevant candidate ranks last in every ranking
    assert out["mi_ranking"][-1][0] == "c"
    assert out["notears_ranking"][-1][0] == "c"
    assert out["combined_ranking"][-1][0] == "c"
    # the two true parents take the top-2 combined slots
    assert {out["combined_ranking"][0][0], out["combined_ranking"][1][0]} == {"a", "b"}
    # directed weights carry the SEM signs
    assert out["notears_parent_weights"]["a"] == pytest.approx(2.0, abs=0.25)
    assert out["notears_parent_weights"]["b"] == pytest.approx(-1.5, abs=0.25)
    assert abs(out["notears_parent_weights"]["c"]) < 0.3
    # honesty: never claims a certified DAG
    assert "not a certified DAG" in out["note"]


def test_term_parent_ranking_validates_inputs() -> None:
    x = np.zeros((10, 2))
    with pytest.raises(ValueError):
        term_parent_ranking(x, np.zeros(10), ["only_one_name"])
    with pytest.raises(ValueError):
        term_parent_ranking(x, np.zeros(9), ["a", "b"])  # sample mismatch


def test_causal_discovery_report_recovers_chain_edges() -> None:
    rep = causal_discovery_report(_chain_sem(), ["x0", "x1", "x2"], w_threshold=0.5)
    assert rep["acyclicity"] < 1e-6
    edge_pairs = {(src, dst) for src, dst, _ in rep["edges"]}
    assert edge_pairs == {("x0", "x1"), ("x1", "x2")}
    # edges are sorted by descending |weight|
    mags = [abs(w) for _, _, w in rep["edges"]]
    assert mags == sorted(mags, reverse=True)
    # MI matrix and parents are exposed; honesty note present
    assert rep["mutual_information_matrix"].shape == (3, 3)
    assert set(rep["parents"]) == {"x0", "x1", "x2"}
    assert "not a certified DAG" in rep["note"]


def test_causal_discovery_report_validates_names() -> None:
    with pytest.raises(ValueError):
        causal_discovery_report(np.zeros((10, 3)), ["a", "b"])
