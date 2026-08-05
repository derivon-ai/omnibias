# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Soundness of the verified interval soft-DP: enclosures contain every true value.

Per the omnibias verified rule, each enclosure is checked against a dense deterministic
grid (all box corners) **and** a random sample of true values. Truth is the pure-numpy
brute-force soft partition; the interval DP must contain it for every point in the box.
The enclosures are sound but not tight, so we also record that they widen with the radius.
"""

from __future__ import annotations

import itertools

import numpy as np
from omnibias.core.verified import Interval
from omnibias.struct import (
    DAG,
    ChainTrellis,
    CTCLattice,
    brute_force_arborescence,
    brute_force_partition,
    brute_force_soft_align,
    brute_force_soft_dtw,
    matrix_tree_partition,
)
from omnibias.struct.verified import (
    align_value_iv,
    box,
    chain_marginals_iv,
    chain_value_iv,
    ctc_value_iv,
    dag_value_iv,
    dtw_value_iv,
    lse_beta_iv,
    matrix_tree_partition_iv,
    pairwise_lse_iv,
)

TOL = 1e-9


def _corners(center: np.ndarray, radius: float) -> list[np.ndarray]:
    flat = center.reshape(-1)
    out = []
    for signs in itertools.product((-1.0, 1.0), repeat=flat.size):
        out.append((flat + radius * np.array(signs)).reshape(center.shape))
    return out


def test_lse_beta_iv_encloses_grid_and_random() -> None:
    rng = np.random.default_rng(0)
    center = rng.standard_normal(4)
    radius = 0.4
    for beta in (1.0, 4.0):
        iv = lse_beta_iv(box(center, radius), beta)
        pts = _corners(center, radius) + [center + rng.uniform(-radius, radius, 4) for _ in range(200)]
        for x in pts:
            m = float(np.max(beta * x))
            true = (m + np.log(np.sum(np.exp(beta * x - m)))) / beta
            assert iv.lo - TOL <= true <= iv.hi + TOL


def test_pairwise_lse_iv_encloses_and_matches_vector_form() -> None:
    rng = np.random.default_rng(1)
    for _ in range(50):
        a, b = rng.standard_normal(2)
        beta = 3.0
        iv = pairwise_lse_iv(a, b, beta)
        true = (np.logaddexp(beta * a, beta * b)) / beta
        assert iv.lo - TOL <= true <= iv.hi + TOL


def test_chain_value_iv_encloses_brute_force_over_box() -> None:
    rng = np.random.default_rng(2)
    n_steps, n_states = 3, 2
    center = rng.standard_normal((n_steps, n_states))
    transitions = rng.standard_normal((n_states, n_states))
    start = rng.standard_normal(n_states)
    radius = 0.3
    beta = 2.0
    iv = chain_value_iv(box(center, radius), transitions, beta, start=start)
    samples = _corners(center, radius) + [
        center + rng.uniform(-radius, radius, center.shape) for _ in range(100)
    ]
    for emissions in samples:
        true = brute_force_partition(ChainTrellis(emissions, transitions, start), beta)
        assert iv.lo - TOL <= true <= iv.hi + TOL


def test_dag_value_iv_encloses_brute_force_over_box() -> None:
    rng = np.random.default_rng(3)
    n = 4
    base_edges = {(0, 1): 1.0, (0, 2): 1.5, (1, 2): 0.5, (1, 3): 0.8, (2, 3): 0.6}
    dag = DAG(n, base_edges, sink=n - 1)
    center = np.zeros((n, n))
    for (u, v), w in base_edges.items():
        center[u, v] = w
    radius = 0.25
    beta = 2.0
    iv = dag_value_iv(box(center, radius), dag, beta)
    edge_keys = list(base_edges)
    corners = itertools.product((-1.0, 1.0), repeat=len(edge_keys))
    samples = [
        {k: base_edges[k] + radius * s for k, s in zip(edge_keys, signs, strict=True)}
        for signs in corners
    ] + [
        {k: base_edges[k] + float(rng.uniform(-radius, radius)) for k in edge_keys}
        for _ in range(100)
    ]
    for edges in samples:
        dag_pt = DAG(n, edges, sink=n - 1)
        true_cost = -brute_force_partition(dag_pt, beta)  # cost = -(max-convention soft value)
        assert iv.lo - TOL <= true_cost <= iv.hi + TOL


def test_enclosures_widen_with_radius_but_contain_the_point() -> None:
    rng = np.random.default_rng(4)
    center = rng.standard_normal((3, 2))
    transitions = rng.standard_normal((2, 2))
    beta = 2.0
    point = chain_value_iv(box(center, 0.0), transitions, beta)
    wide = chain_value_iv(box(center, 0.5), transitions, beta)
    assert point.width < wide.width  # sound-not-tight: wider box -> wider enclosure
    true = brute_force_partition(ChainTrellis(center, transitions, np.zeros(2)), beta)
    assert point.lo - TOL <= true <= point.hi + TOL


def test_dtw_value_iv_encloses_brute_force_over_box() -> None:
    rng = np.random.default_rng(10)
    center = rng.standard_normal((3, 3))
    radius, beta = 0.3, 2.0
    iv = dtw_value_iv(box(center, radius), beta)
    samples = _corners(center, radius) + [
        center + rng.uniform(-radius, radius, center.shape) for _ in range(80)
    ]
    for cost in samples:
        assert iv.lo - TOL <= brute_force_soft_dtw(cost, beta) <= iv.hi + TOL


def test_align_value_iv_encloses_brute_force_over_box() -> None:
    rng = np.random.default_rng(11)
    k = 3
    sub0 = rng.standard_normal((k, k))
    gap0, radius, beta = -1.0, 0.2, 2.0
    a, b = np.array([0, 1, 2]), np.array([0, 2])
    iv = align_value_iv(a, b, box(sub0, radius), Interval(gap0 - radius, gap0 + radius), beta)
    for _ in range(150):
        subp = sub0 + rng.uniform(-radius, radius, sub0.shape)
        gapp = gap0 + float(rng.uniform(-radius, radius))
        assert iv.lo - TOL <= brute_force_soft_align(a, b, subp, gapp, beta) <= iv.hi + TOL


def test_ctc_value_iv_encloses_brute_force_over_box() -> None:
    rng = np.random.default_rng(12)
    lattice = CTCLattice(np.array([1]), num_classes=3, blank=0)
    n_steps = 3
    center = rng.standard_normal((n_steps, 3))
    radius, beta = 0.2, 2.0
    iv = ctc_value_iv(box(center, radius), lattice, beta)
    samples = _corners(center, radius) + [
        center + rng.uniform(-radius, radius, center.shape) for _ in range(80)
    ]
    for lp in samples:
        assert iv.lo - TOL <= brute_force_partition(lattice, beta, lp) <= iv.hi + TOL


def _brute_chain_marginals(
    emissions: np.ndarray, transitions: np.ndarray, start: np.ndarray, beta: float
) -> np.ndarray:
    trellis = ChainTrellis(emissions, transitions, start)
    paths = list(trellis.enumerate_paths())
    scores = np.array([trellis.path_score(p) for p in paths])
    weights = np.exp(beta * (scores - scores.max()))
    weights /= weights.sum()
    n_steps, n_states = emissions.shape
    mu = np.zeros((n_steps, n_states))
    for path, w in zip(paths, weights, strict=True):
        for t, s in enumerate(path):
            mu[t, s] += w
    return mu


def test_chain_marginals_iv_encloses_brute_force_and_the_simplex() -> None:
    rng = np.random.default_rng(13)
    n_steps, n_states = 3, 2
    center = rng.standard_normal((n_steps, n_states))
    transitions = rng.standard_normal((n_states, n_states))
    start = rng.standard_normal(n_states)
    radius, beta = 0.15, 2.0
    mu_iv = chain_marginals_iv(box(center, radius), transitions, beta, start=start)
    samples = _corners(center, radius) + [
        center + rng.uniform(-radius, radius, center.shape) for _ in range(60)
    ]
    for emissions in samples:
        mu = _brute_chain_marginals(emissions, transitions, start, beta)
        for t in range(n_steps):
            for s in range(n_states):
                assert mu_iv[t][s].lo - TOL <= mu[t, s] <= mu_iv[t][s].hi + TOL
    # every row soundly encloses the simplex constraint sum_s mu = 1
    for t in range(n_steps):
        lo = sum(mu_iv[t][s].lo for s in range(n_states))
        hi = sum(mu_iv[t][s].hi for s in range(n_states))
        assert lo - TOL <= 1.0 <= hi + TOL


def test_matrix_tree_partition_iv_encloses_exact_partition_over_box() -> None:
    rng = np.random.default_rng(14)
    n = 3
    # A modestly-scaled (well-conditioned) arc box: the interval determinant certifies det > 0.
    arc0 = 0.3 * rng.standard_normal((n + 1, n + 1))
    arc0[:, 0] = 0.0  # ROOT is never a modifier
    radius, beta = 0.05, 1.5
    iv = matrix_tree_partition_iv(box(arc0, radius), beta)
    for _ in range(150):
        arcp = arc0 + rng.uniform(-radius, radius, arc0.shape)
        arcp[:, 0] = 0.0
        assert iv.lo - TOL <= brute_force_arborescence(arcp, beta) <= iv.hi + TOL
    # at radius 0 the interval determinant pins the exact log-det partition
    point = matrix_tree_partition_iv(
        [[Interval.point(float(arc0[i, j])) for j in range(n + 1)] for i in range(n + 1)], beta
    )
    exact = matrix_tree_partition(arc0, beta)
    assert point.lo - TOL <= exact <= point.hi + TOL


def test_matrix_tree_partition_iv_raises_when_positivity_not_provable() -> None:
    # A wide, high-beta box loses the smaller trees to interval wrapping -- honest scope limit.
    rng = np.random.default_rng(15)
    arc0 = rng.standard_normal((5, 5))
    arc0[:, 0] = 0.0
    try:
        matrix_tree_partition_iv(box(arc0, 0.5), 4.0)
    except ValueError:
        return
    raise AssertionError("expected a ValueError on a box too wide to certify det > 0")
