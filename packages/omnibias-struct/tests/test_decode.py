# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Certified decoding: the winner-vs-runner-up margin over an emission eps-ball is sound.

Soundness is checked operationally against the brute-force oracle: whenever the certificate
reports ``certified`` (``margin.lo > 0``), the Viterbi winner is the *unique* argmax for
every emission matrix in the box (dense corner grid + random sample), and the realized
margin never drops below the certified lower bound. The sealed certificate is
digest-verifiable and drives the Lean bridge (which degrades gracefully with no toolchain).
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from omnibias.struct import DAG, ChainTrellis, shortest_path, viterbi
from omnibias.struct.decode import (
    certify_decoding,
    certify_decoding_dag,
    check_dag_decoding_certificate,
    check_decoding_certificate,
    seal_dag_decoding_certificate,
    seal_decoding_certificate,
)

TOL = 1e-9


def _margin_at(emis: np.ndarray, trans: np.ndarray, start: np.ndarray, winner: tuple[int, ...]) -> float:
    tr = ChainTrellis(emis, trans, start)
    scores = {p: tr.path_score(p) for p in tr.enumerate_paths()}
    w = scores[winner]
    runner = max(v for p, v in scores.items() if p != winner)
    return w - runner


def _corners(center: np.ndarray, radius: float) -> list[np.ndarray]:
    flat = center.reshape(-1)
    return [
        (flat + radius * np.array(signs)).reshape(center.shape)
        for signs in itertools.product((-1.0, 1.0), repeat=flat.size)
    ]


def test_certifies_well_separated_winner_and_matches_viterbi() -> None:
    emis = np.array([[3.0, 0.0], [0.0, 3.0], [3.0, 0.0]])
    trans = np.array([[1.0, 0.0], [0.0, 1.0]])
    cert = certify_decoding(emis, trans, eps=0.25)
    _, winner = viterbi(ChainTrellis(emis, trans, np.zeros(2)))
    assert cert.winner == winner
    assert cert.certified
    assert cert.min_margin > 0.0


def test_soundness_winner_is_argmax_over_full_box() -> None:
    rng = np.random.default_rng(0)
    emis = np.array([[3.0, 0.0], [0.0, 3.0], [3.0, 0.2]])
    trans = np.array([[0.5, -0.5], [-0.5, 0.5]])
    start = np.zeros(2)
    eps = 0.3
    cert = certify_decoding(emis, trans, eps, start=start)
    assert cert.certified
    samples = _corners(emis, eps) + [emis + rng.uniform(-eps, eps, emis.shape) for _ in range(300)]
    for x in samples:
        _, w = viterbi(ChainTrellis(x, trans, start))
        assert w == cert.winner  # decode is stable across the whole box
        assert _margin_at(x, trans, start, cert.winner) >= cert.min_margin - TOL


def test_margin_interval_encloses_center_margin() -> None:
    emis = np.array([[2.0, 0.0], [0.0, 2.0], [2.0, 0.0]])
    trans = np.array([[0.3, 0.1], [0.1, 0.3]])
    start = np.zeros(2)
    cert = certify_decoding(emis, trans, eps=0.2, start=start)
    center = _margin_at(emis, trans, start, cert.winner)
    assert cert.margin.lo - TOL <= center <= cert.margin.hi + TOL


def test_eps_zero_recovers_exact_center_margin() -> None:
    emis = np.array([[2.0, 0.0], [0.0, 2.0], [1.5, 0.0]])
    trans = np.array([[0.2, 0.0], [0.0, 0.2]])
    start = np.zeros(2)
    cert = certify_decoding(emis, trans, eps=0.0, start=start)
    exact = _margin_at(emis, trans, start, cert.winner)
    assert abs(cert.min_margin - exact) < 1e-9
    assert cert.margin.width < 1e-9  # a point box is a (near-)point enclosure


def test_margin_shrinks_with_eps_and_eventually_fails() -> None:
    emis = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    trans = np.array([[0.0, 0.0], [0.0, 0.0]])
    start = np.zeros(2)
    tight = certify_decoding(emis, trans, eps=0.1, start=start)
    loose = certify_decoding(emis, trans, eps=0.4, start=start)
    assert tight.min_margin >= loose.min_margin
    huge = certify_decoding(emis, trans, eps=5.0, start=start)
    assert not huge.certified  # a large ball can flip the decode -> not certified


def test_seal_is_digest_verifiable_and_lean_bridge_is_honest() -> None:
    emis = np.array([[3.0, 0.0], [0.0, 3.0], [3.0, 0.0]])
    trans = np.array([[1.0, 0.0], [0.0, 1.0]])
    cert = certify_decoding(emis, trans, eps=0.2)
    sealed = seal_decoding_certificate(cert)
    assert sealed["honesty"]["unproven_claim"] is False
    verdict = check_decoding_certificate(cert)
    assert verdict.sealed_ok  # tamper-evident digest matches
    assert verdict.obligation_generated  # margin.lo > 0 -> enclosed_quantity_pos
    # theorem_prover_verified is exactly the (real) lake verdict -- never forged.
    assert verdict.theorem_prover_verified == verdict.lean.verified
    if not verdict.lean.available:
        assert verdict.theorem_prover_verified is False


def test_rejects_bad_arguments() -> None:
    emis = np.array([[1.0, 0.0], [0.0, 1.0]])
    trans = np.array([[0.0, 0.0], [0.0, 0.0]])
    with pytest.raises(ValueError, match="eps must be non-negative"):
        certify_decoding(emis, trans, eps=-0.1)
    with pytest.raises(ValueError, match="n_states >= 2"):
        certify_decoding(np.array([[1.0], [1.0]]), np.array([[0.0]]), eps=0.1)


# ---------------------------------------------------------------------------
# DAG generalization (also covers the edge-weighted alignment lattice)
# ---------------------------------------------------------------------------

# A well-separated diamond: the unique shortest path 0->1->3 (cost 0.2) beats 0->2->3 (4.0)
# and 0->1->2->3 (4.1) by a wide margin, so a moderate edge ball keeps the decode stable.
_DIAMOND_EDGES = {(0, 1): 0.1, (0, 2): 2.0, (1, 2): 2.0, (1, 3): 0.1, (2, 3): 2.0}
_DIAMOND = DAG(4, _DIAMOND_EDGES, sink=3)


def _dag_worst_margin(dag: DAG, w: dict[tuple[int, int], float], eps: float) -> float:
    _, winner = shortest_path(dag)
    win_edges = set(zip(winner[:-1], winner[1:], strict=True))

    def cost(path: tuple[int, ...]) -> float:
        return sum(w[(path[i], path[i + 1])] for i in range(len(path) - 1))

    cp = cost(winner)
    best = np.inf
    for q in dag.enumerate_paths():
        if tuple(q) == tuple(winner):
            continue
        symdiff = len(set(zip(q[:-1], q[1:], strict=True)) ^ win_edges)
        best = min(best, (cost(q) - cp) - eps * symdiff)
    return float(best)


@pytest.mark.parametrize("eps", [0.0, 0.05, 0.2])
def test_dag_margin_matches_bruteforce(eps: float) -> None:
    cert = certify_decoding_dag(_DIAMOND_EDGES, _DIAMOND, eps)
    assert abs(cert.min_margin - _dag_worst_margin(_DIAMOND, _DIAMOND_EDGES, eps)) < 1e-9


def test_dag_soundness_winner_is_argmin_over_full_box() -> None:
    rng = np.random.default_rng(0)
    eps = 0.15
    cert = certify_decoding_dag(_DIAMOND_EDGES, _DIAMOND, eps)
    assert cert.certified
    edge_keys = list(_DIAMOND_EDGES)
    corners = itertools.product((-1.0, 1.0), repeat=len(edge_keys))
    samples = [
        {k: _DIAMOND_EDGES[k] + eps * s for k, s in zip(edge_keys, signs, strict=True)}
        for signs in corners
    ] + [
        {k: _DIAMOND_EDGES[k] + float(rng.uniform(-eps, eps)) for k in edge_keys}
        for _ in range(200)
    ]
    for w in samples:
        _, winner = shortest_path(DAG(4, w, sink=3))
        assert winner == cert.winner  # decode stable across the whole edge box


def test_dag_single_path_is_trivially_certified() -> None:
    chain = DAG(3, {(0, 1): 0.7, (1, 2): -0.3}, sink=2)
    cert = certify_decoding_dag({(0, 1): 0.7, (1, 2): -0.3}, chain, eps=1.0)
    assert cert.certified and cert.min_margin > 1e299  # no competitor -> unbeatable


def test_dag_large_ball_not_certified() -> None:
    cert = certify_decoding_dag(_DIAMOND_EDGES, _DIAMOND, eps=5.0)
    assert not cert.certified


def test_dag_seal_is_digest_verifiable_and_lean_bridge_is_honest() -> None:
    cert = certify_decoding_dag(_DIAMOND_EDGES, _DIAMOND, eps=0.1)
    sealed = seal_dag_decoding_certificate(cert)
    assert sealed["honesty"]["unproven_claim"] is False
    verdict = check_dag_decoding_certificate(cert)
    assert verdict.sealed_ok
    assert verdict.obligation_generated
    assert verdict.theorem_prover_verified == verdict.lean.verified
    if not verdict.lean.available:
        assert verdict.theorem_prover_verified is False
