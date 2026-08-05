# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Certified decoding: prove the hard-DP winner survives an input ``epsilon``-ball.

The Viterbi decode of a linear chain is an ``argmax`` over ``S**T`` paths. This module
certifies that the winning path ``p*`` (the decode at the nominal emissions ``x0``) stays
*the* argmax for **every** emission matrix in the box ``||x - x0||_inf <= eps`` -- the
structured analogue of :func:`omnibias.verify.certify_robustness` (winner-vs-runner-up
margin ``> 0`` over an :math:`L^\infty` ball).

The certified quantity is the **worst-case margin**
``M = min_{x in box} [ score(p*, x) - max_{p != p*} score(p, x) ]``.
``M`` is enclosed by one enumeration-free *reduced* min-plus DP (outward-rounded
:mod:`omnibias.core.verified` intervals). Working relative to the winner cancels the
dependency on any emission the winner and a competitor **share** (naively enclosing
``score(p*) - score(runner_up)`` with independent intervals double-counts those shared
entries and is hopelessly loose): the reduced weight of visiting ``(t, s)`` is
``emit[t, p*[t]] - emit[t, s]`` (exactly ``0`` when ``s == p*[t]``, so winner-aligned steps
contribute nothing), plus the exact transition difference. The best competitor is the
shortest *deviating* path in these reduced weights, found with a deviation-tracking min-plus
recurrence (a "must differ from ``p*`` in at least one step" constraint) -- no path
enumeration. Because transitions are exact and each emission perturbs independently, this
enclosure is **tight** (a single deviation node swings by exactly ``2 eps``).

``certified`` is ``margin.lo > 0``. The enclosure is sound, scoped to the given
``local_box`` -- never a global claim. Sealing + the Lean bridge live in
:func:`seal_decoding_certificate` / :func:`check_decoding_certificate`;
``theorem_prover_verified`` is earned only by a genuine ``lake build`` pass, never forged.

The two axes stay separate: this is the ``beta -> inf`` hard decode (the ``argmax``); the
``delta -> 0`` derivative tower is irrelevant here -- decoding stability is a sign fact
about the max-plus semiring, proved with rigorous intervals, not a derivative.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from omnibias.core.proof.certificate import (
    Cert,
    interval_certificate,
    verify_certificate_digest,
)
from omnibias.core.proof.lean_check import (
    LeanCheckResult,
    check_certificate,
    generate_obligation,
)
from omnibias.core.verified import Interval
from omnibias.struct._core.hard_dp import shortest_path, viterbi
from omnibias.struct._core.trellis import DAG, ChainTrellis

_INF = Interval.point(1e300)  # min-plus "unreachable / no deviation yet" sentinel


def _iv_min(items: Sequence[Interval]) -> Interval:
    """Interval enclosure of the ``min`` of interval-valued scores (``[min lo, min hi]``)."""
    lo = min(iv.lo for iv in items)
    hi = min(iv.hi for iv in items)
    return Interval(lo, hi)


@dataclass(frozen=True)
class DecodingCertificate:
    r"""Certified worst-case margin of the Viterbi winner over an emission ``eps``-ball.

    ``margin`` soundly (and tightly) encloses
    ``M = min_{x in box} [score(p*, x) - max_{p != p*} score(p, x)]``; the decode is certified
    stable iff ``margin.lo > 0`` (the winner is the unique argmax for every emission matrix
    within ``eps`` of the nominal input). Scope is the ``local_box``.
    """

    certified: bool
    winner: tuple[int, ...]
    eps: float
    margin: Interval
    n_steps: int
    n_states: int

    @property
    def min_margin(self) -> float:
        """Certified lower bound on the worst-case winner-vs-runner-up margin."""
        return float(self.margin.lo)


def certify_decoding(
    emissions: Sequence[Sequence[float]],
    transitions: Sequence[Sequence[float]],
    eps: float,
    *,
    start: Sequence[float] | None = None,
) -> DecodingCertificate:
    r"""Certify the linear-chain Viterbi decode over ``||x - emissions||_inf <= eps``.

    ``emissions`` is the ``(T, S)`` nominal input (the perturbed quantity); ``transitions``
    ``(S, S)`` and ``start`` ``(S,)`` are exact model parameters. The winner ``p*`` is the
    decode at ``emissions``; the returned :class:`DecodingCertificate` encloses the
    worst-case margin over the box (sound, not tight; ``local_box`` scope).
    """
    if eps < 0.0:
        raise ValueError(f"eps must be non-negative, got {eps}")
    emis = np.asarray(emissions, dtype=float)
    trans = np.asarray(transitions, dtype=float)
    n_steps, n_states = emis.shape
    if n_states < 2:
        raise ValueError("certify_decoding needs n_states >= 2 (no runner-up otherwise)")
    start_arr = np.zeros(n_states) if start is None else np.asarray(start, dtype=float)

    _, winner = viterbi(ChainTrellis(emis, trans, start_arr))

    # Reduced weights, measured *relative to the winner* so shared emissions cancel exactly.
    # re_emit[t][s] encloses emit[t, p*[t]] - emit[t, s] over the box (0 exactly on the winner);
    # transition / start differences are exact (those parameters are not perturbed).
    pad = Interval(-eps, eps)
    win_emit_iv = [Interval.point(float(emis[t, winner[t]])) + pad for t in range(n_steps)]

    def re_emit(t: int, s: int) -> Interval:
        if s == winner[t]:
            return Interval.point(0.0)
        return win_emit_iv[t] - (Interval.point(float(emis[t, s])) + pad)

    def re_trans(t: int, sp: int, s: int) -> Interval:
        # winner edge p*[t-1] -> p*[t] minus competitor edge sp -> s (exact).
        w = float(trans[winner[t - 1], winner[t]])
        return Interval.point(w - float(trans[sp, s]))

    def re_start(s: int) -> Interval:
        return Interval.point(float(start_arr[winner[0]]) - float(start_arr[s]))

    # dev[t][s] = min reduced margin of a prefix ending at (t, s) that has already
    # deviated from the winner in >= 1 step (the winner-aligned prefix stays exactly 0).
    dev: list[Interval] = [
        _INF if s == winner[0] else re_emit(0, s) + re_start(s) for s in range(n_states)
    ]
    for t in range(1, n_steps):
        # best_any[sp]: min reduced prefix to (t-1, sp) over deviated *or* on-winner. The
        # on-winner (exactly 0) prefix exists only at sp == p*[t-1].
        zero = Interval.point(0.0)
        best_any = [
            _iv_min([dev[sp], zero]) if sp == winner[t - 1] else dev[sp] for sp in range(n_states)
        ]
        nxt: list[Interval] = []
        for s in range(n_states):
            if s != winner[t]:
                # This step itself deviates, so the on-winner prefix is admissible too.
                cands = [best_any[sp] + re_trans(t, sp, s) for sp in range(n_states)]
            else:
                # Landing back on the winner: continuing the on-winner 0 prefix does NOT
                # deviate, so at sp == p*[t-1] only an already-deviated prefix (dev) counts.
                cands = [dev[sp] + re_trans(t, sp, s) for sp in range(n_states)]
            nxt.append(re_emit(t, s) + _iv_min(cands))
        dev = nxt

    margin = _iv_min(dev)  # worst-case winner-vs-any-other-path margin over the box
    return DecodingCertificate(
        certified=margin.lo > 0.0,
        winner=tuple(int(w) for w in winner),
        eps=float(eps),
        margin=margin,
        n_steps=int(n_steps),
        n_states=int(n_states),
    )


def seal_decoding_certificate(cert: DecodingCertificate, *, meta: dict[str, Any] | None = None) -> Cert:
    r"""Seal a :class:`DecodingCertificate` as a tamper-evident v1 interval certificate.

    The sealed payload is the worst-case margin interval; when ``margin.lo > 0`` the Lean
    bridge turns it into the finite ``enclosed_quantity_pos`` obligation. Honesty flags fix
    ``unproven_claim=False`` and record the ``local_box`` scope (the omnibias convention).
    """
    claim = (
        f"Viterbi decode {cert.winner} is the unique argmax over the emission "
        f"L_inf ball of radius {cert.eps!r}: worst-case margin in "
        f"[{cert.margin.lo!r}, {cert.margin.hi!r}]"
    )
    payload_meta: dict[str, Any] = {
        "winner": list(cert.winner),
        "eps": cert.eps,
        "n_steps": cert.n_steps,
        "n_states": cert.n_states,
        "scope": "local_box",
        "label": "interval (outward-rounded reduced min-plus margin; tight)",
    }
    if meta:
        payload_meta.update(meta)
    return interval_certificate(
        claim, cert.margin, honesty={"unproven_claim": False, "local_box": True}, meta=payload_meta
    )


@dataclass(frozen=True)
class DecodingProofVerdict:
    """A sealed decoding certificate plus its Lean-kernel adjudication."""

    certificate: Cert
    decoding: DecodingCertificate
    obligation: str | None
    lean: LeanCheckResult

    @property
    def theorem_prover_verified(self) -> bool:
        """``True`` only on a genuine Lean ``lake build`` pass (never forged)."""
        return bool(self.lean.verified)

    @property
    def sealed_ok(self) -> bool:
        """Whether the certificate's tamper-evident digest matches its body."""
        return bool(verify_certificate_digest(self.certificate))

    @property
    def obligation_generated(self) -> bool:
        """Whether a finite, Lean-checkable positivity obligation was produced."""
        return self.obligation is not None


def check_decoding_certificate(
    cert: DecodingCertificate,
    *,
    meta: dict[str, Any] | None = None,
    timeout: float = 600.0,
    start: Path | None = None,
) -> DecodingProofVerdict:
    r"""Seal a :class:`DecodingCertificate` and run the Lean-kernel bridge on it.

    ``theorem_prover_verified`` is set only on a real kernel pass; with no Lean toolchain the
    bridge degrades gracefully (``lean.available is False``), so this is safe in normal CI.
    """
    sealed = seal_decoding_certificate(cert, meta=meta)
    obligation = generate_obligation(sealed)
    lean = check_certificate(sealed, timeout=timeout, start=start)
    return DecodingProofVerdict(sealed, cert, obligation, lean)


# ---------------------------------------------------------------------------
# DAG / DTW generalization: certify the shortest-path decode over an edge eps-ball
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DAGDecodingCertificate:
    r"""Certified worst-case margin of a DAG shortest-path decode over an edge ``eps``-ball.

    Generalizes :class:`DecodingCertificate` from the linear chain to any topologically
    ordered :class:`~omnibias.struct.DAG` (so it also covers the alignment / DTW lattices,
    which are DAGs). ``margin`` soundly encloses
    ``M = min_{q != p*} [cost(q, w) - cost(p*, w) - eps * |edges(q) triangle edges(p*)|]``
    -- the worst-case cost gap of the winning path ``p*`` against every other path when each
    edge weight is perturbed within ``eps``. Certified stable iff ``margin.lo > 0``; scope is
    the ``local_box``.
    """

    certified: bool
    winner: tuple[int, ...]
    eps: float
    margin: Interval
    num_nodes: int
    num_edges: int

    @property
    def min_margin(self) -> float:
        """Certified lower bound on the worst-case winner-vs-runner-up cost margin."""
        return float(self.margin.lo)


def certify_decoding_dag(
    weights: Mapping[tuple[int, int], float] | Sequence[Sequence[float]],
    dag: DAG,
    eps: float,
) -> DAGDecodingCertificate:
    r"""Certify the DAG shortest-path decode over ``||w - weights||_inf <= eps`` on the edges.

    ``weights`` supplies the nominal edge costs (a ``{(u, v): w}`` map or a dense
    ``(n, n)`` matrix; only the ``dag`` edges are read); every edge is perturbed independently
    within ``eps``. Working in edge weights *reduced relative to the winner* -- ``w'_e =
    w_e - eps + 2 eps [e in p*]`` -- turns the worst-case margin into one deviation-tracking
    min-plus shortest path (``no`` path enumeration): shared edges cancel, so the enclosure is
    tight. Returns a :class:`DAGDecodingCertificate` (sound, ``local_box`` scope).
    """
    if eps < 0.0:
        raise ValueError(f"eps must be non-negative, got {eps}")
    wmat = dag.weight_matrix() if not isinstance(weights, Mapping) else None

    def w_of(u: int, v: int) -> float:
        if isinstance(weights, Mapping):
            return float(weights[(u, v)])
        return float(weights[u][v]) if wmat is None else float(wmat[u, v])

    _, winner = shortest_path(dag)
    win_edges = set(zip(winner[:-1], winner[1:], strict=True))
    eps_iv = Interval.point(float(eps))
    two_eps = Interval.point(2.0 * float(eps))

    def w_prime(u: int, v: int) -> Interval:
        base = Interval.point(w_of(u, v)) - eps_iv
        return base + two_eps if (u, v) in win_edges else base

    # on[v]: cost' of the (unique) prefix that still follows p*; dev[v]: best deviated prefix.
    on: list[Interval | None] = [None] * dag.num_nodes
    dev: list[Interval | None] = [None] * dag.num_nodes
    on[dag.source] = Interval.point(0.0)
    for v in range(dag.num_nodes):
        for u in dag.incoming(v):
            wp = w_prime(u, v)
            on_edge = (u, v) in win_edges
            on_u = on[u]
            if on_u is not None:
                cand = on_u + wp
                if on_edge:
                    cur_on = on[v]
                    on[v] = cand if cur_on is None else _iv_min([cur_on, cand])
                else:
                    cur_dev = dev[v]
                    dev[v] = cand if cur_dev is None else _iv_min([cur_dev, cand])
            dev_u = dev[u]
            if dev_u is not None:
                cand = dev_u + wp
                cur_dev2 = dev[v]
                dev[v] = cand if cur_dev2 is None else _iv_min([cur_dev2, cand])

    winner_cost = Interval.point(sum(w_of(u, v) for (u, v) in win_edges))
    offset = winner_cost + Interval.point(float(eps) * len(win_edges))
    dev_sink = dev[dag.sink]
    margin = _INF if dev_sink is None else dev_sink - offset  # no competitor -> unique decode
    return DAGDecodingCertificate(
        certified=margin.lo > 0.0,
        winner=tuple(int(v) for v in winner),
        eps=float(eps),
        margin=margin,
        num_nodes=int(dag.num_nodes),
        num_edges=int(len(dag.edges)),
    )


def seal_dag_decoding_certificate(
    cert: DAGDecodingCertificate, *, meta: dict[str, Any] | None = None
) -> Cert:
    r"""Seal a :class:`DAGDecodingCertificate` as a tamper-evident v1 interval certificate.

    Mirrors :func:`seal_decoding_certificate`: the sealed payload is the worst-case margin
    interval, which the Lean bridge turns into the finite ``enclosed_quantity_pos`` obligation
    when ``margin.lo > 0``. Honesty flags fix ``unproven_claim=False`` and record ``local_box``.
    """
    claim = (
        f"DAG shortest-path decode {cert.winner} is the unique argmin over the edge "
        f"L_inf ball of radius {cert.eps!r}: worst-case margin in "
        f"[{cert.margin.lo!r}, {cert.margin.hi!r}]"
    )
    payload_meta: dict[str, Any] = {
        "winner": list(cert.winner),
        "eps": cert.eps,
        "num_nodes": cert.num_nodes,
        "num_edges": cert.num_edges,
        "scope": "local_box",
        "label": "interval (outward-rounded reduced min-plus deviation margin; tight)",
    }
    if meta:
        payload_meta.update(meta)
    return interval_certificate(
        claim, cert.margin, honesty={"unproven_claim": False, "local_box": True}, meta=payload_meta
    )


@dataclass(frozen=True)
class DAGDecodingProofVerdict:
    """A sealed DAG-decoding certificate plus its Lean-kernel adjudication."""

    certificate: Cert
    decoding: DAGDecodingCertificate
    obligation: str | None
    lean: LeanCheckResult

    @property
    def theorem_prover_verified(self) -> bool:
        """``True`` only on a genuine Lean ``lake build`` pass (never forged)."""
        return bool(self.lean.verified)

    @property
    def sealed_ok(self) -> bool:
        """Whether the certificate's tamper-evident digest matches its body."""
        return bool(verify_certificate_digest(self.certificate))

    @property
    def obligation_generated(self) -> bool:
        """Whether a finite, Lean-checkable positivity obligation was produced."""
        return self.obligation is not None


def check_dag_decoding_certificate(
    cert: DAGDecodingCertificate,
    *,
    meta: dict[str, Any] | None = None,
    timeout: float = 600.0,
    start: Path | None = None,
) -> DAGDecodingProofVerdict:
    r"""Seal a :class:`DAGDecodingCertificate` and run the Lean-kernel bridge on it.

    ``theorem_prover_verified`` is set only on a real kernel pass; with no Lean toolchain the
    bridge degrades gracefully, so this is safe in normal CI.
    """
    sealed = seal_dag_decoding_certificate(cert, meta=meta)
    obligation = generate_obligation(sealed)
    lean = check_certificate(sealed, timeout=timeout, start=start)
    return DAGDecodingProofVerdict(sealed, cert, obligation, lean)


__all__ = [
    "DAGDecodingCertificate",
    "DAGDecodingProofVerdict",
    "DecodingCertificate",
    "DecodingProofVerdict",
    "certify_decoding",
    "certify_decoding_dag",
    "check_dag_decoding_certificate",
    "check_decoding_certificate",
    "seal_dag_decoding_certificate",
    "seal_decoding_certificate",
]
