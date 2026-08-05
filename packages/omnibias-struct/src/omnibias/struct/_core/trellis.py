# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Backend-agnostic (numpy / pure-Python) dynamic-programming problem containers.

Three lattices, one shared idea -- a set of *complete paths*, each with an additive
*score*, over which the DP takes a semiring reduction (``max`` for the hard optimum,
``lse_beta`` for the soft relaxation):

* :class:`ChainTrellis` -- a linear-chain (Viterbi / linear-chain CRF) trellis with
  per-step emission scores ``(T, S)`` and state-to-state transition scores ``(S, S)``.
  Every state sequence is a valid path, so there are exactly ``S ** T`` of them.
* :class:`DAG` -- a weighted directed acyclic graph for shortest / longest path. Nodes
  are ``0 .. num_nodes - 1`` in **topological order** (every edge ``u -> v`` has
  ``u < v``); paths run from :attr:`DAG.source` to :attr:`DAG.sink`.
* :class:`CTCLattice` -- the blank-augmented alignment lattice of Connectionist
  Temporal Classification: alignments of length ``T`` over ``num_classes`` symbols that
  collapse (merge repeats, drop blanks) to a fixed ``targets`` label sequence.

Each container is pure data (no backend tensors); the differentiable soft DP consumes
backend tensors alongside the container's static structure. :func:`count_paths` returns
the exact number of complete paths -- the ``N`` in the closed-form gap bound
``lse_beta <= max + log(N) / beta``.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int_]


@dataclass(frozen=True)
class ChainTrellis:
    r"""A linear-chain trellis: emissions ``(T, S)``, transitions ``(S, S)``, start ``(S,)``.

    The score of a state path ``(s_0, ..., s_{T-1})`` is

    .. math::
        \mathrm{start}[s_0] + \sum_{t=0}^{T-1} E[t, s_t]
            + \sum_{t=1}^{T-1} A[s_{t-1}, s_t].

    Every one of the ``S ** T`` state sequences is a valid path (the transitions are
    finite scores, not a hard adjacency mask).
    """

    emissions: FloatArray
    transitions: FloatArray
    start: FloatArray = field(default=None)  # type: ignore[arg-type]

    def __post_init__(self) -> None:
        emissions = np.asarray(self.emissions, dtype=float)
        transitions = np.asarray(self.transitions, dtype=float)
        if emissions.ndim != 2:
            raise ValueError(f"emissions must be (T, S), got shape {emissions.shape}")
        t, s = emissions.shape
        if t < 1 or s < 1:
            raise ValueError(f"emissions must have T >= 1 and S >= 1, got {emissions.shape}")
        if transitions.shape != (s, s):
            raise ValueError(
                f"transitions must be (S, S) = ({s}, {s}), got {transitions.shape}"
            )
        start = np.zeros(s) if self.start is None else np.asarray(self.start, dtype=float)
        if start.shape != (s,):
            raise ValueError(f"start must be (S,) = ({s},), got {start.shape}")
        object.__setattr__(self, "emissions", emissions)
        object.__setattr__(self, "transitions", transitions)
        object.__setattr__(self, "start", start)

    @property
    def n_steps(self) -> int:
        """Number of time steps ``T``."""
        return int(self.emissions.shape[0])

    @property
    def n_states(self) -> int:
        """Number of states ``S``."""
        return int(self.emissions.shape[1])

    def path_score(self, states: object) -> float:
        """Additive score of the state path ``states`` (length ``T``)."""
        seq = np.asarray(states, dtype=int).reshape(-1)
        if seq.shape[0] != self.n_steps:
            raise ValueError(f"path must have length T = {self.n_steps}, got {seq.shape[0]}")
        score = float(self.start[seq[0]]) + float(self.emissions[0, seq[0]])
        for t in range(1, self.n_steps):
            score += float(self.transitions[seq[t - 1], seq[t]])
            score += float(self.emissions[t, seq[t]])
        return score

    def enumerate_paths(self) -> Iterator[tuple[int, ...]]:
        """Yield every state path (all ``S ** T`` of them); brute-force only, tiny ``T, S``."""
        s, t = self.n_states, self.n_steps
        for idx in range(s**t):
            path, rem = [], idx
            for _ in range(t):
                path.append(rem % s)
                rem //= s
            yield tuple(path)

    def count_paths(self) -> int:
        """Exact number of complete paths, ``S ** T``."""
        return int(self.n_states**self.n_steps)


@dataclass(frozen=True)
class DAG:
    r"""A weighted DAG for shortest / longest path, nodes in topological order.

    ``edges`` maps ``(u, v) -> weight`` with ``u < v`` (topological order). Paths run
    from :attr:`source` (default ``0``) to :attr:`sink` (default ``num_nodes - 1``).
    """

    num_nodes: int
    edges: dict[tuple[int, int], float]
    source: int = 0
    sink: int = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.num_nodes < 1:
            raise ValueError(f"num_nodes must be >= 1, got {self.num_nodes}")
        edges = {(int(u), int(v)): float(w) for (u, v), w in self.edges.items()}
        for u, v in edges:
            if not 0 <= u < self.num_nodes or not 0 <= v < self.num_nodes:
                raise ValueError(f"edge ({u}, {v}) out of range [0, {self.num_nodes})")
            if u >= v:
                raise ValueError(
                    f"edge ({u}, {v}) violates topological order (require u < v); "
                    "relabel nodes so every edge points to a higher index"
                )
        sink = self.num_nodes - 1 if self.sink is None else int(self.sink)
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "sink", sink)

    def incoming(self, v: int) -> list[int]:
        """Sorted predecessors ``u`` with an edge ``u -> v``."""
        return sorted(u for (u, tgt) in self.edges if tgt == v)

    def weight_matrix(self) -> FloatArray:
        """Dense ``(num_nodes, num_nodes)`` weights, ``+inf`` where no edge exists."""
        w = np.full((self.num_nodes, self.num_nodes), np.inf)
        for (u, v), weight in self.edges.items():
            w[u, v] = weight
        return w

    def enumerate_paths(self) -> Iterator[tuple[int, ...]]:
        """Yield every ``source -> sink`` path as a tuple of node indices."""
        succ: dict[int, list[int]] = {}
        for u, v in self.edges:
            succ.setdefault(u, []).append(v)
        for vs in succ.values():
            vs.sort()

        def dfs(node: int, acc: tuple[int, ...]) -> Iterator[tuple[int, ...]]:
            if node == self.sink:
                yield acc
                return
            for nxt in succ.get(node, []):
                yield from dfs(nxt, acc + (nxt,))

        yield from dfs(self.source, (self.source,))

    def path_cost(self, nodes: object) -> float:
        """Total edge weight of a ``source -> sink`` path given as node indices."""
        seq = [int(x) for x in np.asarray(nodes, dtype=int).reshape(-1)]
        return float(sum(self.edges[(seq[i], seq[i + 1])] for i in range(len(seq) - 1)))

    def count_paths(self) -> int:
        """Exact number of ``source -> sink`` paths (topological-order DP)."""
        succ: dict[int, list[int]] = {}
        for u, v in self.edges:
            succ.setdefault(u, []).append(v)
        counts = [0] * self.num_nodes
        counts[self.sink] = 1
        for node in range(self.num_nodes - 1, -1, -1):
            if node == self.sink:
                continue
            counts[node] = sum(counts[v] for v in succ.get(node, []))
        return int(counts[self.source])


@dataclass(frozen=True)
class CTCLattice:
    r"""The blank-augmented CTC alignment lattice for a fixed label sequence.

    ``targets`` are the (non-blank) label ids to emit in order; ``blank`` is the blank
    symbol id; ``num_classes`` is the alphabet size (including blank). A length-``T``
    alignment is a class-id sequence that, after merging equal neighbours and dropping
    blanks, equals ``targets``. The extended path visits positions in the sequence
    ``l' = (blank, y_0, blank, y_1, ..., y_{L-1}, blank)`` of length ``2 L + 1``.
    """

    targets: IntArray
    num_classes: int
    blank: int = 0

    def __post_init__(self) -> None:
        targets = np.asarray(self.targets, dtype=int).reshape(-1)
        if self.num_classes < 1:
            raise ValueError(f"num_classes must be >= 1, got {self.num_classes}")
        if not 0 <= self.blank < self.num_classes:
            raise ValueError(f"blank {self.blank} out of range [0, {self.num_classes})")
        for y in targets:
            if not 0 <= y < self.num_classes:
                raise ValueError(f"target label {y} out of range [0, {self.num_classes})")
            if y == self.blank:
                raise ValueError("target labels must be non-blank")
        object.__setattr__(self, "targets", targets)

    @property
    def n_labels(self) -> int:
        """Number of target labels ``L``."""
        return int(self.targets.shape[0])

    def extended_labels(self) -> IntArray:
        """The blank-interleaved sequence ``l'`` of length ``2 L + 1``."""
        m = 2 * self.n_labels + 1
        ext = np.full(m, self.blank, dtype=int)
        for i, y in enumerate(self.targets):
            ext[2 * i + 1] = int(y)
        return ext

    def incoming(self, s: int) -> list[int]:
        r"""Extended-lattice predecessors of position ``s`` (self, previous, skip)."""
        ext = self.extended_labels()
        preds = [s]
        if s - 1 >= 0:
            preds.append(s - 1)
        if s - 2 >= 0 and ext[s] != self.blank and ext[s] != ext[s - 2]:
            preds.append(s - 2)
        return sorted(preds)

    def collapse(self, seq: object) -> tuple[int, ...]:
        """Merge equal neighbours then drop blanks -- the CTC label of an alignment."""
        out: list[int] = []
        prev = None
        for c in (int(x) for x in np.asarray(seq, dtype=int).reshape(-1)):
            if c != prev:
                out.append(c)
            prev = c
        return tuple(c for c in out if c != self.blank)

    def count_alignments(self, n_steps: int) -> int:
        """Exact number of length-``n_steps`` alignments that collapse to ``targets``."""
        if n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {n_steps}")
        m = 2 * self.n_labels + 1
        if n_steps < self.n_labels:  # too short to emit every label
            return 0
        counts = [0] * m
        counts[0] = 1
        if m > 1:
            counts[1] = 1
        for _ in range(1, n_steps):
            nxt = [0] * m
            for s in range(m):
                nxt[s] = sum(counts[p] for p in self.incoming(s))
            counts = nxt
        total = counts[m - 1]
        if m >= 2:
            total += counts[m - 2]
        return int(total)


def count_paths(problem: ChainTrellis | DAG | CTCLattice, n_steps: int | None = None) -> int:
    r"""Exact number of complete paths / alignments -- the ``N`` in ``log(N) / beta``.

    For a :class:`CTCLattice` the alignment length ``n_steps`` (``= T``) is required.
    """
    if isinstance(problem, CTCLattice):
        if n_steps is None:
            raise ValueError("count_paths on a CTCLattice requires n_steps (= T)")
        return problem.count_alignments(n_steps)
    return problem.count_paths()


__all__ = ["CTCLattice", "ChainTrellis", "DAG", "count_paths"]
