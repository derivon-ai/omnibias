# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Arrangement graph from sampled cells (theory 02-02).

Sampling is a subgraph / lower bound, never a complete face lattice.
``beta -> inf`` is temperature collapse, not founding ``delta -> 0``.
The gap is sound, not P vs NP, not theorem-prover.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from omnibias.partition.arrangement import (
    Arrangement,
    realized_cells,
    sign_vector,
    tope_graph,
)
from omnibias.struct._core.gap import logsumexp_gap_bound

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ArrangementGraph:
    cells: tuple[tuple[int, ...], ...]
    edges: tuple[tuple[int, int, int], ...]  # (u, v, crossed hyperplane)
    representatives: FloatArray

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "representatives", np.asarray(self.representatives, dtype=np.float64)
        )


def build_arrangement_graph(
    arr: Arrangement, samples: FloatArray
) -> ArrangementGraph:
    """Discovery by sampling. Returns what was found; never claims completeness."""
    pts = np.asarray(samples, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    cells = realized_cells(arr, pts)
    signs = sign_vector(arr, pts)
    reps: list[list[float]] = []
    for cell in cells:
        mask = np.all(signs == np.asarray(cell, dtype=np.float64), axis=1)
        chosen = pts[mask]
        reps.append(chosen.mean(axis=0).tolist() if len(chosen) else [0.0] * arr.dim)
    raw_edges = tope_graph(cells)
    edges: list[tuple[int, int, int]] = []
    for u, v in raw_edges:
        diffs = [i for i, (a, b) in enumerate(zip(cells[u], cells[v], strict=True)) if a != b]
        crossed = diffs[0] if diffs else 0
        edges.append((u, v, crossed))
    return ArrangementGraph(cells, tuple(edges), np.asarray(reps, dtype=np.float64))


def node_features(arr: Arrangement, graph: ArrangementGraph, *, beta: float) -> FloatArray:
    """Soft mass and margin at each representative. Temperature collapse in ``beta``."""
    from omnibias.partition.arrangement import margin, soft_membership

    rows = []
    for cell, rep in zip(graph.cells, graph.representatives, strict=True):
        mass = float(soft_membership(arr, rep, cell, beta=beta).reshape(-1)[0])
        mrg = float(np.asarray(margin(arr, rep)).reshape(-1)[0])
        rows.append([mass, mrg, *rep.tolist()])
    return np.asarray(rows, dtype=np.float64)


def certify_facenet_gap(
    logits: Sequence[float], *, beta: float, n_terms: int | None = None
) -> dict[str, float | bool]:
    """Reuse ``logsumexp_gap_bound``. Does not fork the constant."""
    scores = [float(v) for v in logits]
    n = int(n_terms) if n_terms is not None else len(scores)
    gap = logsumexp_gap_bound(n, float(beta))
    hard = max(scores)
    # lse_beta
    m = hard
    lse = m + math.log(sum(math.exp(beta * (s - m)) for s in scores)) / float(beta)
    return {
        "soft": lse,
        "hard": hard,
        "gap": gap,
        "sound": (hard <= lse + 1e-12) and (lse <= hard + gap + 1e-12),
        "temperature_collapse": True,
        "p_vs_np": False,
        "theorem_prover_verified": False,
    }


__all__ = [
    "ArrangementGraph",
    "build_arrangement_graph",
    "certify_facenet_gap",
    "node_features",
]
