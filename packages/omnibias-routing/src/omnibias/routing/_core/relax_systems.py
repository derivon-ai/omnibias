# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Poly-size directed-TSP LP relaxations as temperature-collapse penalty systems (numpy).

Each builder returns a :class:`RelaxSystem` -- the standard-form data
``A_eq x = b_eq``, ``A_ineq x <= b_ineq`` (box included as rows for the penalty
layer) and the variable box ``x_lower <= x <= x_upper`` (for the Neumaier-Shcherbina
LP certificate) -- shared by the differentiable backend layers *and* the
certificate. Three strengths, all with ``x_ij`` in ``[0, 1]`` the directed arc-use:

* ``assignment`` -- degree-1 in/out only (weakest; may contain subtours). ``E`` vars.
* ``flow`` -- single-commodity flow (Gavish-Graves): degree + one depot-rooted flow
  that forbids subtours. ``2E`` vars. Subtour-free, medium strength.
* ``held_karp`` -- multicommodity flow (one commodity per non-depot city): the
  **Held-Karp** (subtour-LP) bound, tightest. ``E * n`` vars -- ``O(n^3)``, so it is
  a small-``n`` / single-instance option (the dense path; a matrix-free operator is
  the staged scalability follow-up, see the cookbook).

``E = n (n - 1)`` directed arcs; depot = city 0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

VALID_KINDS = ("assignment", "flow", "held_karp")


def arc_index(n: int) -> tuple[list[tuple[int, int]], dict[tuple[int, int], int]]:
    """Directed arcs ``(i, j), i != j`` and their index map."""
    arcs = [(i, j) for i in range(n) for j in range(n) if i != j]
    return arcs, {a: e for e, a in enumerate(arcs)}


@dataclass(frozen=True)
class RelaxSystem:
    r"""Standard-form LP data for a TSP relaxation (shared by layer + certificate)."""

    kind: str
    n: int
    n_vars: int
    n_arcs: int
    arcs: tuple[tuple[int, int], ...]
    A_eq: np.ndarray
    b_eq: np.ndarray
    A_ineq: np.ndarray
    b_ineq: np.ndarray
    x_lower: np.ndarray
    x_upper: np.ndarray

    def cost_vector(self, cost: np.ndarray) -> np.ndarray:
        """Objective ``c`` over all vars: arc cost on the ``x`` block, 0 on flow."""
        c = np.asarray(cost, dtype=float)
        vec = np.zeros(self.n_vars)
        for e, (i, j) in enumerate(self.arcs):
            vec[e] = c[i, j]
        return vec

    def arc_matrix(self, x: np.ndarray) -> np.ndarray:
        """Reshape the first ``E`` (arc-use) vars into an ``(n, n)`` matrix."""
        xv = np.asarray(x, dtype=float)
        mat = np.zeros((self.n, self.n))
        for e, (i, j) in enumerate(self.arcs):
            mat[i, j] = xv[e]
        return mat


def _degree_rows(n: int, aid: dict[tuple[int, int], int], n_vars: int) -> tuple[list[np.ndarray], list[float]]:
    rows: list[np.ndarray] = []
    rhs: list[float] = []
    for i in range(n):  # out-degree 1
        row = np.zeros(n_vars)
        for j in range(n):
            if i != j:
                row[aid[(i, j)]] = 1.0
        rows.append(row)
        rhs.append(1.0)
    for j in range(n):  # in-degree 1
        row = np.zeros(n_vars)
        for i in range(n):
            if i != j:
                row[aid[(i, j)]] = 1.0
        rows.append(row)
        rhs.append(1.0)
    return rows, rhs


def _box_rows(lo: np.ndarray, hi: np.ndarray) -> tuple[list[np.ndarray], list[float]]:
    """``x <= hi`` and ``-x <= -lo`` as inequality rows (for the penalty layer)."""
    nv = lo.shape[0]
    rows: list[np.ndarray] = []
    rhs: list[float] = []
    for k in range(nv):
        r = np.zeros(nv)
        r[k] = 1.0
        rows.append(r)
        rhs.append(float(hi[k]))
    for k in range(nv):
        r = np.zeros(nv)
        r[k] = -1.0
        rows.append(r)
        rhs.append(float(-lo[k]))
    return rows, rhs


def assignment_system(n: int) -> RelaxSystem:
    """Degree-constrained (assignment) relaxation: ``E`` vars, ``x in [0, 1]``."""
    arcs, aid = arc_index(n)
    e = len(arcs)
    a_eq, b_eq = _degree_rows(n, aid, e)
    lo = np.zeros(e)
    hi = np.ones(e)
    a_in, b_in = _box_rows(lo, hi)
    return RelaxSystem(
        kind="assignment", n=n, n_vars=e, n_arcs=e, arcs=tuple(arcs),
        A_eq=np.asarray(a_eq), b_eq=np.asarray(b_eq),
        A_ineq=np.asarray(a_in), b_ineq=np.asarray(b_in),
        x_lower=lo, x_upper=hi,
    )


def flow_system(n: int) -> RelaxSystem:
    """Single-commodity-flow relaxation (subtour-free): ``2E`` vars ``[x, f]``."""
    arcs, aid = arc_index(n)
    e = len(arcs)
    nv = 2 * e
    cap = float(n - 1)
    a_eq, b_eq = _degree_rows(n, aid, nv)
    for i in range(1, n):  # non-depot flow conservation: inflow - outflow = 1
        row = np.zeros(nv)
        for j in range(n):
            if i != j:
                row[e + aid[(j, i)]] += 1.0
                row[e + aid[(i, j)]] -= 1.0
        a_eq.append(row)
        b_eq.append(1.0)
    # coupling f_e / (n-1) - x_e <= 0  (scaled to keep the penalty well-conditioned).
    a_in: list[np.ndarray] = []
    b_in: list[float] = []
    for (i, j) in arcs:
        row = np.zeros(nv)
        row[e + aid[(i, j)]] = 1.0 / cap
        row[aid[(i, j)]] = -1.0
        a_in.append(row)
        b_in.append(0.0)
    lo = np.zeros(nv)
    hi = np.concatenate([np.ones(e), cap * np.ones(e)])
    box_rows, box_rhs = _box_rows(lo, hi)
    a_in.extend(box_rows)
    b_in.extend(box_rhs)
    return RelaxSystem(
        kind="flow", n=n, n_vars=nv, n_arcs=e, arcs=tuple(arcs),
        A_eq=np.asarray(a_eq), b_eq=np.asarray(b_eq),
        A_ineq=np.asarray(a_in), b_ineq=np.asarray(b_in),
        x_lower=lo, x_upper=hi,
    )


def held_karp_system(n: int) -> RelaxSystem:
    """Multicommodity-flow (Held-Karp / subtour-LP) relaxation: ``E * n`` vars."""
    arcs, aid = arc_index(n)
    e = len(arcs)
    k_comm = n - 1  # commodities: one unit shipped depot 0 -> destination c (c = 1..n-1)
    nv = e + k_comm * e

    def fidx(c: int, arc: tuple[int, int]) -> int:
        return e + (c - 1) * e + aid[arc]

    a_eq, b_eq = _degree_rows(n, aid, nv)
    for c in range(1, n):  # per-commodity conservation at every node
        for v in range(n):
            row = np.zeros(nv)
            for u in range(n):
                if u != v:
                    row[fidx(c, (u, v))] += 1.0  # inflow
                    row[fidx(c, (v, u))] -= 1.0  # outflow
            rhs = 1.0 if v == c else (-1.0 if v == 0 else 0.0)
            a_eq.append(row)
            b_eq.append(rhs)
    a_in: list[np.ndarray] = []
    b_in: list[float] = []
    for c in range(1, n):  # coupling f^c_ij <= x_ij
        for arc in arcs:
            row = np.zeros(nv)
            row[fidx(c, arc)] = 1.0
            row[aid[arc]] = -1.0
            a_in.append(row)
            b_in.append(0.0)
    lo = np.zeros(nv)
    hi = np.ones(nv)  # x in [0,1], each commodity flow in [0,1]
    box_rows, box_rhs = _box_rows(lo, hi)
    a_in.extend(box_rows)
    b_in.extend(box_rhs)
    return RelaxSystem(
        kind="held_karp", n=n, n_vars=nv, n_arcs=e, arcs=tuple(arcs),
        A_eq=np.asarray(a_eq), b_eq=np.asarray(b_eq),
        A_ineq=np.asarray(a_in), b_ineq=np.asarray(b_in),
        x_lower=lo, x_upper=hi,
    )


def build_system(n: int, kind: str = "flow") -> RelaxSystem:
    """Dispatch to the requested relaxation strength."""
    if kind == "assignment":
        return assignment_system(n)
    if kind == "flow":
        return flow_system(n)
    if kind == "held_karp":
        return held_karp_system(n)
    raise ValueError(f"unknown relaxation kind {kind!r}; choose from {VALID_KINDS}")


__all__ = [
    "RelaxSystem",
    "VALID_KINDS",
    "arc_index",
    "assignment_system",
    "build_system",
    "flow_system",
    "held_karp_system",
]
