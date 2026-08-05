# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Quadratic Assignment Problem (QAP) as a QUBO-form ``DiscreteProblem``.

QAP assigns ``dim`` facilities to ``dim`` locations to minimise
``sum_{i,k} F[i,k] D[perm(i),perm(k)]`` (flow ``F`` times distance ``D``). It is
**NP-hard** (it contains the travelling-salesman and graph-partitioning problems). We
encode a permutation with ``dim^2`` binary variables ``x[i,j] = [facility i -> location
j]`` and minimise, over ``x in {0, 1}^{dim^2}``,

.. math::
    E(x) = \sum_{i,k,j,l} F_{ik} D_{jl}\, x_{ij} x_{kl}
         + \lambda\Bigl(\sum_i \bigl(\textstyle\sum_j x_{ij} - 1\bigr)^2
                       + \sum_j \bigl(\textstyle\sum_i x_{ij} - 1\bigr)^2\Bigr),

a QUBO with interaction matrix ``Q = F (x) D`` (Kronecker product, index ``p = i*dim +
j``) plus the row/column one-hot permutation penalty ``lambda``. A valid permutation has
zero penalty, so its energy equals the QAP objective; a large-enough ``lambda`` keeps the
minimiser a permutation.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from omnibias.qubo import QUBOProblem

if TYPE_CHECKING:  # pragma: no cover - typing only
    from omnibias.sos import Polynomial

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.intp]


def permutation_penalty_arrays(dim: int) -> tuple[FloatArray, FloatArray, float]:
    r"""The unit (``lambda = 1``) row/column one-hot permutation penalty ``(Q, c, const)``.

    Expanding ``sum_i (sum_j x_ij - 1)^2 + sum_j (sum_i x_ij - 1)^2`` gives the quadratic
    ``Q = kron(I, J) + kron(J, I)`` (``J`` all-ones ``dim x dim``), linear ``c = -4``, and
    constant ``2*dim``; a caller scales all three by ``lambda``.
    """
    eye = np.eye(dim)
    ones = np.ones((dim, dim))
    q_pen: FloatArray = np.kron(eye, ones) + np.kron(ones, eye)
    c_pen: FloatArray = -4.0 * np.ones(dim * dim)
    return q_pen, c_pen, 2.0 * float(dim)


def qap_qubo_arrays(
    flow: FloatArray, distance: FloatArray, penalty: float
) -> tuple[FloatArray, FloatArray, float]:
    r"""The QAP QUBO ``(Q, c, const)``: ``kron(F, D)`` interaction + the permutation penalty."""
    interaction: FloatArray = np.asarray(np.kron(flow, distance), dtype=np.float64)
    q_pen, c_pen, const_pen = permutation_penalty_arrays(int(flow.shape[0]))
    q: FloatArray = interaction + penalty * q_pen
    c: FloatArray = penalty * c_pen
    return q, c, penalty * const_pen


@dataclass(frozen=True)
class QAPProblem:
    r"""A QAP instance encoded as a QUBO over ``dim^2`` permutation bits.

    Attributes
    ----------
    flow:
        ``(dim, dim)`` flow matrix ``F``.
    distance:
        ``(dim, dim)`` distance matrix ``D``.
    penalty:
        The one-hot permutation penalty ``lambda`` (large enough that the minimiser is a
        permutation; :func:`qap` picks a safe default).
    name:
        Optional label.
    """

    flow: FloatArray
    distance: FloatArray
    penalty: float
    name: str | None = None

    def __post_init__(self) -> None:
        flow = np.asarray(self.flow, dtype=float)
        distance = np.asarray(self.distance, dtype=float)
        for mat, what in ((flow, "flow"), (distance, "distance")):
            if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
                raise ValueError(f"{what} must be a square (dim, dim) matrix, got {mat.shape}")
        if flow.shape != distance.shape:
            raise ValueError(f"flow {flow.shape} and distance {distance.shape} must match")
        if flow.shape[0] < 1:
            raise ValueError("QAP needs at least one facility")
        object.__setattr__(self, "flow", flow)
        object.__setattr__(self, "distance", distance)
        object.__setattr__(self, "penalty", float(self.penalty))

    @property
    def dim(self) -> int:
        """The number of facilities / locations."""
        return int(self.flow.shape[0])

    @property
    def n(self) -> int:
        """The number of binary variables (``dim^2``)."""
        return self.dim * self.dim

    def _arrays(self) -> tuple[FloatArray, FloatArray, float]:
        cache = self.__dict__.get("_qubo_arrays")
        if cache is None:
            cache = qap_qubo_arrays(self.flow, self.distance, self.penalty)
            object.__setattr__(self, "_qubo_arrays", cache)
        arrays: tuple[FloatArray, FloatArray, float] = cache
        return arrays

    def objective(self, x: object) -> float | FloatArray:
        r"""The pure QAP objective ``x^T (F (x) D) x`` (no penalty) at a point / batch."""
        xv = np.asarray(x, dtype=float)
        interaction = np.kron(self.flow, self.distance)
        quad = np.sum((xv @ interaction) * xv, axis=-1)
        return float(quad) if xv.ndim == 1 else quad

    def energy(self, x: object) -> float | FloatArray:
        r"""The QUBO energy (QAP objective + permutation penalty) at a point / batch."""
        q, c, const = self._arrays()
        xv = np.asarray(x, dtype=float)
        quad = np.sum((xv @ q) * xv, axis=-1)
        lin = xv @ c
        out = quad + lin + const
        return float(out) if xv.ndim == 1 else out

    def flip_deltas(self, x: object) -> FloatArray:
        r"""Closed-form single-bit flip deltas (delegated to the QUBO fast path)."""
        deltas: FloatArray = self.to_qubo().flip_deltas(x)
        return deltas

    def to_qubo(self) -> QUBOProblem:
        r"""The equivalent :class:`omnibias.qubo.QUBOProblem` (Kronecker Q + penalty)."""
        q, c, const = self._arrays()
        return QUBOProblem(Q=q, c=c, const=const, name=self.name or "qap")

    def to_polynomial(self) -> Polynomial:
        r"""The energy as an :class:`omnibias.sos.Polynomial` (via the QUBO)."""
        return self.to_qubo().to_polynomial()


def qap(
    flow: object, distance: object, *, penalty: float | None = None, name: str | None = None
) -> QAPProblem:
    r"""Build a :class:`QAPProblem` from a flow and distance matrix.

    ``penalty`` defaults to the safe ``sum|F|*max|D| + max|F|*sum|D| + 1`` -- an upper
    bound on the objective change a single one-hot violation can cause, so the minimiser
    is a permutation (verified feasible across a seed sweep in the separate
    ``omnibias_experiments`` project); pass a
    larger value to be extra safe or a smaller calibrated one to tighten the gap.
    """
    flow_arr = np.asarray(flow, dtype=float)
    distance_arr = np.asarray(distance, dtype=float)
    if penalty is None:
        penalty = (
            float(
                np.abs(flow_arr).sum() * np.abs(distance_arr).max()
                + np.abs(flow_arr).max() * np.abs(distance_arr).sum()
            )
            + 1.0
        )
    return QAPProblem(flow_arr, distance_arr, penalty, name)


def placement_qap(
    connectivity: object,
    grid: tuple[int, int],
    *,
    penalty: float | None = None,
    name: str | None = "placement",
) -> QAPProblem:
    r"""Build a chip block-placement QAP: place ``N`` modules on an ``rows x cols`` grid.

    This is the canonical VLSI floorplanning model as a Koopmans-Beckmann QAP.
    ``connectivity`` is the ``(N, N)`` inter-module netlist connectivity (the flow ``F``);
    the distance ``D`` is the **integer Manhattan distance** between grid slots (slot ``s``
    at row ``s // cols``, column ``s % cols``), with ``N = rows * cols`` slots -- one per
    module. Minimising the QAP objective minimises the connectivity-weighted total
    wirelength. Integer ``connectivity`` keeps the whole instance integer, so the
    Gilmore-Lawler certificate (:func:`omnibias.nphard.gilmore_lawler_bound`) is exact.

    Placement is **NP-hard**: solve it with the differentiable ``relax`` -> ``decode``
    *heuristic* and certify the result with a sound but generally **non-tight** Gilmore-Lawler
    optimality-gap (``certify_gap(kind="glb")``) -- never an exact-optimum (``P = NP``) claim.
    """
    rows, cols = int(grid[0]), int(grid[1])
    n_slots = rows * cols
    conn = np.asarray(connectivity, dtype=float)
    if conn.ndim != 2 or conn.shape[0] != conn.shape[1]:
        raise ValueError(f"connectivity must be a square (N, N) matrix, got {conn.shape}")
    if conn.shape[0] != n_slots:
        raise ValueError(
            f"connectivity has {conn.shape[0]} modules but the {rows}x{cols} grid has "
            f"{n_slots} slots; they must match"
        )
    coords = np.array([(s // cols, s % cols) for s in range(n_slots)])
    distance = np.abs(coords[:, None, :] - coords[None, :, :]).sum(axis=-1).astype(float)
    return qap(conn, distance, penalty=penalty, name=name)


# --------------------------------------------------------------------------------------
# structured decoder + named classical baseline + exact exponential oracle
# --------------------------------------------------------------------------------------


def perm_to_x(perm: Sequence[int] | NDArray[np.intp], dim: int) -> FloatArray:
    r"""The ``dim^2`` binary indicator of a permutation ``perm`` (``facility i -> perm[i]``)."""
    x = np.zeros((dim, dim))
    for i, j in enumerate(perm):
        x[i, int(j)] = 1.0
    return x.reshape(-1)


def qap_round(relaxed: object, dim: int) -> FloatArray:
    r"""Round a relaxed heatmap to a permutation by Hungarian assignment **only** (no search).

    Unlike :func:`qap_decode` (which adds 2-opt local search), this is the raw *decision*
    the relaxation heatmap makes -- the right target for a decision-focused training loop,
    where local search would otherwise mask the heatmap's quality.
    """
    from scipy.optimize import linear_sum_assignment

    heat = np.asarray(relaxed, dtype=float).reshape(dim, dim)
    _, col = linear_sum_assignment(-heat)
    return perm_to_x([int(j) for j in col], dim)


def _two_opt(perm: list[int], problem: QAPProblem) -> tuple[list[int], float]:
    """Best-improvement 2-swap local search on a permutation (small ``dim``)."""
    dim = problem.dim
    best = list(perm)
    best_e = float(problem.energy(perm_to_x(best, dim)))
    improved = True
    while improved:
        improved = False
        for a in range(dim):
            for b in range(a + 1, dim):
                cand = list(best)
                cand[a], cand[b] = cand[b], cand[a]
                e = float(problem.energy(perm_to_x(cand, dim)))
                if e < best_e - 1e-12:
                    best, best_e, improved = cand, e, True
    return best, best_e


def qap_decode(
    problem: QAPProblem, *, relaxed: object | None = None
) -> tuple[tuple[int, ...], float]:
    r"""Decode a relaxed heatmap to a permutation (Hungarian) then 2-opt local search.

    ``relaxed`` is the ``(dim, dim)`` (or flat ``dim^2``) soft assignment from the
    differentiable relaxation; ``None`` starts from a uniform heatmap. Returns the binary
    indicator ``x in {0, 1}^{dim^2}`` (as a tuple) and its energy -- a heuristic *upper*
    bound.
    """
    from scipy.optimize import linear_sum_assignment

    dim = problem.dim
    if relaxed is None:
        heat = np.ones((dim, dim))
    else:
        heat = np.asarray(relaxed, dtype=float).reshape(dim, dim)
    _, col = linear_sum_assignment(-heat)  # maximise soft-assignment mass
    perm, energy = _two_opt([int(j) for j in col], problem)
    x = perm_to_x(perm, dim)
    return tuple(int(v) for v in x), energy


def qap_classical(problem: QAPProblem) -> tuple[tuple[int, ...], float]:
    r"""Named heuristic baseline: ``scipy.optimize.quadratic_assignment`` (FAQ + 2-opt).

    Runs both the Fast Approximate QAP (``"faq"``) and the ``"2opt"`` methods and keeps
    the better permutation. This is a strong classical *heuristic*, **not** an exact
    solver -- QAP is NP-hard, so no poly-time method is guaranteed optimal.
    """
    from scipy.optimize import quadratic_assignment

    dim = problem.dim
    best_perm: list[int] | None = None
    best_e = np.inf
    for method in ("faq", "2opt"):
        res = quadratic_assignment(problem.flow, problem.distance, method=method)
        perm = [int(j) for j in res.col_ind]
        e = float(problem.energy(perm_to_x(perm, dim)))
        if e < best_e:
            best_perm, best_e = perm, e
    assert best_perm is not None
    x = perm_to_x(best_perm, dim)
    return tuple(int(v) for v in x), best_e


def qap_brute_force(
    problem: QAPProblem, *, max_dim: int = 8
) -> tuple[tuple[int, ...], float]:
    r"""Exact optimum by enumerating all ``dim!`` permutations (**exponential**; ``dim`` tiny).

    Guarded to ``dim <= max_dim`` (``8! = 40320``). This is the exponential oracle used
    only to self-check the certified sandwich on small instances.
    """
    dim = problem.dim
    if dim > max_dim:
        raise ValueError(f"brute force is exponential (dim!); refusing dim={dim} > {max_dim}")
    best_perm: tuple[int, ...] | None = None
    best_e = np.inf
    for perm in itertools.permutations(range(dim)):
        e = float(problem.energy(perm_to_x(perm, dim)))
        if e < best_e:
            best_perm, best_e = perm, e
    assert best_perm is not None
    x = perm_to_x(best_perm, dim)
    return tuple(int(v) for v in x), best_e


__all__ = [
    "QAPProblem",
    "perm_to_x",
    "permutation_penalty_arrays",
    "placement_qap",
    "qap",
    "qap_brute_force",
    "qap_classical",
    "qap_decode",
    "qap_qubo_arrays",
    "qap_round",
]
