# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Lovasz extension and **exact** submodular *minimization* (a P-class problem).

This is the honest mirror image of the rest of the package. Constrained submodular
**maximization** is NP-hard, so everything else here delivers a *certified approximation*.
Unconstrained submodular **minimization** ``min_{S subseteq [n]} f(S)`` is, by contrast,
solvable **exactly in polynomial time** (Grotschel-Lovasz-Schrijver; Iwata-Fleischer-Fujishige;
Schrijver) -- this is a genuine P-class result and asserting it is *not* a ``P = NP`` claim,
because minimization and maximization are different problems.

* :func:`lovasz_extension` -- the exact convex closure of ``f``. For ``p in [0, 1]^n``, sort
  the coordinates descending and telescope the marginals along the induced chain:
  ``f_L(p) = f(empty) + sum_i p_{sigma(i)} [f(S_i) - f(S_{i-1})]``. It agrees with ``f`` on
  every ``0/1`` vertex and is **convex iff ``f`` is submodular**, so minimizing ``f`` over
  subsets equals minimizing the convex ``f_L`` over the cube.
* :func:`min_norm_point` -- Wolfe's minimum-norm-point algorithm over the base polytope
  ``B(f)`` (Edmonds' greedy supplies the exact linear oracle). Fujishige's theorem: the
  min-norm point ``x*`` of ``B(f)`` yields ``min_S f(S) = sum_i min(x*_i, 0)`` with minimizer
  ``{i : x*_i < 0}``.
* :func:`submodular_minimize` -- the user-facing exact minimizer: runs :func:`min_norm_point`
  and recovers the optimal set from the chain of ``x*``-threshold sets (each evaluated exactly),
  returning a :class:`MinimizerResult`.

Pure numpy; no backend. The recovered minimizer is checked against the exact enumeration on
small ``n`` in the tests.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from omnibias.submodular.functions import SubmodularFunction

FloatArray = NDArray[np.float64]

_TOL = 1e-12


@dataclass(frozen=True)
class MinimizerResult:
    r"""The exact minimizer of a submodular ``f`` (unconstrained, P-class).

    Attributes
    ----------
    selection:
        The ``0/1`` indicator of the minimizing set ``argmin_S f(S)``.
    value:
        ``f(selection)`` -- the exact minimum (evaluated on the set, not the relaxation).
    point:
        The base-polytope min-norm point ``x*`` the minimizer was recovered from (or ``None``).
    """

    selection: tuple[int, ...]
    value: float
    point: FloatArray | None = None

    @property
    def n(self) -> int:
        return len(self.selection)

    @property
    def support(self) -> tuple[int, ...]:
        """The indices of the chosen elements (``{i : selection_i = 1}``)."""
        return tuple(i for i, v in enumerate(self.selection) if v)


def lovasz_extension(function: SubmodularFunction, p: object) -> float:
    r"""The Lovasz extension ``f_L(p)`` -- the exact convex closure evaluated at ``p in [0,1]^n``.

    Sorts the coordinates of ``p`` in descending order ``sigma`` and telescopes
    ``f_L(p) = f(empty) + sum_i p_{sigma(i)} [f(S_i) - f(S_{i-1})]`` along the chain
    ``S_i = {sigma(1), ..., sigma(i)}``. Agrees with ``f`` on the cube and is convex iff ``f``
    is submodular. ``O(n)`` value evaluations.
    """
    pv = np.asarray(p, dtype=float).reshape(-1)
    n = function.n
    if pv.shape[0] != n:
        raise ValueError(f"p must have length {n}, got {pv.shape[0]}")
    x = np.zeros(n, dtype=float)
    prev = float(function.value(x))
    total = prev  # f(empty) offset, so f_L agrees with f on vertices even if f(empty) != 0
    for i in np.argsort(-pv, kind="stable"):  # descending
        x[int(i)] = 1.0
        cur = float(function.value(x))
        total += float(pv[int(i)]) * (cur - prev)
        prev = cur
    return float(total)


def _base_vertex_min(function: SubmodularFunction, w: FloatArray) -> FloatArray:
    r"""The vertex of the base polytope ``B(f)`` minimizing ``<w, x>`` (Edmonds' greedy).

    Ordering ``w`` ascending and telescoping the marginals gives ``argmin_{x in B(f)} <w, x>``.
    """
    n = function.n
    q = np.zeros(n, dtype=float)
    x = np.zeros(n, dtype=float)
    prev = float(function.value(x))
    for i in np.argsort(w, kind="stable"):  # ascending
        x[int(i)] = 1.0
        cur = float(function.value(x))
        q[int(i)] = cur - prev
        prev = cur
    return q


def _affine_min_norm(corners: list[FloatArray]) -> FloatArray:
    r"""Minimum-norm affine combination: ``argmin_alpha ||sum_i alpha_i q_i||^2`` s.t. ``sum alpha = 1``."""
    q = np.stack(corners, axis=1)  # (n, k)
    k = q.shape[1]
    gram = q.T @ q
    ones = np.ones(k, dtype=float)
    kkt = np.zeros((k + 1, k + 1), dtype=float)
    kkt[:k, :k] = 2.0 * gram
    kkt[:k, k] = ones
    kkt[k, :k] = ones
    rhs = np.zeros(k + 1, dtype=float)
    rhs[k] = 1.0
    try:
        sol = np.linalg.solve(kkt, rhs)
    except np.linalg.LinAlgError:
        sol = np.linalg.lstsq(kkt, rhs, rcond=None)[0]
    return np.asarray(sol[:k], dtype=float)


def min_norm_point(
    function: SubmodularFunction, *, max_iter: int = 1000, tol: float = 1e-12
) -> FloatArray:
    r"""Wolfe's minimum-norm point of the base polytope ``B(f)`` (pure numpy).

    Alternates the exact linear oracle :func:`_base_vertex_min` (major cycle) with an affine
    minimization over the retained corners (minor cycle), converging to
    ``x* = argmin_{x in B(f)} ||x||^2``. Returns ``x*``; :func:`submodular_minimize` turns it
    into the optimal set.
    """
    n = function.n
    corners: list[FloatArray] = [_base_vertex_min(function, np.zeros(n, dtype=float))]
    lam = np.array([1.0], dtype=float)
    x = corners[0].copy()
    for _ in range(max_iter):
        q = _base_vertex_min(function, x)
        gap = float(x @ x) - float(x @ q)
        if gap <= tol * max(1.0, float(x @ x)):
            break
        if any(float(np.max(np.abs(q - c))) <= tol for c in corners):
            break
        corners.append(q)
        lam = np.append(lam, 0.0)
        for _ in range(max_iter):  # minor cycle
            alpha = _affine_min_norm(corners)
            if np.all(alpha > tol):
                x = np.stack(corners, axis=1) @ alpha
                lam = alpha
                break
            y = np.stack(corners, axis=1) @ alpha
            theta = 1.0
            for i in range(len(corners)):
                if alpha[i] < lam[i]:
                    denom = lam[i] - alpha[i]
                    if denom > tol:
                        theta = min(theta, float(lam[i] / denom))
            lam = (1.0 - theta) * lam + theta * alpha
            x = (1.0 - theta) * x + theta * y
            keep = lam > tol
            corners = [corners[i] for i in range(len(corners)) if keep[i]]
            lam = lam[keep]
    return np.asarray(x, dtype=float)


def submodular_minimize(
    function: SubmodularFunction, *, max_iter: int = 1000, tol: float = 1e-12
) -> MinimizerResult:
    r"""Exact unconstrained submodular minimization ``min_{S} f(S)`` (P-class, Fujishige-Wolfe).

    Computes the base-polytope min-norm point ``x*`` and recovers the minimizer from the chain
    of ``x*``-threshold sets: the optimal ``{i : x*_i < 0}`` is a prefix of the ascending-``x*``
    order, so evaluating ``f`` along that chain (``n + 1`` sets) returns the exact minimum. This
    is a genuine polynomial-time exact result -- **not** a ``P = NP`` claim (minimization, not
    the NP-hard maximization).
    """
    n = function.n
    x = min_norm_point(function, max_iter=max_iter, tol=tol)
    cur = np.zeros(n, dtype=float)
    best_sel = cur.copy()
    best_val = float(function.value(cur))
    for i in np.argsort(x, kind="stable"):  # ascending: most-negative x* first
        cur[int(i)] = 1.0
        val = float(function.value(cur))
        if val < best_val - _TOL:
            best_val = val
            best_sel = cur.copy()
    return MinimizerResult(
        selection=tuple(int(v) for v in best_sel), value=best_val, point=x
    )


__all__ = [
    "MinimizerResult",
    "lovasz_extension",
    "min_norm_point",
    "submodular_minimize",
]
