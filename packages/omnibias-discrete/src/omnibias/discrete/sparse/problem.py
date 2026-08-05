# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Sparse recovery as two problems on the ``DiscreteProblem`` seam.

Best-subset selection -- choose the support ``S`` of a coefficient vector to minimise a
least-squares fit plus a cardinality (``l_0``) penalty -- is the combinatorial heart of
sparse regression, and it is NP-hard. Two honest encodings live here:

* :class:`SupportSelectionProblem` (Fork A) -- the **pseudo-Boolean surrogate**. The
  binary variables *are* the support ``z in {0, 1}^n`` and the energy
  ``E(z) = 1/2 ||A z - b||^2 + lambda 1^T z`` is a quadratic (QUBO), so it implements the
  full substrate seam (``energy`` + ``to_polynomial`` + a closed-form ``flip_deltas``) and
  is certified directly by :func:`omnibias.discrete.certify_gap` (Lasserre / SOS).

* :class:`BestSubsetProblem` (Fork B) -- the **continuous-coefficient** objective
  ``E(z) = min_w ||A[:, supp(z)] w - b||^2 + lambda |z|`` (an inner OLS refit on the
  selected columns). This is *not* pseudo-Boolean, so it deliberately omits
  ``to_polynomial`` and is certified by the custom convex bound in
  :func:`omnibias.discrete.sparse.certify_best_subset_gap`; it still plugs into the
  energy-only decoder / oracle (:func:`omnibias.discrete.decode`,
  :func:`omnibias.discrete.brute_force_min`).

yes-if framing: exact best-subset selection in poly time would imply ``P = NP``; the
deliverable is a *certified optimality gap*, never an exactness claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:  # pragma: no cover - typing only
    from omnibias.sos import Polynomial

FloatArray = NDArray[np.float64]


def _as_design(a: object, b: object) -> tuple[FloatArray, FloatArray]:
    """Validate and coerce a design matrix ``A`` ``(m, n)`` and target ``b`` ``(m,)``."""
    am = np.asarray(a, dtype=float)
    bv = np.asarray(b, dtype=float).reshape(-1)
    if am.ndim != 2:
        raise ValueError(f"A must be a 2-D (m, n) design matrix, got shape {am.shape}")
    if am.shape[0] < 1 or am.shape[1] < 1:
        raise ValueError("A must have at least one row and one column")
    if bv.shape[0] != am.shape[0]:
        raise ValueError(f"b must have length {am.shape[0]} (rows of A), got {bv.shape[0]}")
    return am, bv


@dataclass(frozen=True)
class SupportSelectionProblem:
    r"""Best-subset selection as a pseudo-Boolean QUBO over the support ``z in {0, 1}^n``.

    ``E(z) = 1/2 ||A z - b||^2 + lambda 1^T z`` expands to the QUBO
    ``z^T Q z + c^T z + const`` with ``Q = 1/2 A^T A`` (symmetric), ``c = -A^T b + lambda 1``
    and ``const = 1/2 ||b||^2``; on the cube ``z_i^2 = z_i`` so ``energy`` and
    ``to_polynomial`` agree. Implements the substrate's ``DiscreteProblem`` seam (with the
    QUBO closed-form ``flip_deltas`` fast path), so it plugs straight into
    :func:`omnibias.discrete.decode` / :func:`omnibias.discrete.certify_gap` and the
    ``l_p`` annealed relaxation twins.

    Attributes
    ----------
    A:
        ``(m, n)`` design / dictionary matrix.
    b:
        ``(m,)`` target vector.
    lam:
        Nonnegative cardinality (``l_0``) penalty ``lambda`` per selected column.
    name:
        Optional label.
    """

    A: FloatArray
    b: FloatArray
    lam: float = 0.0
    name: str | None = None
    Q: FloatArray = field(init=False, repr=False)
    c: FloatArray = field(init=False, repr=False)
    const: float = field(init=False, repr=False)
    gram_matrix: FloatArray = field(init=False, repr=False)
    correlation: FloatArray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        am, bv = _as_design(self.A, self.b)
        if self.lam < 0.0:
            raise ValueError("lam (the l_0 penalty) must be nonnegative")
        gram = am.T @ am
        corr = am.T @ bv
        n = am.shape[1]
        object.__setattr__(self, "A", am)
        object.__setattr__(self, "b", bv)
        object.__setattr__(self, "lam", float(self.lam))
        object.__setattr__(self, "gram_matrix", gram)
        object.__setattr__(self, "correlation", corr)
        object.__setattr__(self, "Q", 0.5 * gram)
        object.__setattr__(self, "c", -corr + float(self.lam) * np.ones(n))
        object.__setattr__(self, "const", 0.5 * float(bv @ bv))

    @property
    def n(self) -> int:
        return int(self.A.shape[1])

    def energy(self, x: object) -> float | FloatArray:
        r"""``1/2 ||A x - b||^2 + lambda 1^T x`` at one point ``(n,)`` or a batch ``(m, n)``."""
        xv = np.asarray(x, dtype=float)
        single = xv.ndim == 1
        matrix = xv.reshape(1, -1) if single else xv
        resid = matrix @ self.A.T - self.b
        fit = 0.5 * np.sum(resid * resid, axis=-1)
        penalty = self.lam * np.sum(matrix, axis=-1)
        total = fit + penalty
        return float(total[0]) if single else total

    def flip_deltas(self, x: object) -> FloatArray:
        r"""Closed-form energy change of flipping each single bit of ``x`` (one matvec).

        For the QUBO ``z^T Q z + c^T z`` the flip-delta on the cube is
        ``(1 - 2 z_i)(Q_ii + 2 (Q z)_i - 2 Q_ii z_i + c_i)`` -- the fast path the shared
        local-search decoder uses instead of the generic batched-energy fallback.
        """
        q = self.Q
        diag = np.diag(q).copy()
        xv = np.asarray(x, dtype=float)
        grad = diag + 2.0 * (q @ xv) - 2.0 * diag * xv + self.c
        deltas: FloatArray = (1.0 - 2.0 * xv) * grad
        return deltas

    def to_polynomial(self) -> Polynomial:
        r"""The QUBO energy as an :class:`omnibias.sos.Polynomial` over ``n`` variables."""
        from omnibias.sos import Polynomial

        n = self.n
        poly = Polynomial.constant(float(self.const), n)
        variables = [Polynomial.variable(i, n) for i in range(n)]
        for i in range(n):
            ci = float(self.c[i])
            if ci != 0.0:
                poly = poly + variables[i] * ci
            for j in range(n):
                qij = float(self.Q[i, j])
                if qij != 0.0:
                    poly = poly + (variables[i] * variables[j]) * qij
        return poly

    def grad_scale(self) -> float:
        r"""A conservative Lipschitz-like ``scale`` for the data-fit gradient ``A^T A x - A^T b``."""
        gram_norm = float(np.linalg.norm(self.gram_matrix, 2))
        return max(1.0, gram_norm + float(np.max(np.abs(self.c))))


@dataclass(frozen=True)
class BestSubsetProblem:
    r"""Continuous-coefficient best-subset selection ``min_w ||A_S w - b||^2 + lambda |z|``.

    The binary ``z in {0, 1}^n`` selects the support ``S = supp(z)``; the coefficients on
    ``S`` are the ordinary-least-squares fit of the selected columns, so
    ``energy(z) = 1/2 ||A[:, S] w* - b||^2 + lambda |S|`` with ``w* = lstsq(A[:, S], b)``.
    This objective is **not** a pseudo-Boolean polynomial (the inner OLS is a linear
    solve), so it exposes ``n`` + ``energy`` (+ :meth:`refit`) for the energy-only decoder /
    oracle but **not** ``to_polynomial``; certify it with
    :func:`omnibias.discrete.sparse.certify_best_subset_gap` instead of
    :func:`omnibias.discrete.certify_gap`.

    Attributes
    ----------
    A, b:
        Design matrix ``(m, n)`` and target ``(m,)``.
    lam:
        Nonnegative cardinality (``l_0``) penalty per selected column.
    name:
        Optional label.
    """

    A: FloatArray
    b: FloatArray
    lam: float = 0.0
    name: str | None = None

    def __post_init__(self) -> None:
        am, bv = _as_design(self.A, self.b)
        if self.lam < 0.0:
            raise ValueError("lam (the l_0 penalty) must be nonnegative")
        object.__setattr__(self, "A", am)
        object.__setattr__(self, "b", bv)
        object.__setattr__(self, "lam", float(self.lam))

    @property
    def n(self) -> int:
        return int(self.A.shape[1])

    def _residual_on_support(self, support: NDArray[np.bool_]) -> float:
        """The OLS residual ``1/2 ||A[:, S] w* - b||^2`` for a boolean support mask ``S``."""
        if not bool(np.any(support)):
            return 0.5 * float(self.b @ self.b)
        a_sel = self.A[:, support]
        w, *_ = np.linalg.lstsq(a_sel, self.b, rcond=None)
        resid = a_sel @ w - self.b
        return 0.5 * float(resid @ resid)

    def energy(self, x: object) -> float | FloatArray:
        r"""``1/2 ||A_S w* - b||^2 + lambda |z|`` at one point ``(n,)`` or a batch ``(m, n)``."""
        xv = np.asarray(x, dtype=float)
        single = xv.ndim == 1
        matrix = xv.reshape(1, -1) if single else xv
        out = np.empty(matrix.shape[0], dtype=float)
        for row in range(matrix.shape[0]):
            support = matrix[row] >= 0.5
            out[row] = self._residual_on_support(support) + self.lam * float(np.sum(support))
        return float(out[0]) if single else out

    def refit(self, x: object) -> tuple[FloatArray, float]:
        r"""OLS refit on ``supp(x)``: the full-length coefficient vector ``w`` and residual.

        ``w`` is zero off the support and the least-squares fit of ``A[:, S]`` on the
        support; the returned residual is ``1/2 ||A w - b||^2`` (the fit term of
        :meth:`energy`, without the penalty).
        """
        xv = np.asarray(x, dtype=float).reshape(-1)
        support = xv >= 0.5
        w = np.zeros(self.n, dtype=float)
        if bool(np.any(support)):
            a_sel = self.A[:, support]
            w_sel, *_ = np.linalg.lstsq(a_sel, self.b, rcond=None)
            w[support] = w_sel
        resid = self.A @ w - self.b
        return w, 0.5 * float(resid @ resid)


__all__ = ["BestSubsetProblem", "SupportSelectionProblem"]
