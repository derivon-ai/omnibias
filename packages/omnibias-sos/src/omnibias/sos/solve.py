# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""A compact float SDP *proposer* for Sum-of-Squares Gram matrices.

**This module never proves anything.**  It proposes a floating-point,
strictly-positive-definite Gram matrix ``Q`` with ``z(x)^T Q z(x) approx p(x)``.
The proof happens later in :mod:`omnibias.sos.certify`, which rationally rounds
``Q`` and runs a rigorous interval ``LDL^T`` positive-definiteness check.  This
mirrors the repo's ``neumann_inverse_norm_bound`` pattern: a float solver
proposes, a verified routine certifies -- so the SDP can be floating-point and
even swapped for an external backend without touching soundness.

The core is a primal log-det interior-point method in the symmetric-vectorisation
(``svec``) space, parametrised over the affine "coefficient-matching" subspace so
that every iterate satisfies ``z^T Q z = p`` exactly (up to float round-off).  For
pure SOS feasibility it computes the analytic centre (the maximum-determinant
Gram, which rounds well); a general free-variable + linear-objective form (used by
the auxiliary-functional method) follows a short barrier path.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from omnibias.sos.monomials import gram_products
from omnibias.sos.problem import Exponent, Polynomial

_SQRT2 = float(np.sqrt(2.0))


@dataclass(frozen=True)
class SDPResult:
    """Outcome of the float SDP proposer (advisory only, never a proof)."""

    status: str
    """``"solved"`` (a PD proposal found), ``"infeasible"``, or ``"failed"``."""
    gram: np.ndarray | None
    """The proposed symmetric Gram matrix ``Q`` (float), or ``None``."""
    free_vars: np.ndarray | None
    """Optimised free variables ``y`` (auxiliary form), or ``None``."""
    min_eig: float
    """Smallest eigenvalue of the proposed ``Q`` (a float health check)."""
    detail: str


# --------------------------------------------------------------------------- #
# symmetric vectorisation (svec): <A, Q>_F == svec(A) . svec(Q)
# --------------------------------------------------------------------------- #


def _svec_dim(m: int) -> int:
    return m * (m + 1) // 2


def _svec(matrix: np.ndarray, m: int) -> np.ndarray:
    out = np.empty(_svec_dim(m))
    k = 0
    for i in range(m):
        for j in range(i, m):
            out[k] = matrix[i, i] if i == j else _SQRT2 * matrix[i, j]
            k += 1
    return out


def _smat(vector: np.ndarray, m: int) -> np.ndarray:
    matrix = np.zeros((m, m))
    k = 0
    for i in range(m):
        for j in range(i, m):
            if i == j:
                matrix[i, i] = vector[k]
            else:
                matrix[i, j] = matrix[j, i] = vector[k] / _SQRT2
            k += 1
    return matrix


def _svec_indices(m: int) -> dict[tuple[int, int], int]:
    idx: dict[tuple[int, int], int] = {}
    k = 0
    for i in range(m):
        for j in range(i, m):
            idx[(i, j)] = k
            k += 1
    return idx


def sos_constraint_system(
    polynomial: Polynomial, basis: Sequence[Exponent]
) -> tuple[np.ndarray, np.ndarray] | None:
    r"""Build the coefficient-matching rows ``A`` and targets ``t``.

    Returns ``(A, t)`` with ``A[k] . svec(Q) == t[k]`` encoding "the coefficient of
    product monomial ``alpha_k`` in ``z^T Q z`` equals ``p_{alpha_k}``", or ``None``
    if the polynomial has a monomial no basis product can build (so it cannot be
    SOS in this basis).
    """
    m = len(basis)
    idx = _svec_indices(m)
    products = gram_products(basis)
    if not polynomial.support <= set(products):
        return None
    alphas = sorted(products)
    rows = np.zeros((len(alphas), _svec_dim(m)))
    targets = np.zeros(len(alphas))
    for row, alpha in enumerate(alphas):
        for i, j, _mult in products[alpha]:
            rows[row, idx[(i, j)]] = 1.0 if i == j else _SQRT2
        targets[row] = polynomial.coefficient(alpha)
    return rows, targets


# --------------------------------------------------------------------------- #
# interior-point core
# --------------------------------------------------------------------------- #


def _nullspace_and_particular(
    lhs: np.ndarray, rhs: np.ndarray, feas_tol: float
) -> tuple[np.ndarray, np.ndarray] | None:
    """Least-norm particular solution + orthonormal nullspace of ``lhs u = rhs``."""
    particular, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
    residual = lhs @ particular - rhs
    scale = 1.0 + float(np.linalg.norm(rhs))
    if float(np.linalg.norm(residual)) > feas_tol * scale:
        return None
    _u, sing, vt = np.linalg.svd(lhs)
    tol = max(lhs.shape) * np.finfo(float).eps * (float(sing.max()) if sing.size else 0.0)
    rank = int((sing > tol).sum())
    nullspace = vt[rank:].T
    return particular, nullspace


def _interior_point(
    lhs: np.ndarray,
    rhs: np.ndarray,
    obj: np.ndarray,
    m: int,
    *,
    max_newton: int,
    barrier_steps: int,
    feas_tol: float,
    pd_target: float,
    reg: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Minimise ``obj . u + 0.5 * sum_i reg_i u_i^2`` over ``{lhs u = rhs, smat(u_Q) >= 0}``.

    ``u = (svec(Q), y)``; the optional per-coordinate ridge ``reg`` keeps chosen free
    variables well-scaled (a tie-breaker; it never affects soundness -- the proof is
    the downstream verified certificate).  Returns ``(Q, y)`` on success or ``None``.
    """
    svec_dim = _svec_dim(m)
    solved = _nullspace_and_particular(lhs, rhs, feas_tol)
    if solved is None:
        return None
    particular, nullspace = solved
    n_dir = nullspace.shape[1]
    reg_vec = np.zeros(particular.shape[0]) if reg is None else np.asarray(reg, dtype=float)

    def u_of(s: np.ndarray) -> np.ndarray:
        return np.asarray(particular + nullspace @ s, dtype=float)

    def q_of(s: np.ndarray) -> np.ndarray:
        return _smat(u_of(s)[:svec_dim], m)

    def y_of(s: np.ndarray) -> np.ndarray:
        return u_of(s)[svec_dim:]

    if n_dir == 0:
        matrix = q_of(np.zeros(0))
        eig = float(np.linalg.eigvalsh(matrix).min())
        if eig <= 0.0:
            return None
        return matrix, y_of(np.zeros(0))

    # Directions dQ_k for the gradient of -log det Q along each nullspace column.
    dq = [_smat(nullspace[:svec_dim, k], m) for k in range(n_dir)]
    obj_dir = obj @ nullspace  # gradient of obj . u along each direction
    reg_hess = nullspace.T @ (reg_vec[:, None] * nullspace)  # Hessian of the ridge term

    started = _phase_one(particular, nullspace, dq, m, svec_dim, pd_target)
    if started is None:
        return None
    s: np.ndarray = started

    def is_pd(mat: np.ndarray) -> bool:
        try:
            np.linalg.cholesky(mat)
        except np.linalg.LinAlgError:
            return False
        return True

    weight = 0.0 if not np.any(obj) else 1.0
    n_outer = 1 if weight == 0.0 else barrier_steps
    for _outer in range(n_outer):
        for _inner in range(max_newton):
            u = u_of(s)
            matrix = _smat(u[:svec_dim], m)
            qinv = np.linalg.inv(matrix)
            grad = np.empty(n_dir)
            hess = np.empty((n_dir, n_dir))
            qinv_dq = [qinv @ d for d in dq]
            reg_grad = nullspace.T @ (reg_vec * u)
            for k in range(n_dir):
                grad[k] = weight * obj_dir[k] - np.trace(qinv_dq[k]) + reg_grad[k]
                for lidx in range(k, n_dir):
                    val = float(np.sum(qinv_dq[k] * qinv_dq[lidx].T))
                    hess[k, lidx] = hess[lidx, k] = val
            hess += reg_hess + 1e-12 * np.eye(n_dir)
            try:
                step = -np.linalg.solve(hess, grad)
            except np.linalg.LinAlgError:
                break
            decrement = float(-grad @ step)
            if decrement < 1e-14:
                break
            alpha = 1.0
            base = _phi(matrix, u, obj, weight, reg_vec)
            while alpha > 1e-10:
                cand_u = u_of(s + alpha * step)
                cand = _smat(cand_u[:svec_dim], m)
                if is_pd(cand) and _phi(cand, cand_u, obj, weight, reg_vec) <= base - 1e-4 * alpha * decrement:
                    break
                alpha *= 0.5
            if alpha <= 1e-10:
                break
            s = s + alpha * step
        weight *= 10.0

    matrix = q_of(s)
    eig = float(np.linalg.eigvalsh(matrix).min())
    if eig <= 0.0:
        return None
    return matrix, y_of(s)


def _phi(
    matrix: np.ndarray, u: np.ndarray, obj: np.ndarray, weight: float, reg: np.ndarray
) -> float:
    sign, logdet = np.linalg.slogdet(matrix)
    if sign <= 0.0:
        return float("inf")
    linear = weight * float(obj @ u)
    ridge = 0.5 * float(np.sum(reg * u * u))
    return linear + ridge - float(logdet)


def _phase_one(
    particular: np.ndarray,
    nullspace: np.ndarray,
    dq: list[np.ndarray],
    m: int,
    svec_dim: int,
    pd_target: float,
) -> np.ndarray | None:
    """Find nullspace coordinates ``s`` with ``smat(u_Q) >= pd_target * I``."""
    q_dir = nullspace[:svec_dim, :]
    q_part = particular[:svec_dim]
    # Warm start: the feasible Q closest to the identity.
    target = _svec(np.eye(m), m) - q_part
    s: np.ndarray = np.asarray(np.linalg.lstsq(q_dir, target, rcond=None)[0], dtype=float)

    def q_of(sv: np.ndarray) -> np.ndarray:
        return _smat(q_part + q_dir @ sv, m)

    for it in range(4000):
        matrix = q_of(s)
        eigvals, eigvecs = np.linalg.eigh(matrix)
        lam = float(eigvals[0])
        if lam > pd_target:
            return s
        vec = eigvecs[:, 0]
        subgrad = np.array([float(vec @ (d @ vec)) for d in dq])
        norm = float(np.linalg.norm(subgrad))
        if norm < 1e-14:
            return None
        step = (pd_target - lam + 1.0) / norm
        s = s + step * subgrad / norm
        if it > 200 and lam <= -1e6:
            return None
    matrix = q_of(s)
    return s if float(np.linalg.eigvalsh(matrix).min()) > pd_target else None


# --------------------------------------------------------------------------- #
# public entry points
# --------------------------------------------------------------------------- #

GramProposer = Callable[[np.ndarray, np.ndarray, int], "np.ndarray | None"]


def solve_sos_gram(
    polynomial: Polynomial,
    basis: Sequence[Exponent],
    *,
    max_newton: int = 60,
    feas_tol: float = 1e-8,
    pd_target: float = 1e-4,
    external: GramProposer | None = None,
) -> SDPResult:
    r"""Propose a PD Gram matrix ``Q`` with ``z(x)^T Q z(x) approx polynomial``.

    Parameters
    ----------
    polynomial, basis:
        The polynomial and the monomial basis ``z(x)`` (exponent tuples).
    external:
        Optional external backend ``(A, t, m) -> Q`` (e.g. a wrapper around a
        dedicated SDP solver).  Advisory only -- the proposal is still rationally
        rounded and rigorously certified downstream.

    Returns
    -------
    SDPResult
        ``status == "solved"`` carries a floating-point PD proposal.  This is
        **not** a proof of nonnegativity; certification happens in
        :func:`omnibias.sos.certify.certify_sos`.
    """
    system = sos_constraint_system(polynomial, basis)
    if system is None:
        return SDPResult(
            "infeasible", None, None, float("nan"),
            "polynomial has a monomial no basis product can build",
        )
    rows, targets = system
    m = len(basis)

    if external is not None:
        proposal = external(rows, targets, m)
        if proposal is None:
            return SDPResult("failed", None, None, float("nan"), "external solver returned no Gram")
        matrix = np.asarray(proposal, dtype=float)
        eig = float(np.linalg.eigvalsh(matrix).min())
        status = "solved" if eig > 0.0 else "failed"
        return SDPResult(status, matrix, None, eig, f"external solver (min eig {eig:.3e})")

    result = _interior_point(
        rows, targets, np.zeros(rows.shape[1]), m,
        max_newton=max_newton, barrier_steps=1, feas_tol=feas_tol, pd_target=pd_target,
    )
    if result is None:
        return SDPResult(
            "infeasible", None, None, float("nan"),
            "no positive-definite Gram matrix matches the coefficients (not SOS in this basis)",
        )
    matrix, _y = result
    eig = float(np.linalg.eigvalsh(matrix).min())
    return SDPResult("solved", matrix, None, eig, f"analytic-centre Gram (min eig {eig:.3e})")


def solve_gram_program(
    m: int,
    lhs: np.ndarray,
    rhs: np.ndarray,
    objective: np.ndarray,
    *,
    max_newton: int = 40,
    barrier_steps: int = 6,
    feas_tol: float = 1e-8,
    pd_target: float = 1e-4,
    reg: np.ndarray | None = None,
) -> SDPResult:
    r"""General SOS program: minimise ``objective . (svec(Q), y)`` with ``Q >= 0``.

    Used by the auxiliary-functional method, where the free variables ``y`` are an
    auxiliary functional's coefficients and a bound ``C``.  The optional per-coordinate
    ridge ``reg`` keeps chosen free variables well-scaled (a numeric tie-breaker only;
    soundness rests entirely on the downstream verified certificate).  Advisory float
    proposal only; the certified bound comes from certifying the resulting SOS
    polynomial.
    """
    result = _interior_point(
        lhs, rhs, objective, m,
        max_newton=max_newton, barrier_steps=barrier_steps, feas_tol=feas_tol,
        pd_target=pd_target, reg=reg,
    )
    if result is None:
        return SDPResult("infeasible", None, None, float("nan"), "no PD feasible point found")
    matrix, y = result
    eig = float(np.linalg.eigvalsh(matrix).min())
    return SDPResult("solved", matrix, y, eig, f"barrier-path solution (min eig {eig:.3e})")


__all__ = [
    "GramProposer",
    "SDPResult",
    "solve_gram_program",
    "solve_sos_gram",
    "sos_constraint_system",
]
