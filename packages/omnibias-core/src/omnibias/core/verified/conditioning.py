# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Certified conditioning of a symmetric matrix -- the rigorous register of the
``eps -> 0`` rank / regularization collapse.

Tikhonov-regularized solving ``x_eps = (A + eps I)^{-1} b`` collapses onto the
minimum-norm / Moore-Penrose solution ``A^+ b`` as ``eps -> 0``; naively taking
``eps`` to zero blows up whenever ``A`` is ill-conditioned or rank-deficient.
This module certifies the quantities that govern that collapse, all in
outward-rounded interval arithmetic on top of
:mod:`omnibias.core.verified.eig_operator`:

* :func:`certified_min_eigenvalue` / :func:`certified_max_eigenvalue` -- rigorous
  two-sided enclosures of ``lambda_min(A)`` / ``lambda_max(A)`` (the smallest /
  largest eigenvalue of the symmetric part of ``A``) by inertia bisection.
* :func:`certified_condition_number` -- an enclosure of ``kappa(A) =
  lambda_max / lambda_min`` (upper endpoint ``+inf`` when positive-definiteness
  cannot be certified -- the honest rank-deficient signal).
* :func:`certified_damping` -- the smallest ``eps`` for which ``kappa(A + eps I)``
  is provably ``<= target_condition`` (a certified damping *selection*).
* :func:`certified_regularization_error` -- a sound upper bound on
  ``||x_eps - A^+ b||_2`` on ``range(A)`` (the regime where the collapse
  converges), assuming a certified positive smallest eigenvalue.
* :func:`conditioning_certificate` -- a sealed v1 certificate bundling the above,
  carrying the ``LDL^T`` pivots so a positive-definiteness (``lambda_min > 0``)
  obligation can be handed to the Lean bridge unchanged.

This is a **rigorous** register: the enclosures are theorem-grade, but the
underlying regularized solve is a numerical (LAPACK-class) operation carried out
by the differentiable consumers (see :mod:`omnibias.curvature.regularize`). The
``eps -> 0`` limit is a *distinct* collapse from the founding multi-bias
``delta -> 0`` derivative limit and the ``beta -> inf`` feasibility penalty --
same spirit, different parameter, never conflated. Nothing here rides the
``sigma`` derivative tower; the omnibias value is the collapse framing plus the
certificate.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import inf
from typing import Any

from omnibias.core.verified.eig_operator import (
    generalized_eigenvalue_enclosure,
    interval_ldlt_pivots,
)
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.linalg import identity_matrix

# NOTE: ``omnibias.core.proof.certificate`` imports from ``omnibias.core.verified``,
# so the certificate-sealing helpers are imported lazily inside the two functions
# that seal certificates -- importing them at module load would create a cycle when
# ``verified/__init__`` pulls in this module.

FloatMatrix = Sequence[Sequence[float]]
FloatVector = Sequence[float]


def _symmetrize(matrix: FloatMatrix) -> list[list[float]]:
    r"""The symmetric part ``(A + A^T)/2`` as a float matrix (validates squareness).

    Curvature / Fisher / Gauss-Newton matrices are symmetric by construction; the
    certified quantities are statements about this symmetric object, so any tiny
    floating asymmetry is folded away first (and recorded as ``scope`` in the
    certificate).
    """
    rows = [[float(x) for x in row] for row in matrix]
    n = len(rows)
    if n == 0:
        raise ValueError("matrix must be non-empty")
    for row in rows:
        if len(row) != n:
            raise ValueError("matrix must be square")
    return [[0.5 * (rows[i][j] + rows[j][i]) for j in range(n)] for i in range(n)]


def _l2_norm_enclosure(vec: FloatVector) -> Interval:
    r"""Outward-rounded enclosure of ``||vec||_2`` (its ``.hi`` is a rigorous upper bound)."""
    acc = Interval.point(0.0)
    for x in vec:
        acc = acc + Interval.point(float(x)) ** 2
    return acc.sqrt()


def certified_min_eigenvalue(matrix: FloatMatrix) -> Interval:
    r"""Rigorous enclosure of the smallest eigenvalue of the symmetric part of ``matrix``.

    Wraps :func:`~omnibias.core.verified.eig_operator.generalized_eigenvalue_enclosure`
    on the pencil ``(A, I)`` at index ``1``. The ``.lo`` endpoint is a certified
    lower bound on ``lambda_min``; it is ``<= 0`` exactly when positive-definiteness
    cannot be certified (a rank-deficient / indefinite signal).
    """
    a = _symmetrize(matrix)
    return generalized_eigenvalue_enclosure(a, identity_matrix(len(a)), 1)


def certified_max_eigenvalue(matrix: FloatMatrix) -> Interval:
    r"""Rigorous enclosure of the largest eigenvalue of the symmetric part of ``matrix``."""
    a = _symmetrize(matrix)
    n = len(a)
    return generalized_eigenvalue_enclosure(a, identity_matrix(n), n)


def _condition_from(lam_min: Interval, lam_max: Interval) -> Interval:
    r"""Compose ``kappa`` from certified extreme-eigenvalue enclosures (``+inf`` if not PD)."""
    if lam_min.lo <= 0.0:
        lo = 1.0
        if lam_min.hi > 0.0 and lam_max.lo > 0.0:
            lo = (Interval.point(lam_max.lo) * Interval.point(lam_min.hi).reciprocal()).lo
        return Interval(max(1.0, lo), inf)
    return lam_max * lam_min.reciprocal()


def certified_condition_number(matrix: FloatMatrix) -> Interval:
    r"""Enclosure of the spectral condition number ``kappa(A) = lambda_max / lambda_min``.

    When ``lambda_min`` cannot be certified positive (rank-deficient / indefinite
    ``A``), the upper endpoint is ``+inf`` -- the honest statement that the
    condition number is not bounded above by this certificate.
    """
    a = _symmetrize(matrix)
    n = len(a)
    lam_min = generalized_eigenvalue_enclosure(a, identity_matrix(n), 1)
    lam_max = generalized_eigenvalue_enclosure(a, identity_matrix(n), n)
    return _condition_from(lam_min, lam_max)


def certified_damping(matrix: FloatMatrix, *, target_condition: float) -> float:
    r"""Smallest ``eps >= 0`` provably giving ``kappa(A + eps I) <= target_condition``.

    ``kappa(A + eps I) = (lambda_max + eps)/(lambda_min + eps)`` is decreasing in
    ``eps``, so the sound threshold ``eps* = max(0, (lambda_max.hi -
    T * lambda_min.lo)/(T - 1))`` -- built from the *upper* enclosure of
    ``lambda_max`` and the *lower* enclosure of ``lambda_min`` -- guarantees the
    target for the true matrix (a larger ``eps`` only lowers ``kappa``, so the
    outward-rounded value is sufficient). Works for indefinite ``A`` too: the same
    ``eps`` lifts ``lambda_min + eps`` strictly positive.
    """
    t = float(target_condition)
    if t <= 1.0:
        raise ValueError("target_condition must be > 1")
    a = _symmetrize(matrix)
    n = len(a)
    lam_min = generalized_eigenvalue_enclosure(a, identity_matrix(n), 1)
    lam_max = generalized_eigenvalue_enclosure(a, identity_matrix(n), n)
    num = Interval.point(lam_max.hi) - Interval.point(t) * Interval.point(lam_min.lo)
    den = Interval.point(t - 1.0)  # exact-point positive denominator
    eps = (num * den.reciprocal()).hi
    return max(0.0, eps)


def certified_regularization_error(
    matrix: FloatMatrix, rhs: FloatVector, eps: float, *, min_eig: float | None = None
) -> Interval:
    r"""Sound upper bound on ``||(A + eps I)^{-1} b - A^+ b||_2`` on ``range(A)``.

    On the range of a symmetric ``A`` the Tikhonov error is
    ``x_eps - x0 = -eps (A + eps I)^{-1} x0`` with ``x0 = A^+ b``, so

    .. math:: \|x_\eps - x_0\| \le \frac{\eps}{\lambda_{\min}+\eps}\,\frac{\|b\|}{\lambda_{\min}}.

    The returned interval is ``[0, bound]`` with ``bound`` built from the certified
    lower bound ``lambda_min.lo`` (the value that maximises -- hence soundly
    bounds -- the error) and an outward enclosure of ``||b||_2``.

    Scope (``range_consistent``): this bounds the error only for the component of
    ``b`` in ``range(A)``. If ``b`` has a null-space component the naive Tikhonov
    solve carries a ``(1/eps) b_null`` term that **diverges** as ``eps -> 0`` --
    exactly the blow-up the collapse must avoid -- and is *not* covered here. For a
    rank-deficient ``A`` pass ``min_eig`` = a certified lower bound on the smallest
    **nonzero** eigenvalue (and drive the null space out with
    :func:`omnibias.curvature.regularize.min_norm_solve`).
    """
    e = float(eps)
    if e < 0.0:
        raise ValueError("eps must be >= 0")
    if min_eig is None:
        lam_lo = certified_min_eigenvalue(matrix).lo
    else:
        lam_lo = float(min_eig)
    if lam_lo <= 0.0:
        raise ValueError(
            "certified_regularization_error needs a positive certified smallest "
            "eigenvalue; for rank-deficient A pass min_eig = a certified lower bound "
            "on the smallest nonzero eigenvalue (range-consistent scope)"
        )
    lam_iv = Interval.point(lam_lo)
    eps_iv = Interval.point(e)
    factor = eps_iv * (lam_iv + eps_iv).reciprocal()  # eps / (lambda_min + eps) <= 1
    bound = factor * (_l2_norm_enclosure(rhs) * lam_iv.reciprocal())
    return Interval(0.0, bound.hi)


def conditioning_certificate(
    matrix: FloatMatrix, *, target_condition: float | None = None, eps: float | None = None
) -> dict[str, Any]:
    r"""A sealed v1 certificate bundling the certified conditioning of ``matrix``.

    Carries the ``lambda_min`` / ``lambda_max`` / ``kappa`` enclosures and, when
    the symmetric part is certified positive definite, its interval ``LDL^T``
    pivots -- the exact payload the Lean bridge turns into the ``allPivotsPos``
    (``lambda_min > 0``) obligation. When ``target_condition`` is given the
    certified damping is recorded; when ``eps`` is given (and the matrix is PD) the
    regularization-error bound is attached. ``theorem_prover_verified`` is *not*
    set here -- it is earned only by passing the sealed certificate through the
    proof machine with a Lean toolchain present.
    """
    from omnibias.core.proof.certificate import encode_interval, make_certificate

    a = _symmetrize(matrix)
    n = len(a)
    lam_min = generalized_eigenvalue_enclosure(a, identity_matrix(n), 1)
    lam_max = generalized_eigenvalue_enclosure(a, identity_matrix(n), n)
    kappa = _condition_from(lam_min, lam_max)
    pivots = interval_ldlt_pivots(a)
    is_pd = pivots is not None and all(p.lo > 0.0 for p in pivots)

    payload: dict[str, Any] = {
        "type": "conditioning",
        "n": n,
        "lambda_min": encode_interval(lam_min),
        "lambda_max": encode_interval(lam_max),
        "condition_number": encode_interval(kappa),
        "positive_definite": is_pd,
    }
    if pivots is not None:
        payload["pivots"] = [encode_interval(p) for p in pivots]
    if target_condition is not None:
        payload["target_condition"] = float(target_condition)
        payload["certified_damping"] = certified_damping(a, target_condition=target_condition)
    if eps is not None:
        payload["eps"] = float(eps)
        if is_pd:
            # The error bound scales linearly in ||b||, so expose the certified
            # per-unit-||b|| factor eps / ((lambda_min + eps) * lambda_min).
            eps_iv = Interval.point(float(eps))
            unit_factor = eps_iv * (lam_min + eps_iv).reciprocal() * lam_min.reciprocal()
            payload["regularization_error_per_unit_rhs_norm"] = encode_interval(
                Interval(0.0, unit_factor.hi)
            )

    meta = {"scope": "symmetric_part", "collapse_axis": "eps->0"}
    honesty = {"unproven_claim": False}
    return make_certificate(
        claim=f"conditioning enclosure of a {n}x{n} symmetric matrix (eps->0 collapse)",
        payload=payload,
        honesty=honesty,
        meta=meta,
    )


def positive_definite_pivots_certificate(
    matrix: FloatMatrix, *, claim: str | None = None
) -> dict[str, Any]:
    r"""A sealed positive-definiteness certificate (``lambda_min > 0``) for ``matrix``.

    Convenience wrapper that seals the interval ``LDL^T`` pivots of the symmetric
    part as a scalar ``lambda_min`` lower-bound interval certificate -- the
    Lean-dischargeable ``lambda_min >= c > 0`` obligation. Raises ``ValueError``
    when positive-definiteness cannot be certified.
    """
    from omnibias.core.proof.certificate import interval_certificate

    lam_min = certified_min_eigenvalue(matrix)
    if lam_min.lo <= 0.0:
        raise ValueError("matrix is not certified positive definite (lambda_min.lo <= 0)")
    return interval_certificate(
        claim or "lambda_min(A) certified positive",
        lam_min,
        meta={"scope": "symmetric_part", "collapse_axis": "eps->0"},
    )


__all__ = [
    "FloatMatrix",
    "FloatVector",
    "certified_condition_number",
    "certified_damping",
    "certified_max_eigenvalue",
    "certified_min_eigenvalue",
    "certified_regularization_error",
    "conditioning_certificate",
    "positive_definite_pivots_certificate",
]
