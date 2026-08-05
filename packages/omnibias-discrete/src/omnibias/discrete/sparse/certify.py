# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Certificates for sparse recovery: the continuous (Fork B) bound and the hybrid (Fork C).

* :func:`certify_best_subset_gap` (Fork B) certifies a decoded support for the
  continuous-coefficient :class:`~omnibias.discrete.sparse.problem.BestSubsetProblem`,
  which is *not* pseudo-Boolean. Its lower bound is the always-valid **full-OLS-residual
  floor** ``1/2 ||A w_OLS - b||^2 <= any-subset residual <= E_min``, sealed (when
  ``A^T A`` is positive definite) by :func:`omnibias.convex.certify_qp_optimum` via weak
  duality at the zero dual (``g(0) = -1/2 c^T Q^{-1} c`` is exactly the unconstrained
  least-squares minimum). Degrades honestly to the unsealed float floor
  (``method="ols_floor"``, ``certified=False``) when ``omnibias-convex`` is absent or the
  conditioning check declines ``A^T A``.

* :func:`certified_sparse_fit` (Fork C) is the hybrid: encode the pseudo-Boolean
  **surrogate** (:class:`~omnibias.discrete.sparse.problem.SupportSelectionProblem`),
  decode a support, seal *that surrogate's* gap with
  :func:`omnibias.discrete.certify_gap` (Lasserre / SOS), then OLS-refit on the decoded
  support for the continuous coefficients. The returned :class:`SparseFitResult` states
  explicitly that the seal is on the pseudo-Boolean surrogate, not the continuous
  best-subset objective.

yes-if framing: exact best-subset selection is NP-hard; every certificate here is a
*sandwich* ``lower_bound <= optimum <= energy`` -- never an exactness (P = NP) claim.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from omnibias.discrete._core.decode import decode, is_binary
from omnibias.discrete._core.solution import GapCertificate
from omnibias.discrete.certify import certify_gap
from omnibias.discrete.sparse.problem import BestSubsetProblem, SupportSelectionProblem

FloatArray = NDArray[np.float64]


def certify_best_subset_gap(problem: BestSubsetProblem, z: object) -> GapCertificate:
    r"""Certify a decoded support ``z`` for the continuous best-subset objective.

    Parameters
    ----------
    problem:
        The :class:`~omnibias.discrete.sparse.problem.BestSubsetProblem` (its ``A`` / ``b``
        / ``lambda`` define the energy).
    z:
        A binary support ``z in {0, 1}^n`` (e.g. from :func:`omnibias.discrete.decode`) --
        the upper bound ``E(z)``.

    Returns
    -------
    A :class:`~omnibias.discrete._core.solution.GapCertificate` with ``method="convex"``
    (interval-sealed OLS-residual floor) or ``method="ols_floor"`` (valid but unsealed
    float floor) as the rigorous lower bound and ``E(z)`` as the upper bound.
    """
    n = problem.n
    zv = np.asarray(z, dtype=float).reshape(-1)
    if zv.shape[0] != n:
        raise ValueError(f"z must have length {n}, got {zv.shape[0]}")
    if not is_binary(zv):
        raise ValueError("z must be a binary support in {0, 1}^n (decode the relaxation first)")
    upper = float(problem.energy(zv))

    a_mat = np.asarray(problem.A, dtype=float)
    b_vec = np.asarray(problem.b, dtype=float)
    # Always-valid floor: the unconstrained OLS residual over *all* features is <= the
    # residual of any subset (adding columns can only lower it), and the penalty is >= 0,
    # so 1/2 ||A w_OLS - b||^2 <= E(z*) = E_min for every support z*.
    w_ols, *_ = np.linalg.lstsq(a_mat, b_vec, rcond=None)
    resid = a_mat @ w_ols - b_vec
    floor = 0.5 * float(resid @ resid)

    try:
        from omnibias.convex import certify_qp_optimum

        gram = a_mat.T @ a_mat
        corr = a_mat.T @ b_vec
        # Weak duality at dual = 0: g(0) = -1/2 c^T Q^{-1} c is exactly the unconstrained
        # minimum of 1/2 w^T (A^T A) w - (A^T b)^T w, so enclosure.lo + 1/2||b||^2 is a
        # rigorous lower bound on that OLS residual (hence on E_min). The box only has to
        # contain the interior point x = 0 we certify at; it does not constrain the bound.
        m_box = 1.0
        a_box = np.vstack([np.eye(n), -np.eye(n)])
        b_box = np.concatenate([m_box * np.ones(n), m_box * np.ones(n)])
        cert = certify_qp_optimum(gram, -corr, a_box, b_box, np.zeros(n), np.zeros(2 * n))
        lower = float(cert.enclosure.lo) + 0.5 * float(b_vec @ b_vec)
        return GapCertificate(
            lower_bound=lower, energy=upper, method="convex", level=0,
            certified=True, sealed=None,
        )
    except Exception:
        return GapCertificate(
            lower_bound=floor, energy=upper, method="ols_floor", level=0,
            certified=False, sealed=None,
        )


@dataclass(frozen=True)
class SparseFitResult:
    r"""The result of a certified sparse fit (Fork C).

    Attributes
    ----------
    support:
        The selected column indices (``supp(z)``), ascending.
    coefficients:
        The full-length ``(n,)`` coefficient vector, zero off the support and the OLS fit
        of the selected columns on it.
    certificate:
        The :class:`~omnibias.discrete._core.solution.GapCertificate` for the
        pseudo-Boolean **surrogate** energy ``1/2 ||A z - b||^2 + lambda 1^T z``.
    surrogate_energy:
        The decoded surrogate energy (the certificate's upper bound).
    refit_residual:
        ``1/2 ||A w - b||^2`` for the refit coefficients (the continuous data-fit term).
    note:
        Explicit scope note: the seal is on the pseudo-Boolean surrogate, not the
        continuous best-subset objective.
    """

    support: tuple[int, ...]
    coefficients: FloatArray
    certificate: GapCertificate
    surrogate_energy: float
    refit_residual: float
    note: str = (
        "The certificate seals the pseudo-Boolean surrogate energy "
        "1/2||A z - b||^2 + lambda 1^T z, not the continuous best-subset objective; "
        "the coefficients are an OLS refit on the decoded support."
    )

    @property
    def n_selected(self) -> int:
        """Number of selected columns ``|support|``."""
        return len(self.support)


def certified_sparse_fit(
    A: object,
    b: object,
    lam: float,
    *,
    relaxed: object | None = None,
    level: int = 2,
    bisection_steps: int = 24,
    n_starts: int = 16,
    seed: int = 0,
    name: str | None = None,
) -> SparseFitResult:
    r"""Certified sparse fit: seal the QUBO surrogate, refit coefficients on its support.

    Encodes the pseudo-Boolean :class:`SupportSelectionProblem` surrogate, decodes a
    support (seeding from the optional ``relaxed`` soft assignment produced by the
    ``l_p`` relaxation twins), seals *that surrogate's* optimality gap with
    :func:`omnibias.discrete.certify_gap`, and refits the continuous coefficients by OLS on
    the decoded support.

    Parameters
    ----------
    A, b:
        Design matrix ``(m, n)`` and target ``(m,)``.
    lam:
        Nonnegative cardinality penalty ``lambda``.
    relaxed:
        Optional soft assignment ``x in (0, 1)^n`` (from
        :func:`omnibias.discrete.sparse.torch.sparse_relaxation` /
        :func:`omnibias.discrete.sparse.jax.sparse_relaxation`) to seed the decoder.
    level, bisection_steps:
        Forwarded to :func:`omnibias.discrete.certify_gap` (SOS half-degree; ``level >= 2``
        represents the degree-2 surrogate energy).
    n_starts, seed:
        Forwarded to :func:`omnibias.discrete.decode`.
    name:
        Optional label for the surrogate.

    Returns
    -------
    A :class:`SparseFitResult` carrying the support, refit coefficients, and the sealed
    **surrogate** certificate.
    """
    surrogate = SupportSelectionProblem(
        A=np.asarray(A, dtype=float), b=np.asarray(b, dtype=float), lam=float(lam), name=name
    )
    assignment, energy = decode(surrogate, relaxed=relaxed, n_starts=n_starts, seed=seed)
    z = np.asarray(assignment, dtype=float)
    certificate = certify_gap(
        surrogate, z, level=level, bisection_steps=bisection_steps,
        claim_label="sparse support-selection energy",
    )
    refitter = BestSubsetProblem(
        A=surrogate.A, b=surrogate.b, lam=float(lam), name=name
    )
    coefficients, refit_residual = refitter.refit(z)
    support = tuple(int(i) for i, v in enumerate(assignment) if v)
    return SparseFitResult(
        support=support,
        coefficients=coefficients,
        certificate=certificate,
        surrogate_energy=float(energy),
        refit_residual=float(refit_residual),
    )


__all__ = ["SparseFitResult", "certified_sparse_fit", "certify_best_subset_gap"]
