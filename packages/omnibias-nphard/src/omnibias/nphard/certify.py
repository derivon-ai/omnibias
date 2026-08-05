# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Rigorous -- but honestly non-tight -- optimality-gap certificate for an NP-hard family.

:func:`certify_gap` sandwiches the true minimum between a **rigorous lower bound** and the
**decoded point's energy** (the upper bound). Three lower bounds, honest about their
strength (a weaker bound only widens the certified gap; it is never asserted zero):

* ``kind="sos"`` -- Lasserre / Positivstellensatz over the Boolean hypercube (hash-sealed
  and Lean-checkable via :mod:`omnibias.sos`); tightest, but only tractable for small ``n``;
* ``kind="spectral"`` -- eigenvalue-shift / box-QP bound (interval-sealed via
  :mod:`omnibias.convex`); any ``n``, but loose;
* ``kind="glb"`` -- the QAP-specific **Gilmore-Lawler** bound
  (:func:`omnibias.nphard.gilmore_lawler_bound`); ``O(dim^3)``, sound, and the only bound
  that stays *non-trivial at realistic block counts* (``dim ~ 12-25``) where SOS is
  intractable and the spectral bound is useless. QAP-only.

Because these families are **NP-hard**, the gap is **generally non-zero** -- unlike the
P-class :mod:`omnibias.combinatorics` (integral polytope -> tight gap), it is never asserted
zero. The QUBO bounds degrade honestly: ``kind="sos"`` falls back to spectral without
``omnibias-sos``, and the spectral bound to a valid unsealed float (``certified=False``)
without ``omnibias-convex``.
"""

from __future__ import annotations

import numpy as np
from omnibias.nphard._core.bound import gilmore_lawler_bound
from omnibias.nphard._core.decode import Problem
from omnibias.nphard._core.qap import QAPProblem
from omnibias.nphard.problem import NPHardCertificate
from omnibias.qubo import certify_qubo_gap


def _is_permutation_matrix(x: object, dim: int) -> bool:
    """Whether ``x`` (flat ``dim^2`` or ``dim x dim``) is a 0/1 permutation matrix."""
    xv = np.asarray(x, dtype=float).reshape(-1)
    if xv.shape[0] != dim * dim:
        return False
    if not np.all((xv == 0.0) | (xv == 1.0)):
        return False
    mat = xv.reshape(dim, dim)
    return bool(np.all(mat.sum(axis=0) == 1.0) and np.all(mat.sum(axis=1) == 1.0))


def certify_gap(
    problem: Problem,
    x: object,
    *,
    kind: str = "sos",
    level: int = 1,
    bisection_steps: int = 24,
) -> NPHardCertificate:
    r"""Certify how close the point ``x`` is to optimal for an NP-hard ``problem``.

    Parameters
    ----------
    problem:
        A :class:`~omnibias.nphard.QAPProblem`, :class:`~omnibias.nphard.GAPProblem` or
        :class:`~omnibias.nphard.SchedulingProblem`; its ``to_qubo()`` defines the energy.
    x:
        A binary point ``x in {0, 1}^n`` in the QUBO variable space (e.g. from
        :func:`omnibias.nphard.decode`) -- the certified *upper* bound. For ``kind="glb"``
        it must be a valid permutation matrix (a feasible placement).
    kind:
        Lower-bound strength: ``"sos"`` (Lasserre / Positivstellensatz, sealed),
        ``"spectral"`` (eigenvalue-shift / box-QP), or ``"glb"`` (QAP-only Gilmore-Lawler,
        sound and scalable to realistic block counts -- see the module docstring).
    level:
        SOS relaxation half-degree for ``kind="sos"`` (higher is tighter, costlier).
    bisection_steps:
        Bisection steps used to search for the largest provable SOS bound.

    Returns
    -------
    :class:`~omnibias.nphard.NPHardCertificate` (the discrete substrate's gap-shaped
    ``GapCertificate``): ``lower_bound <= optimum <= energy``, with a generally **non-zero**
    (never asserted zero) certified gap. For ``kind="glb"`` the certificate is on the *pure*
    QAP permutation objective (the total wirelength), the natural placement quantity.
    """
    if kind == "glb":
        if not isinstance(problem, QAPProblem):
            raise TypeError(
                "kind='glb' (Gilmore-Lawler) is QAP-specific; use 'sos' or 'spectral' for "
                f"{type(problem).__name__}"
            )
        if not _is_permutation_matrix(x, problem.dim):
            raise ValueError(
                "kind='glb' certifies the QAP permutation objective, so x must be a valid "
                "permutation matrix (a feasible placement); decode the relaxation first"
            )
        lower, sound = gilmore_lawler_bound(problem)
        upper = float(problem.objective(x))
        return NPHardCertificate(
            lower_bound=lower,
            energy=upper,
            method="gilmore_lawler",
            level=0,
            certified=sound,
            sealed=None,
        )
    return certify_qubo_gap(
        problem.to_qubo(), x, kind=kind, level=level, bisection_steps=bisection_steps
    )


__all__ = ["certify_gap"]
