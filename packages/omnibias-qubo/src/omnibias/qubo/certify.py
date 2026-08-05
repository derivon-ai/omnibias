# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Rigorous optimality-gap certificate for a decoded QUBO point.

:func:`certify_qubo_gap` sandwiches the true minimum energy between

* a **rigorous lower bound** -- either the Lasserre / SOS bound over the Boolean
  hypercube (``kind="sos"``; :func:`omnibias.qubo._core.bound.lasserre_lower_bound`,
  hash-sealed and Lean-checkable) or the cheap spectral / box-QP bound
  (``kind="spectral"``; interval-sealed by :func:`omnibias.convex.certify_qp_optimum`);

* the **decoded point's energy** as the upper bound.

The result is a certified optimality gap ``lower <= optimum <= energy`` -- never an
exact-optimality (P = NP) claim, and honest about bound strength (a weaker bound only
widens the certified gap). ``kind="sos"`` falls back to the spectral bound when
``omnibias-sos`` is unavailable or proves nothing; the spectral bound falls back to a
valid float value (``certified=False``) without ``omnibias-convex``.
"""

from __future__ import annotations

import numpy as np
from omnibias.qubo._core.bound import lasserre_lower_bound, spectral_lower_bound
from omnibias.qubo._core.decode import is_binary
from omnibias.qubo.problem import QUBOCertificate, QUBOProblem


def certify_qubo_gap(
    problem: QUBOProblem,
    x: object,
    *,
    kind: str = "sos",
    level: int = 1,
    bisection_steps: int = 24,
) -> QUBOCertificate:
    r"""Certify how close the binary point ``x`` is to optimal for ``problem``.

    Parameters
    ----------
    problem:
        The QUBO instance (its ``Q`` / ``c`` / ``const`` define the energy).
    x:
        A binary point ``x in {0, 1}^n`` (e.g. from :func:`omnibias.qubo.decode_qubo`) --
        the upper bound.
    kind:
        Lower-bound strength: ``"sos"`` (Lasserre / Positivstellensatz, certified and
        sealed, small / moderate ``n``) or ``"spectral"`` (eigenvalue-shift / box-QP,
        any ``n``).
    level:
        SOS relaxation half-degree for ``kind="sos"`` (``1`` is the basic Boolean
        relaxation; higher is tighter and more expensive).
    bisection_steps:
        Number of bisection steps used to search for the largest provable SOS bound.

    Returns
    -------
    :class:`~omnibias.qubo.problem.QUBOCertificate` with the rigorous lower bound, the
    decoded energy, and a ``sealed`` v1 certificate when an SOS bound was proved.
    """
    n = problem.n
    xv = np.asarray(x, dtype=float).reshape(-1)
    if xv.shape[0] != n:
        raise ValueError(f"x must have length {n}, got {xv.shape[0]}")
    if not is_binary(xv):
        raise ValueError("x must be a binary point in {0, 1}^n (decode the relaxation first)")
    upper = float(problem.energy(xv))

    if kind == "spectral":
        lower, certified = spectral_lower_bound(problem)
        return QUBOCertificate(
            lower_bound=lower, energy=upper, method="spectral", level=0,
            certified=certified, sealed=None,
        )
    if kind == "sos":
        spec_lower, spec_certified = spectral_lower_bound(problem)
        result = lasserre_lower_bound(
            problem, level=level, seed_lower=spec_lower, upper=upper,
            steps=bisection_steps, claim_label="QUBO energy",
        )
        if result is None:  # sos unavailable or nothing proved -> spectral fallback
            return QUBOCertificate(
                lower_bound=spec_lower, energy=upper, method="spectral", level=0,
                certified=spec_certified, sealed=None,
            )
        gamma, sealed = result
        return QUBOCertificate(
            lower_bound=gamma, energy=upper, method="sos", level=level,
            certified=True, sealed=sealed,
        )
    raise ValueError(f"unknown kind {kind!r}; choose 'sos' or 'spectral'")


__all__ = ["certify_qubo_gap"]
