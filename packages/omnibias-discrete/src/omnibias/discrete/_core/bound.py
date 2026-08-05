# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Rigorous lower bounds on the minimum energy (the certificate's lower side).

Two backend-agnostic bounds, both **sound** -- they never exceed the true minimum over
the Boolean hypercube:

* :func:`negative_coeff_lower_bound` (any polynomial, trivial): every monomial
  ``x^alpha in [0, 1]`` on the cube, so
  ``E(x) = sum_alpha c_alpha x^alpha >= sum_{alpha : c_alpha < 0} c_alpha``. A cheap,
  always-valid floor that seeds the SOS bisection (and is the fallback when
  ``omnibias-sos`` is absent).

* :func:`lasserre_lower_bound` (small / moderate ``n``, headline): the Lasserre /
  moment-SOS bound. Bisect ``gamma`` and certify ``E - gamma >= 0`` on the Boolean
  hypercube with :func:`omnibias.sos.certify_nonneg_on_set` (Putinar Positivstellensatz);
  the largest proved ``gamma`` is a rigorous, rational lower bound, hash-sealed by
  :func:`omnibias.sos.seal_positivstellensatz_certificate`.

:func:`gershgorin_min_eig_lower` is a small linear-algebra helper reused by quadratic
consumers (e.g. QUBO's spectral bound) for a positive-definite diagonal shift.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from omnibias.discrete._core.problem import DiscreteProblem, boolean_constraints

if TYPE_CHECKING:  # pragma: no cover - typing only
    from omnibias.core.proof.certificate import Cert
    from omnibias.sos import Polynomial

FloatArray = NDArray[np.float64]


def gershgorin_min_eig_lower(matrix: FloatArray) -> float:
    r"""A lower bound on the smallest eigenvalue of a symmetric matrix (Gershgorin)."""
    m = np.asarray(matrix, dtype=float)
    diag = np.diag(m)
    off = np.sum(np.abs(m), axis=1) - np.abs(diag)
    return float(np.min(diag - off))


def negative_coeff_lower_bound(poly: Polynomial) -> float:
    r"""The sum of the polynomial's negative coefficients -- a valid floor on ``{0,1}^n``.

    On the cube every monomial ``x^alpha in {0, 1} subseteq [0, 1]``, so the positive
    terms contribute ``>= 0`` and each negative term ``c_alpha x^alpha >= c_alpha``;
    hence ``E(x) >= sum_{c_alpha < 0} c_alpha`` for all ``x in {0, 1}^n``. Cheap and
    always sound (not interval-sealed, so certificates label it ``certified=False``).
    """
    return float(sum(c for c in poly.coeffs.values() if c < 0.0))


def lasserre_lower_bound(
    problem: DiscreteProblem,
    *,
    level: int,
    seed_lower: float,
    upper: float,
    steps: int = 24,
    claim_label: str = "energy",
) -> tuple[float, Cert] | None:
    r"""The certified Lasserre / SOS lower bound on the minimum energy, or ``None``.

    Bisects ``gamma in [seed_lower, upper]`` (both valid brackets around the SOS bound),
    certifying ``E - gamma >= 0`` on the Boolean hypercube at each step; returns the
    largest proved ``gamma`` with a hash-sealed v1 certificate, or ``None`` if
    ``omnibias-sos`` is unavailable or no ``gamma`` could be proved. ``claim_label`` is
    the human-readable subject of the sealed claim (e.g. ``"QUBO energy"``).
    """
    try:
        from omnibias.sos import (
            certify_nonneg_on_set,
            seal_positivstellensatz_certificate,
        )
    except ImportError:
        return None

    energy_poly = problem.to_polynomial()
    cons = boolean_constraints(problem.n)

    lo, hi = float(seed_lower), float(upper)
    best_gamma: float | None = None
    best_cert = None
    for _ in range(max(steps, 1)):
        mid = 0.5 * (lo + hi)
        certificate = certify_nonneg_on_set(energy_poly - mid, cons, half_degree=level)
        if certificate.certified:
            best_gamma, best_cert = mid, certificate
            lo = mid
        else:
            hi = mid
    if best_gamma is None or best_cert is None:
        return None

    sealed = seal_positivstellensatz_certificate(
        best_cert,
        claim=(
            f"{claim_label} >= {best_gamma} for all x in {{0,1}}^{problem.n} "
            f"(Lasserre / Putinar level {level})"
        ),
    )
    return best_gamma, sealed


__all__ = [
    "gershgorin_min_eig_lower",
    "lasserre_lower_bound",
    "negative_coeff_lower_bound",
]
