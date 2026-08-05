# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Rigorous optimality-gap certificate for a decoded discrete point.

:func:`certify_gap` sandwiches the true minimum energy between

* a **rigorous lower bound** -- the Lasserre / SOS bound over the Boolean hypercube
  (:func:`omnibias.discrete._core.bound.lasserre_lower_bound`, hash-sealed and
  Lean-checkable), seeded and back-stopped by the always-valid
  :func:`omnibias.discrete._core.bound.negative_coeff_lower_bound`; and

* the **decoded point's energy** as the upper bound.

The result is a certified optimality gap ``lower <= optimum <= energy`` -- never an
exact-optimality (P = NP) claim, and honest about bound strength (a weaker bound only
widens the certified gap). Without ``omnibias-sos`` the SOS bound is unavailable and the
certificate degrades to the trivial floor (``certified=False``), or to ``method="none"``
when even the polynomial cannot be built.
"""

from __future__ import annotations

import numpy as np
from omnibias.discrete._core.bound import lasserre_lower_bound, negative_coeff_lower_bound
from omnibias.discrete._core.decode import is_binary
from omnibias.discrete._core.problem import DiscreteProblem
from omnibias.discrete._core.solution import GapCertificate


def certify_gap(
    problem: DiscreteProblem,
    x: object,
    *,
    level: int = 1,
    bisection_steps: int = 24,
    seed_lower: float | None = None,
    claim_label: str = "energy",
) -> GapCertificate:
    r"""Certify how close the binary point ``x`` is to optimal for ``problem``.

    Parameters
    ----------
    problem:
        Any :class:`~omnibias.discrete._core.problem.DiscreteProblem`.
    x:
        A binary point ``x in {0, 1}^n`` (e.g. from :func:`omnibias.discrete.decode`) --
        the upper bound.
    level:
        SOS relaxation half-degree (``1`` is the basic Boolean relaxation; higher is
        tighter and more expensive). Must be at least ``ceil(deg(E) / 2)`` to represent
        the energy.
    bisection_steps:
        Number of bisection steps used to search for the largest provable SOS bound.
    seed_lower:
        Optional lower bracket for the bisection; defaults to the always-valid
        :func:`negative_coeff_lower_bound`.
    claim_label:
        Human-readable subject of the sealed SOS claim.

    Returns
    -------
    :class:`~omnibias.discrete._core.solution.GapCertificate` with the rigorous lower
    bound, the decoded energy, and a ``sealed`` v1 certificate when an SOS bound proved.
    """
    n = problem.n
    xv = np.asarray(x, dtype=float).reshape(-1)
    if xv.shape[0] != n:
        raise ValueError(f"x must have length {n}, got {xv.shape[0]}")
    if not is_binary(xv):
        raise ValueError("x must be a binary point in {0, 1}^n (decode the relaxation first)")
    upper = float(problem.energy(xv))

    try:
        import omnibias.sos  # noqa: F401  (availability gate for the polynomial + SOS bound)
    except ImportError:
        return GapCertificate(
            lower_bound=float("-inf"), energy=upper, method="none", level=0,
            certified=False, sealed=None,
        )

    poly = problem.to_polynomial()
    floor = negative_coeff_lower_bound(poly)
    seed = floor if seed_lower is None else float(seed_lower)
    result = lasserre_lower_bound(
        problem, level=level, seed_lower=seed, upper=upper,
        steps=bisection_steps, claim_label=claim_label,
    )
    if result is None:  # sos present but nothing proved -> the valid trivial floor
        return GapCertificate(
            lower_bound=floor, energy=upper, method="negative_coeff", level=0,
            certified=False, sealed=None,
        )
    gamma, sealed = result
    return GapCertificate(
        lower_bound=gamma, energy=upper, method="sos", level=level,
        certified=True, sealed=sealed,
    )


__all__ = ["certify_gap"]
