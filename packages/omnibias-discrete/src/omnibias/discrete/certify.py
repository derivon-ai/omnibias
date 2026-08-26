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

from collections.abc import Sequence
from dataclasses import dataclass

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


@dataclass(frozen=True)
class TightenedGap:
    """Best sound Lasserre sandwich among increasing relaxation levels.

    ``tight`` is always ``False``. A smaller gap is a better sound floor,
    never a P = NP claim.
    """

    certificate: GapCertificate
    level_used: int
    levels_tried: tuple[int, ...]
    tight: bool = False

    @property
    def gap(self) -> float:
        return self.certificate.absolute_gap

    def honesty(self) -> dict[str, bool]:
        return {
            "p_vs_np_claim": False,
            "tight": False,
        }


def tighten_gap(
    problem: DiscreteProblem,
    x: object,
    *,
    levels: Sequence[int] = (1, 2),
    bisection_steps: int = 24,
    seed_lower: float | None = None,
    claim_label: str = "energy",
) -> TightenedGap:
    """Run increasing Lasserre levels and keep the best sound lower bound.

    A higher level that fails or is not sound is skipped. Tightness is
    never claimed.
    """
    if not levels:
        raise ValueError("levels must be non-empty")
    ordered = tuple(int(level) for level in levels)
    if any(level < 0 for level in ordered):
        raise ValueError("levels must be >= 0")
    best: GapCertificate | None = None
    used = ordered[0]
    fallback: GapCertificate | None = None
    for level in ordered:
        nxt = certify_gap(
            problem,
            x,
            level=level,
            bisection_steps=bisection_steps,
            seed_lower=seed_lower,
            claim_label=claim_label,
        )
        if fallback is None:
            fallback = nxt
        if not nxt.is_sound:
            continue
        if best is None or nxt.lower_bound + 1e-12 >= best.lower_bound:
            best = nxt
            used = level
    if best is None:
        best = fallback
    return TightenedGap(
        certificate=best,
        level_used=used,
        levels_tried=ordered,
        tight=False,
    )


__all__ = ["TightenedGap", "certify_gap", "tighten_gap"]
