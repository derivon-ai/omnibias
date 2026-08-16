# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Strong-coupling polymer bound for 4-D SU(2) Wilson at fixed ``β``.

Two finite objects, neither of which is continuum Yang-Mills:

* :func:`certified_wilson_character_gap` encloses ``-ln(I₂(β)/I₁(β))``, the
  exact lattice-unit gap of the **infinite** character-basis SU(2) Wilson
  transfer (today :func:`~.matrices.su2_wilson_transfer` is a hard
  ``n_modes`` truncation of that series).
* :func:`certified_strong_coupling_glueball_bound` turns the same activity
  ``u(β) = I₂(β)/I₁(β)`` into a **self-contained** lower bound on a 4-D
  lattice glueball mass.  The default is the two-scale polymer remainder
  ``Σ N_n u^n ≤ u + A u² / (1 - B u)`` with first-step ``A = 4(2d-3)``
  (20 in four dimensions) and subsequent branching
  ``B = 3(2d-3)`` (15 in four dimensions).  Single-scale majorants
  ``counting="backtrack"`` (``C = 15``, **not** a bound on ``N_2``) and
  ``counting="crude"`` (``C = 24``) remain available.  ``certified=True``
  only when the enclosed contraction ratio is strictly less than 1.

Neither count is a formalization of Osterwalder-Seiler.  Every result is
one coupling, one spacing, one group; ``continuum_claim`` stays false.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

from omnibias.core.verified.interval import Interval
from omnibias.core.verified.series import geometric_series_closed_form, geometric_tail_enclosure
from omnibias.core.verified.transcend import besseli_iv, ln_iv

Scalar = float | int | Fraction
Counting = Literal["two_scale", "backtrack", "crude"]

#: Locked test coupling that interval-certifies the two-scale remainder, ``d=4``.
BETA_LOCK = Fraction(1, 5)

#: Locked coupling that interval-certifies the cruder ``C = 24`` majorant.
BETA_LOCK_CRUDE = Fraction(1, 8)

#: Default method tag (not a literature theorem name).
POLYMER_METHOD = "two_scale_polymer_count"

#: Single-scale backtrack majorant (not a bound on ``N_2``).
BACKTRACK_POLYMER_METHOD = "backtrack_polymer_count"

#: Older overcounting majorant.
CRUDE_POLYMER_METHOD = "crude_polymer_count"

#: Method tag on the infinite-character Wilson gap.
WILSON_CHARACTER_METHOD = "wilson_character_bessel_ratio"


def polymer_coordination(spacetime_dim: int) -> int:
    """Locked surface-counting majorant ``C = 8(d - 1)``.

    Derivation (crude, documented, not sharp).  A plaquette has 4 edges.
    In ``d`` dimensions each edge is shared by ``2(d - 1)`` plaquettes.
    A branching majorant that *includes* the current plaquette (so it
    strictly overcounts trees) is ``4 · 2(d - 1) = 8(d - 1)``.  For
    ``d = 4`` this is ``C = 24``.
    """
    if spacetime_dim < 2:
        raise ValueError(f"spacetime_dim must be >= 2, got {spacetime_dim}")
    return 8 * (int(spacetime_dim) - 1)


def polymer_coordination_backtrack(spacetime_dim: int) -> int:
    """Backtrack-excluding tree majorant ``C = 3(2d - 3)``.

    A connected n-plaquette surface through a fixed plaquette is majorized
    by rooted trees (trees overcount surfaces that have cycles).  Each new
    plaquette attaches along one edge and has 3 free edges; each free edge
    has at most ``2(d-1)-1 = 2d-3`` other plaquettes (no immediate
    backtrack through the attachment).  Hence ``N_n ≤ C^{n-1}`` with
    ``C = 3(2d-3)``.  For ``d = 4`` this is ``C = 15``.

    Still a counting majorant, not Osterwalder-Seiler.  This is the
    **subsequent** branching ``B``, not a bound on ``N_2``.
    """
    if spacetime_dim < 2:
        raise ValueError(f"spacetime_dim must be >= 2, got {spacetime_dim}")
    return 3 * (2 * int(spacetime_dim) - 3)


def polymer_first_step(spacetime_dim: int) -> int:
    """First-attachment majorant ``A = 4(2d - 3)``.

    A fixed plaquette has 4 edges; each edge has at most ``2d-3`` other
    plaquettes (no immediate backtrack).  Hence ``N_2 ≤ A`` with
    ``A = 4(2d-3)``.  For ``d = 4`` this is ``A = 20``.  Later
    generations have 3 free edges, so they use
    :func:`polymer_coordination_backtrack`.
    """
    if spacetime_dim < 2:
        raise ValueError(f"spacetime_dim must be >= 2, got {spacetime_dim}")
    return 4 * (2 * int(spacetime_dim) - 3)


def su2_wilson_activity(beta: Scalar) -> Interval:
    """Enclosure of the SU(2) Wilson fundamental activity ``u = I₂(β)/I₁(β)``."""
    argument = Interval.from_value(beta)
    if argument.lo <= 0.0:
        raise ValueError(f"beta must be > 0, got {beta!r}")
    i1 = besseli_iv(1, argument)
    i2 = besseli_iv(2, argument)
    if i1.lo <= 0.0:
        raise ValueError(f"I1(beta) is not certifiably positive at beta={beta!r}")
    return i2 / i1


@dataclass(frozen=True)
class WilsonCharacterGapResult:
    """Certified ``-ln(I₂/I₁)`` of the infinite character-basis Wilson transfer."""

    beta: float
    activity: Interval
    spectral_gap_lower: float
    subdominant_ratio_upper: float
    method: str = WILSON_CHARACTER_METHOD

    @property
    def certified(self) -> bool:
        return self.spectral_gap_lower > 0.0 and self.subdominant_ratio_upper < 1.0


@dataclass(frozen=True)
class StrongCouplingGapResult:
    """Polymer lower bound on a 4-D SU(2) Wilson glueball mass at one ``β``."""

    beta: float
    spacetime_dim: int
    coordination: int
    first_step: int | None
    counting: str
    activity: Interval
    activity_times_c: Interval
    tail_bound: Interval | None
    spectral_gap_lower: float
    subdominant_ratio_upper: float
    in_convergence_domain: bool
    method: str = POLYMER_METHOD

    @property
    def certified(self) -> bool:
        return (
            self.in_convergence_domain
            and self.spectral_gap_lower > 0.0
            and self.subdominant_ratio_upper < 1.0
        )


def certified_wilson_character_gap(beta: Scalar) -> WilsonCharacterGapResult:
    """Enclose ``-ln(I₂(β)/I₁(β))`` for the infinite SU(2) character transfer.

    This is a 0+1-D character-basis statement, not a 4-D lattice bound and
    not a continuum claim.
    """
    activity = su2_wilson_activity(beta)
    if activity.hi >= 1.0:
        return WilsonCharacterGapResult(
            beta=float(beta),
            activity=activity,
            spectral_gap_lower=0.0,
            subdominant_ratio_upper=float(activity.hi),
        )
    gap = -ln_iv(activity)
    return WilsonCharacterGapResult(
        beta=float(beta),
        activity=activity,
        spectral_gap_lower=float(gap.lo),
        subdominant_ratio_upper=float(activity.hi),
    )


def certified_strong_coupling_glueball_bound(
    beta: Scalar,
    *,
    spacetime_dim: int = 4,
    counting: Counting = "two_scale",
) -> StrongCouplingGapResult:
    """Polymer lower bound on ``m a`` at one fixed ``β``.

    Default ``counting="two_scale"`` encloses ``u + A u² / (1 - B u)``
    with ``A = polymer_first_step`` and ``B = polymer_coordination_backtrack``.
    ``certified=True`` only when that enclosure's upper end is ``< 1``.
    Single-scale ``backtrack`` / ``crude`` keep ``m a ≥ -ln(C u)`` with
    ``C u.hi < 1``.  Out of domain the leading activity is still returned
    and must not be sealed as proved.
    """
    activity = su2_wilson_activity(beta)
    if counting == "two_scale":
        first_step = polymer_first_step(spacetime_dim)
        coordination = polymer_coordination_backtrack(spacetime_dim)
        method = POLYMER_METHOD
        ratio = Interval.from_value(coordination) * activity
        if ratio.hi >= 1.0:
            product = activity + Interval.from_value(first_step) * activity * activity
            return StrongCouplingGapResult(
                beta=float(beta),
                spacetime_dim=int(spacetime_dim),
                coordination=coordination,
                first_step=first_step,
                counting=counting,
                activity=activity,
                activity_times_c=product,
                tail_bound=None,
                spectral_gap_lower=0.0,
                subdominant_ratio_upper=float(max(product.hi, ratio.hi)),
                in_convergence_domain=False,
                method=method,
            )
        rest = geometric_series_closed_form(
            Interval.from_value(first_step) * activity * activity, ratio
        )
        product = activity + rest
        tail = geometric_tail_enclosure(
            Interval.from_value(first_step) * activity * activity, ratio
        )
        in_domain = product.hi < 1.0
        if not in_domain:
            return StrongCouplingGapResult(
                beta=float(beta),
                spacetime_dim=int(spacetime_dim),
                coordination=coordination,
                first_step=first_step,
                counting=counting,
                activity=activity,
                activity_times_c=product,
                tail_bound=tail,
                spectral_gap_lower=0.0,
                subdominant_ratio_upper=float(product.hi),
                in_convergence_domain=False,
                method=method,
            )
        gap = -ln_iv(product)
        return StrongCouplingGapResult(
            beta=float(beta),
            spacetime_dim=int(spacetime_dim),
            coordination=coordination,
            first_step=first_step,
            counting=counting,
            activity=activity,
            activity_times_c=product,
            tail_bound=tail,
            spectral_gap_lower=float(gap.lo),
            subdominant_ratio_upper=float(product.hi),
            in_convergence_domain=True,
            method=method,
        )
    if counting == "backtrack":
        coordination = polymer_coordination_backtrack(spacetime_dim)
        method = BACKTRACK_POLYMER_METHOD
        first_step = None
    elif counting == "crude":
        coordination = polymer_coordination(spacetime_dim)
        method = CRUDE_POLYMER_METHOD
        first_step = None
    else:
        raise ValueError(
            f"counting must be 'two_scale', 'backtrack', or 'crude', got {counting!r}"
        )
    product = Interval.from_value(coordination) * activity
    in_domain = product.hi < 1.0
    tail = geometric_tail_enclosure(product, product) if in_domain else None
    if not in_domain:
        return StrongCouplingGapResult(
            beta=float(beta),
            spacetime_dim=int(spacetime_dim),
            coordination=coordination,
            first_step=first_step,
            counting=counting,
            activity=activity,
            activity_times_c=product,
            tail_bound=None,
            spectral_gap_lower=0.0,
            subdominant_ratio_upper=float(product.hi),
            in_convergence_domain=False,
            method=method,
        )
    gap = -ln_iv(product)
    return StrongCouplingGapResult(
        beta=float(beta),
        spacetime_dim=int(spacetime_dim),
        coordination=coordination,
        first_step=first_step,
        counting=counting,
        activity=activity,
        activity_times_c=product,
        tail_bound=tail,
        spectral_gap_lower=float(gap.lo),
        subdominant_ratio_upper=float(product.hi),
        in_convergence_domain=True,
        method=method,
    )


__all__ = [
    "BACKTRACK_POLYMER_METHOD",
    "BETA_LOCK",
    "BETA_LOCK_CRUDE",
    "CRUDE_POLYMER_METHOD",
    "POLYMER_METHOD",
    "WILSON_CHARACTER_METHOD",
    "StrongCouplingGapResult",
    "WilsonCharacterGapResult",
    "certified_strong_coupling_glueball_bound",
    "certified_wilson_character_gap",
    "polymer_coordination",
    "polymer_coordination_backtrack",
    "polymer_first_step",
    "su2_wilson_activity",
]
