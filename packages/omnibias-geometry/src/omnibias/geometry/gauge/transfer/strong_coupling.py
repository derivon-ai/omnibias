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
  ``counting="backtrack"`` (``C = 15``, **not** a bound on ``N_2``),
  ``counting="crude"`` (``C = 24``), and a term-by-term
  ``counting="cluster"`` (explicit keep plus geometric tail) remain
  available.  ``certified=True`` only when the enclosed contraction
  ratio is strictly less than 1.

Neither count is a formalization of Osterwalder-Seiler.  Every result is
one coupling, one spacing, one group; ``continuum_claim`` stays false.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

from omnibias.core.verified.interval import Interval
from omnibias.core.verified.series import geometric_series_closed_form, geometric_tail_enclosure
from omnibias.core.verified.transcend import besseli_iv, ln_iv

Scalar = float | int | Fraction
Counting = Literal["two_scale", "backtrack", "crude", "cluster"]

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

#: Term-by-term keep plus geometric tail (still not Osterwalder-Seiler).
CLUSTER_POLYMER_METHOD = "finite_polymer_cluster"

#: Method tag on the infinite-character Wilson gap.
WILSON_CHARACTER_METHOD = "wilson_character_bessel_ratio"

#: Locked dyadic grid ``k/32`` for ``k = 1..16``.  Not a continuum interval of ``β``.
POLYMER_BETA_GRID: tuple[Fraction, ...] = tuple(Fraction(k, 32) for k in range(1, 17))

#: Method tag for the majorant domain on :data:`POLYMER_BETA_GRID`.
POLYMER_BETA_DOMAIN_METHOD = "polymer_beta_domain_grid"

#: Wider locked grid for the infinite character-basis Wilson gap.
#: Starts at the polymer two-scale failure ``1/4`` and continues past
#: :data:`POLYMER_BETA_GRID` (which stops at ``1/2``).
WILSON_CHARACTER_BETA_GRID: tuple[Fraction, ...] = (
    Fraction(1, 4),
    Fraction(1, 2),
    Fraction(1),
    Fraction(2),
    Fraction(4),
)

#: Method tag for the Wilson-character domain on :data:`WILSON_CHARACTER_BETA_GRID`.
WILSON_CHARACTER_BETA_DOMAIN_METHOD = "wilson_character_beta_domain_grid"

#: Polymer two-scale failure used as the Wilson-domain contrast point.
WILSON_CHARACTER_CONTRAST_BETA = Fraction(1, 4)


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
    n_keep: int | None = None

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
    n_keep: int = 3,
) -> StrongCouplingGapResult:
    """Polymer lower bound on ``m a`` at one fixed ``β``.

    Default ``counting="two_scale"`` encloses ``u + A u² / (1 - B u)``
    with ``A = polymer_first_step`` and ``B = polymer_coordination_backtrack``.
    ``certified=True`` only when that enclosure's upper end is ``< 1``.
    Single-scale ``backtrack`` / ``crude`` keep ``m a ≥ -ln(C u)`` with
    ``C u.hi < 1``.  ``counting="cluster"`` keeps the first ``n_keep``
    two-scale terms explicitly and encloses the rest with
    :func:`~omnibias.core.verified.series.geometric_tail_enclosure`.
    Out of domain the leading activity is still returned and must not be
    sealed as proved.
    """
    activity = su2_wilson_activity(beta)
    if counting == "cluster":
        if n_keep < 2:
            raise ValueError(f"n_keep must be >= 2, got {n_keep}")
        first_step = polymer_first_step(spacetime_dim)
        coordination = polymer_coordination_backtrack(spacetime_dim)
        method = CLUSTER_POLYMER_METHOD
        scale_a = Interval.from_value(first_step)
        scale_b = Interval.from_value(coordination)
        ratio = scale_b * activity
        terms = [activity]
        u_pow = activity * activity
        terms.append(scale_a * u_pow)
        b_pow = Interval.point(1.0)
        for _ in range(3, n_keep + 1):
            u_pow = u_pow * activity
            b_pow = b_pow * scale_b
            terms.append(scale_a * b_pow * u_pow)
        partial = terms[0]
        for term in terms[1:]:
            partial = partial + term
        if ratio.hi >= 1.0:
            return StrongCouplingGapResult(
                beta=float(beta),
                spacetime_dim=int(spacetime_dim),
                coordination=coordination,
                first_step=first_step,
                counting=counting,
                activity=activity,
                activity_times_c=partial,
                tail_bound=None,
                spectral_gap_lower=0.0,
                subdominant_ratio_upper=float(max(partial.hi, ratio.hi)),
                in_convergence_domain=False,
                method=method,
                n_keep=int(n_keep),
            )
        tail = geometric_tail_enclosure(terms[-1], ratio)
        product = partial + tail
        in_domain = product.lo > 0.0 and product.hi < 1.0
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
                n_keep=int(n_keep),
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
            n_keep=int(n_keep),
        )
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
        in_domain = product.lo > 0.0 and product.hi < 1.0
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
            "counting must be 'two_scale', 'backtrack', 'crude', or 'cluster', "
            f"got {counting!r}"
        )
    product = Interval.from_value(coordination) * activity
    in_domain = product.lo > 0.0 and product.hi < 1.0
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


@dataclass(frozen=True)
class PolymerDomainResult:
    """Majorant domain on a locked dyadic ``β`` grid.

    ``beta_certified`` is the largest grid point whose majorant
    interval-certifies; ``beta_outside`` is the smallest strictly larger
    grid point that fails.  That is the majorant's domain on this grid,
    not a physical critical coupling, not ``a -> 0``, and not
    Osterwalder-Seiler.
    """

    counting: str
    spacetime_dim: int
    n_keep: int | None
    grid: tuple[Fraction, ...]
    beta_certified: Fraction
    beta_outside: Fraction
    certified_result: StrongCouplingGapResult
    outside_result: StrongCouplingGapResult
    method: str = POLYMER_BETA_DOMAIN_METHOD
    continuum_claim: bool = False
    yang_mills_claim: bool = False

    @property
    def certified(self) -> bool:
        return (
            self.certified_result.certified
            and not self.outside_result.certified
            and self.beta_certified < self.beta_outside
            and self.continuum_claim is False
            and self.yang_mills_claim is False
        )


def certified_polymer_beta_domain(
    *,
    counting: Counting = "two_scale",
    spacetime_dim: int = 4,
    n_keep: int = 3,
    grid: Sequence[Fraction] | None = None,
) -> PolymerDomainResult:
    """Largest certifying dyadic ``β`` and the next grid failure.

    Evaluates :func:`certified_strong_coupling_glueball_bound` on a locked
    dyadic grid.  The claim is only about those points.  Bessel
    monotonicity is not promoted to a continuum interval of ``β``.
    """
    points = tuple(Fraction(item) for item in (grid if grid is not None else POLYMER_BETA_GRID))
    if len(points) < 2:
        raise ValueError("grid must contain at least two positive betas")
    if any(item <= 0 for item in points):
        raise ValueError("grid betas must be > 0")
    ordered = tuple(sorted(points))
    if len(set(ordered)) != len(ordered):
        raise ValueError("grid betas must be unique")
    kwargs: dict[str, object] = {"spacetime_dim": spacetime_dim, "counting": counting}
    keep: int | None = None
    if counting == "cluster":
        if n_keep < 2:
            raise ValueError(f"n_keep must be >= 2, got {n_keep}")
        kwargs["n_keep"] = int(n_keep)
        keep = int(n_keep)
    evaluated: list[tuple[Fraction, StrongCouplingGapResult]] = []
    for beta in ordered:
        evaluated.append(
            (beta, certified_strong_coupling_glueball_bound(beta, **kwargs))  # type: ignore[arg-type]
        )
    certified_points = [item for item in evaluated if item[1].certified]
    if not certified_points:
        raise ValueError("no grid point interval-certifies the majorant")
    beta_star, star_result = max(certified_points, key=lambda item: item[0])
    failures = [item for item in evaluated if item[0] > beta_star and not item[1].certified]
    if not failures:
        raise ValueError(
            "no larger grid point fails the majorant; cannot record a first failure"
        )
    beta_out, out_result = min(failures, key=lambda item: item[0])
    return PolymerDomainResult(
        counting=counting,
        spacetime_dim=int(spacetime_dim),
        n_keep=keep,
        grid=ordered,
        beta_certified=beta_star,
        beta_outside=beta_out,
        certified_result=star_result,
        outside_result=out_result,
    )


@dataclass(frozen=True)
class WilsonCharacterDomainResult:
    """Wilson-character gap on a locked ``β`` grid wider than the polymer cutoff.

    ``beta_certified`` is the largest certifying grid point.
    ``beta_outside`` is ``None`` and ``grid_exhausted`` is ``True`` when
    every grid point certifies (the typical case: ``I₂/I₁ < 1`` at every
    finite ``β`` with a tight Bessel enclosure).  If a large-``β``
    enclosure ever has ``activity.hi >= 1``, that failure is recorded
    the same way as the polymer domain.

    This is the 0+1-D infinite character transfer on this grid.  It is
    not 4-D Yang-Mills and not a physical critical coupling.
    """

    grid: tuple[Fraction, ...]
    beta_certified: Fraction
    beta_outside: Fraction | None
    grid_exhausted: bool
    quarter_certified: bool
    certified_result: WilsonCharacterGapResult
    outside_result: WilsonCharacterGapResult | None
    method: str = WILSON_CHARACTER_BETA_DOMAIN_METHOD
    continuum_claim: bool = False
    yang_mills_claim: bool = False

    @property
    def certified(self) -> bool:
        exhausted_ok = self.grid_exhausted and self.beta_outside is None
        failure_ok = (
            not self.grid_exhausted
            and self.beta_outside is not None
            and self.outside_result is not None
            and not self.outside_result.certified
            and self.beta_certified < self.beta_outside
        )
        return (
            self.certified_result.certified
            and self.quarter_certified
            and self.beta_certified > WILSON_CHARACTER_CONTRAST_BETA
            and (exhausted_ok or failure_ok)
            and self.continuum_claim is False
            and self.yang_mills_claim is False
        )


def certified_wilson_character_beta_domain(
    *,
    grid: Sequence[Fraction] | None = None,
) -> WilsonCharacterDomainResult:
    """Largest certifying Wilson-character ``β`` on a grid past the polymer cutoff.

    The locked grid includes ``1/4``, where the 4-D polymer two-scale
    majorant already fails, and continues to ``4``.  ``certified=True``
    only when Wilson certifies at ``1/4`` and at least one strictly
    larger grid point.  Bessel monotonicity is not promoted to a
    continuum interval of ``β``.
    """
    points = tuple(
        Fraction(item) for item in (grid if grid is not None else WILSON_CHARACTER_BETA_GRID)
    )
    if len(points) < 2:
        raise ValueError("grid must contain at least two positive betas")
    if any(item <= 0 for item in points):
        raise ValueError("grid betas must be > 0")
    ordered = tuple(sorted(points))
    if len(set(ordered)) != len(ordered):
        raise ValueError("grid betas must be unique")
    if WILSON_CHARACTER_CONTRAST_BETA not in ordered:
        raise ValueError(
            "grid must include 1/4 (the polymer two-scale failure used as contrast)"
        )
    evaluated: list[tuple[Fraction, WilsonCharacterGapResult]] = []
    for beta in ordered:
        evaluated.append((beta, certified_wilson_character_gap(beta)))
    certified_points = [item for item in evaluated if item[1].certified]
    if not certified_points:
        raise ValueError("no grid point interval-certifies the Wilson character gap")
    beta_star, star_result = max(certified_points, key=lambda item: item[0])
    failures = [item for item in evaluated if item[0] > beta_star and not item[1].certified]
    if failures:
        beta_out, out_result = min(failures, key=lambda item: item[0])
        exhausted = False
    else:
        beta_out, out_result = None, None
        exhausted = True
    quarter_certified = any(
        item[0] == WILSON_CHARACTER_CONTRAST_BETA and item[1].certified for item in evaluated
    )
    return WilsonCharacterDomainResult(
        grid=ordered,
        beta_certified=beta_star,
        beta_outside=beta_out,
        grid_exhausted=exhausted,
        quarter_certified=quarter_certified,
        certified_result=star_result,
        outside_result=out_result,
    )


__all__ = [
    "BACKTRACK_POLYMER_METHOD",
    "BETA_LOCK",
    "BETA_LOCK_CRUDE",
    "CLUSTER_POLYMER_METHOD",
    "CRUDE_POLYMER_METHOD",
    "POLYMER_BETA_DOMAIN_METHOD",
    "POLYMER_BETA_GRID",
    "POLYMER_METHOD",
    "WILSON_CHARACTER_BETA_DOMAIN_METHOD",
    "WILSON_CHARACTER_BETA_GRID",
    "WILSON_CHARACTER_CONTRAST_BETA",
    "WILSON_CHARACTER_METHOD",
    "PolymerDomainResult",
    "StrongCouplingGapResult",
    "WilsonCharacterDomainResult",
    "WilsonCharacterGapResult",
    "certified_polymer_beta_domain",
    "certified_strong_coupling_glueball_bound",
    "certified_wilson_character_beta_domain",
    "certified_wilson_character_gap",
    "polymer_coordination",
    "polymer_coordination_backtrack",
    "polymer_first_step",
    "su2_wilson_activity",
]
