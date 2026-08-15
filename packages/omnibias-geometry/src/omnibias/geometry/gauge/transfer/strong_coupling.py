# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Crude strong-coupling polymer bound for 4-D SU(2) Wilson at fixed ``β``.

Two finite objects, neither of which is continuum Yang-Mills:

* :func:`certified_wilson_character_gap` encloses ``-ln(I₂(β)/I₁(β))``, the
  exact lattice-unit gap of the **infinite** character-basis SU(2) Wilson
  transfer (today :func:`~.matrices.su2_wilson_transfer` is a hard
  ``n_modes`` truncation of that series).
* :func:`certified_strong_coupling_glueball_bound` turns the same activity
  ``u(β) = I₂(β)/I₁(β)`` into a **self-contained** lower bound
  ``m a ≥ -ln(C u)`` on a 4-D lattice glueball mass, using a locked
  plaquette-surface counting majorant ``C = 8(d-1)``.  ``certified=True``
  only when the interval-certified product ``C u`` is strictly less than 1.

The counting constant is derived in :func:`polymer_coordination` and is
deliberately crude (it overcounts backtracking).  The method tag is
``crude_polymer_count``, not a formalization of Osterwalder-Seiler.  Every
result is one coupling, one spacing, one group; ``continuum_claim`` stays
false.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from omnibias.core.verified.interval import Interval
from omnibias.core.verified.series import geometric_tail_enclosure
from omnibias.core.verified.transcend import besseli_iv, ln_iv

Scalar = float | int | Fraction

#: Locked test coupling that interval-certifies ``C u < 1`` for ``d = 4``.
BETA_LOCK = Fraction(1, 8)

#: Method tag on a certified polymer bound (not a literature theorem name).
POLYMER_METHOD = "crude_polymer_count"

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
    """Crude polymer lower bound on a 4-D SU(2) Wilson glueball mass at one ``β``."""

    beta: float
    spacetime_dim: int
    coordination: int
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
) -> StrongCouplingGapResult:
    """Crude polymer lower bound ``m a ≥ -ln(C u(β))`` at one fixed ``β``.

    ``certified=True`` only when ``C u.hi < 1``.  Out of domain the leading
    activity is still returned and ``certified`` stays ``False`` -- do not
    seal that result as proved.
    """
    coordination = polymer_coordination(spacetime_dim)
    activity = su2_wilson_activity(beta)
    product = Interval.from_value(coordination) * activity
    in_domain = product.hi < 1.0
    tail = geometric_tail_enclosure(product, product) if in_domain else None
    if not in_domain:
        return StrongCouplingGapResult(
            beta=float(beta),
            spacetime_dim=int(spacetime_dim),
            coordination=coordination,
            activity=activity,
            activity_times_c=product,
            tail_bound=None,
            spectral_gap_lower=0.0,
            subdominant_ratio_upper=float(product.hi),
            in_convergence_domain=False,
        )
    gap = -ln_iv(product)
    return StrongCouplingGapResult(
        beta=float(beta),
        spacetime_dim=int(spacetime_dim),
        coordination=coordination,
        activity=activity,
        activity_times_c=product,
        tail_bound=tail,
        spectral_gap_lower=float(gap.lo),
        subdominant_ratio_upper=float(product.hi),
        in_convergence_domain=True,
    )


__all__ = [
    "BETA_LOCK",
    "POLYMER_METHOD",
    "WILSON_CHARACTER_METHOD",
    "StrongCouplingGapResult",
    "WilsonCharacterGapResult",
    "certified_strong_coupling_glueball_bound",
    "certified_wilson_character_gap",
    "polymer_coordination",
    "su2_wilson_activity",
]
