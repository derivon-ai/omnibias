# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Weighted-section obstruction for the Hilbert-XVI G1 passage.

The separation-scale section

``h = eps**3 * sep**2 * eta0``

has a positive intrinsic event speed while ``sep > 0``.  That fact does not
transport the physical return family through either singular overlap:

* at ``D intersect C``, with ``q = sep**2``, the scalar hit time already has
  first and second coefficient derivatives proportional to ``q**-1`` and
  ``q**-2`` despite finite weighted derivatives;
* at chart ``O``, the matching interface ``x = r1 * (1 + theta)`` has event
  speed tending to zero as ``r1 -> 0``.

These exact obstructions falsify the proposed weighted-transversality route
as a G1 discharge.  They do not prove that no different closing map or
function class can work, and they do not prove Hilbert's sixteenth problem.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction

from omnibias.core.collapse.verdict import adjudicate_residual
from omnibias.core.verified.interval import Interval
from omnibias.dynamics.honesty_inventory import build_honesty
from omnibias.dynamics.scale_dichotomy import residual_blowup_height

__all__ = [
    "WeightedSectionReport",
    "identity_verdicts",
    "report",
    "residual_origin_weighted_speed",
    "residual_weighted_hit_d2q",
    "residual_weighted_hit_dq",
    "sample_coefficient_growth",
    "sample_origin_collapse",
]


def residual_weighted_hit_dq(
    rate: Fraction,
    q: Fraction,
    hit_time_dq: Fraction,
) -> Fraction:
    """``rate*q*tau_q + 1 = 0`` for ``tau=(1/rate) log(eta0/(q*eta_i))``."""
    return rate * q * hit_time_dq + 1


def residual_weighted_hit_d2q(
    rate: Fraction,
    q: Fraction,
    hit_time_d2q: Fraction,
) -> Fraction:
    """``rate*q^2*tau_qq - 1 = 0`` for the same scalar hit time."""
    return rate * q**2 * hit_time_d2q - 1


def residual_origin_weighted_speed(
    q: Fraction,
    theta: Fraction,
    eta0: Fraction,
    eta_tau: Fraction,
) -> Fraction:
    """Exact chart-O speed on ``r1=q, r2=2-q, L=q(2-q)``.

    Here ``sep=2-2q`` and the matching interface is
    ``x=r1(1+theta)``.  Thus ``eta_tau=x*eta0`` tends to zero with ``q``.
    The factored identity also records ``2L=q(2+sep)``.
    """
    sep = 2 - 2 * q
    L = q * (2 - q)
    return (2 + sep) * eta_tau - 2 * L * (1 + theta) * eta0


def _verdict(residual: Fraction) -> str:
    box = Interval.point(float(residual)) if residual == 0 else Interval.from_rational(residual)
    return adjudicate_residual(box, existential=False).status


def identity_verdicts() -> dict[str, str]:
    rate, q = Fraction(3, 2), Fraction(1, 16)
    theta, eta0 = Fraction(1, 8), Fraction(2)
    return {
        "weighted_hit_dq": _verdict(
            residual_weighted_hit_dq(rate, q, -1 / (rate * q))
        ),
        "weighted_hit_d2q": _verdict(
            residual_weighted_hit_d2q(rate, q, 1 / (rate * q**2))
        ),
        "origin_weighted_speed": _verdict(
            residual_origin_weighted_speed(
                q,
                theta,
                eta0,
                q * (1 + theta) * eta0,
            )
        ),
    }


def sample_coefficient_growth(
    *,
    rate: Fraction = Fraction(3, 2),
    n_lo: int = 16,
    n_hi: int = 256,
) -> dict[str, Fraction | bool]:
    """Compare ordinary coefficient derivatives on ``q=1/n``."""
    if rate <= 0 or n_lo <= 0 or n_hi <= n_lo:
        raise ValueError("require rate > 0 and 0 < n_lo < n_hi")
    q_lo = Fraction(1, n_lo)
    q_hi = Fraction(1, n_hi)
    d1_lo = 1 / (rate * q_lo)
    d1_hi = 1 / (rate * q_hi)
    d2_lo = 1 / (rate * q_lo**2)
    d2_hi = 1 / (rate * q_hi**2)
    return {
        "q_lo": q_lo,
        "q_hi": q_hi,
        "d1_lo": d1_lo,
        "d1_hi": d1_hi,
        "d2_lo": d2_lo,
        "d2_hi": d2_hi,
        "first_grows": d1_hi > d1_lo,
        "second_grows": d2_hi > d2_lo,
        "weighted_first": Fraction(1, rate),
        "weighted_second": Fraction(1, rate),
    }


def sample_origin_collapse(
    *,
    theta: Fraction = Fraction(1, 8),
    eta0: Fraction = Fraction(2),
    n_lo: int = 16,
    n_hi: int = 256,
) -> dict[str, Fraction | bool]:
    """Compare chart-O intrinsic event speeds on ``r1=q=1/n``."""
    if theta <= -1 or eta0 <= 0 or n_lo <= 0 or n_hi <= n_lo:
        raise ValueError("require theta > -1, eta0 > 0, and 0 < n_lo < n_hi")
    q_lo = Fraction(1, n_lo)
    q_hi = Fraction(1, n_hi)
    speed_lo = q_lo * (1 + theta) * eta0
    speed_hi = q_hi * (1 + theta) * eta0
    return {
        "q_lo": q_lo,
        "q_hi": q_hi,
        "speed_lo": speed_lo,
        "speed_hi": speed_hi,
        "speed_collapses": 0 < speed_hi < speed_lo,
    }


def _honesty(*, weighted_section_obstruction: bool) -> dict[str, object]:
    return build_honesty(
        weighted_section_obstruction=weighted_section_obstruction,
        new_closing_map=False,
        physical_c2_remainder=False,
        outgoing_first_hit=False,
        physical_return_membership_proved=False,
        g1_passed=False,
        g4_passed=False,
        full_graphic_cyclicity_proved=False,
        full_hilbert16_solved=False,
    )


@dataclass(frozen=True)
class WeightedSectionReport:
    """Exact negative assessment of H1.  Never a parent discharge."""

    identities: Mapping[str, str]
    coefficient_growth: Mapping[str, Fraction | bool]
    origin_collapse: Mapping[str, Fraction | bool]
    section_collapses_at_sep_zero: bool
    weighted_section_obstruction: bool
    honesty: Mapping[str, object]

    def to_payload(self) -> dict[str, object]:
        coefficient_growth = {
            key: float(value) if isinstance(value, Fraction) else value
            for key, value in self.coefficient_growth.items()
        }
        origin_collapse = {
            key: float(value) if isinstance(value, Fraction) else value
            for key, value in self.origin_collapse.items()
        }
        return {
            "schema": "hilbert16-weighted-section-v1",
            "identities": dict(self.identities),
            "coefficient_growth": coefficient_growth,
            "origin_collapse": origin_collapse,
            "section_collapses_at_sep_zero": self.section_collapses_at_sep_zero,
            "weighted_section_obstruction": self.weighted_section_obstruction,
            "weighted_section_closes_g1": False,
            "honesty": dict(self.honesty),
            "scope": (
                "The intrinsic eta-section is transverse for sep>0, but its "
                "ordinary q=sep^2 hit-time derivatives grow like q^-1 and "
                "q^-2 at D intersect C, while its chart-O matching speed "
                "vanishes with r1. This falsifies H1 as a G1 discharge, not "
                "every possible closing map or Hilbert XVI."
            ),
        }


def report() -> WeightedSectionReport:
    """Replay the three exact identities and both singular-overlap controls."""
    identities = identity_verdicts()
    growth = sample_coefficient_growth()
    origin = sample_origin_collapse()
    section_collapse = residual_blowup_height(
        Fraction(1, 16),
        Fraction(0),
        Fraction(2),
        Fraction(0),
    ) == 0
    sealed = (
        all(status == "PROVED" for status in identities.values())
        and bool(growth["first_grows"])
        and bool(growth["second_grows"])
        and bool(origin["speed_collapses"])
        and section_collapse
    )
    return WeightedSectionReport(
        identities=identities,
        coefficient_growth=growth,
        origin_collapse=origin,
        section_collapses_at_sep_zero=section_collapse,
        weighted_section_obstruction=sealed,
        honesty=_honesty(weighted_section_obstruction=sealed),
    )
