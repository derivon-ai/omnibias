# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Quantified one-scale dichotomy on the Hilbert-XVI kill sequence.

For ``eps=1/n``, ``sep=exp(-n**2)``, and a monomial blow-up scale
``sigma=eps**a * sep**b``, the logarithms of the two scalar factors are

``log(sigma*kappa) = (1-b)n**2 - a log(n)``

and, for a frozen section ``h_max=eps**N``,

``log(W_max/W_e) = 2bn + (2a+3-N)log(n)/n``.

Their exact asymptotic classifications are incompatible for every rational
``(a,b)``.  The same argument excludes every single positive scale when the
event and frozen-section factors must be bounded separately.  It does not
exclude moving sections, tracked cancellations, multistage charts, or general
quasi-homogeneous atlases.  Indeed ``h_max=eps**3*sep**2`` is an exact scalar
escape; :mod:`omnibias.dynamics.weighted_section` records why that escape
still fails the physical overlap requirements.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction

from omnibias.core.collapse.verdict import adjudicate_residual
from omnibias.core.verified.interval import Interval
from omnibias.dynamics.honesty_inventory import build_honesty

__all__ = [
    "QuasihomogeneousDichotomyReport",
    "classify_monomial_weight",
    "identity_verdicts",
    "monomial_bounds_both",
    "report",
    "residual_event_log_expansion",
    "residual_moving_section_log_expansion",
    "residual_outgoing_log_expansion",
]


def residual_event_log_expansion(
    log_event: Fraction,
    a: Fraction,
    b: Fraction,
    n_squared: Fraction,
    log_n: Fraction,
) -> Fraction:
    """Formal kill-sequence identity for ``log(sigma*kappa)``."""
    return log_event - ((1 - b) * n_squared - a * log_n)


def residual_outgoing_log_expansion(
    log_ratio: Fraction,
    a: Fraction,
    b: Fraction,
    height_power: int,
    n: Fraction,
    log_n: Fraction,
) -> Fraction:
    """Formal kill-sequence identity for a frozen-section W-ratio."""
    return log_ratio - (
        2 * b * n
        + (2 * a + 3 - height_power) * log_n / n
    )


def residual_moving_section_log_expansion(
    log_ratio: Fraction,
    a: Fraction,
    b: Fraction,
    sep_power: Fraction,
    height_power: int,
    n: Fraction,
    log_n: Fraction,
) -> Fraction:
    """Formal W-ratio identity for ``h_max=eps**N*sep**sep_power``."""
    return log_ratio - (
        (2 * b - sep_power) * n
        + (2 * a + 3 - height_power) * log_n / n
    )


def _verdict(residual: Fraction) -> str:
    box = Interval.point(float(residual)) if residual == 0 else Interval.from_rational(residual)
    return adjudicate_residual(box, existential=False).status


def identity_verdicts() -> dict[str, str]:
    a, b = Fraction(2, 3), Fraction(5, 4)
    n, log_n = Fraction(7), Fraction(3, 2)
    event_log = (1 - b) * n**2 - a * log_n
    frozen_log = 2 * b * n + (2 * a + 3) * log_n / n
    moving_log = (2 * b - 2) * n + (2 * a) * log_n / n
    return {
        "quasi_event_log": _verdict(
            residual_event_log_expansion(event_log, a, b, n**2, log_n)
        ),
        "quasi_frozen_log": _verdict(
            residual_outgoing_log_expansion(frozen_log, a, b, 0, n, log_n)
        ),
        "quasi_moving_log": _verdict(
            residual_moving_section_log_expansion(
                moving_log,
                a,
                b,
                Fraction(2),
                3,
                n,
                log_n,
            )
        ),
    }


def classify_monomial_weight(a: Fraction, b: Fraction) -> dict[str, bool | str]:
    """Exact asymptotic classification for rational monomial weights."""
    event_bounded = b > 1 or (b == 1 and a >= 0)
    frozen_ratio_bounded_above = b <= 0
    frozen_log_two_sided_bounded = b == 0
    if b < 1:
        event_behavior = "diverges"
    elif b > 1 or a > 0:
        event_behavior = "tends_to_zero"
    elif a == 0:
        event_behavior = "tends_to_one"
    else:
        event_behavior = "diverges"
    if b > 0:
        outgoing_behavior = "diverges_positive"
    elif b < 0:
        outgoing_behavior = "diverges_negative"
    else:
        outgoing_behavior = "tends_to_zero"
    return {
        "event_bounded": event_bounded,
        "frozen_ratio_bounded_above": frozen_ratio_bounded_above,
        "frozen_log_two_sided_bounded": frozen_log_two_sided_bounded,
        "bounds_both": event_bounded and frozen_ratio_bounded_above,
        "event_behavior": event_behavior,
        "outgoing_behavior": outgoing_behavior,
    }


def monomial_bounds_both(a: Fraction, b: Fraction) -> bool:
    """Always false: the event requires ``b>=1`` and the ratio ``b<=0``."""
    return bool(classify_monomial_weight(a, b)["bounds_both"])


def _honesty(*, frozen_section_scale_no_go: bool) -> dict[str, object]:
    return build_honesty(
        frozen_section_scale_no_go=frozen_section_scale_no_go,
        all_quasihomogeneous_atlases_excluded=False,
        new_closing_map=False,
        scale_dichotomy_c2_remainder=False,
        physical_c2_remainder=False,
        g1_passed=False,
        g4_passed=False,
        full_graphic_cyclicity_proved=False,
        full_hilbert16_solved=False,
    )


@dataclass(frozen=True)
class QuasihomogeneousDichotomyReport:
    """Maximal true H2 theorem plus its moving-section counterexample."""

    identities: Mapping[str, str]
    boundary_cases: Mapping[str, Mapping[str, bool | str]]
    all_rational_monomial_weights_excluded: bool
    moving_section_counterexample: bool
    frozen_section_scale_no_go: bool
    honesty: Mapping[str, object]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": "hilbert16-quasihomogeneous-dichotomy-v1",
            "identities": dict(self.identities),
            "boundary_cases": {
                key: dict(value) for key, value in self.boundary_cases.items()
            },
            "all_rational_monomial_weights_excluded": (
                self.all_rational_monomial_weights_excluded
            ),
            "moving_section_counterexample": self.moving_section_counterexample,
            "frozen_section_scale_no_go": self.frozen_section_scale_no_go,
            "all_quasihomogeneous_atlases_excluded": False,
            "g1_passed": False,
            "honesty": dict(self.honesty),
            "scope": (
                "No single positive scale, including every rational monomial "
                "sigma=eps^a sep^b, separately bounds the event factor and "
                "the W-ratio to a frozen eps^N section. A moving sep^2 "
                "section is an exact scalar counterexample to the broader "
                "atlas claim. Not physical C2, G1, or Hilbert XVI."
            ),
        }


def report() -> QuasihomogeneousDichotomyReport:
    """Replay the quantified monomial classification and its boundary cases."""
    identities = identity_verdicts()
    cases = {
        "b_below_one": classify_monomial_weight(Fraction(0), Fraction(1, 2)),
        "b_one_a_negative": classify_monomial_weight(Fraction(-1), Fraction(1)),
        "b_one_a_zero": classify_monomial_weight(Fraction(0), Fraction(1)),
        "b_one_a_positive": classify_monomial_weight(Fraction(1), Fraction(1)),
        "b_above_one": classify_monomial_weight(Fraction(-10), Fraction(2)),
        "b_zero": classify_monomial_weight(Fraction(3), Fraction(0)),
        "b_negative": classify_monomial_weight(Fraction(3), Fraction(-1)),
    }
    classified = (
        not bool(cases["b_below_one"]["event_bounded"])
        and not bool(cases["b_one_a_negative"]["event_bounded"])
        and bool(cases["b_one_a_zero"]["event_bounded"])
        and bool(cases["b_one_a_positive"]["event_bounded"])
        and bool(cases["b_above_one"]["event_bounded"])
        and bool(cases["b_zero"]["frozen_ratio_bounded_above"])
        and bool(cases["b_negative"]["frozen_ratio_bounded_above"])
        and not any(bool(case["bounds_both"]) for case in cases.values())
    )
    all_monomials = classified
    moving_escape = (
        not monomial_bounds_both(Fraction(0), Fraction(1))
        and residual_moving_section_log_expansion(
            Fraction(0),
            Fraction(0),
            Fraction(1),
            Fraction(2),
            3,
            Fraction(7),
            Fraction(3, 2),
        )
        == 0
    )
    sealed = (
        all(status == "PROVED" for status in identities.values())
        and all_monomials
        and moving_escape
    )
    return QuasihomogeneousDichotomyReport(
        identities=identities,
        boundary_cases=cases,
        all_rational_monomial_weights_excluded=all_monomials,
        moving_section_counterexample=moving_escape,
        frozen_section_scale_no_go=sealed,
        honesty=_honesty(frozen_section_scale_no_go=sealed),
    )
