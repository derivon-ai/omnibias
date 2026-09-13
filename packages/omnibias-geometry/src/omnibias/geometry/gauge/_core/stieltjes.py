# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Finite rational necessary conditions for an unsubtracted Stieltjes model.

The model is ``D(s) = integral rho(dt)/(s+t)`` for a positive measure on
``[0, infinity)`` and finite values at the two supplied momenta. Neither this
model's applicability to an observable nor the coverage of supplied value
boxes is established here. In particular statistical error bars are not
automatically sound enclosures, and a gauge-fixed gluon correlator need not
have this positive representation.

The elementary necessary inequalities are D >= 0, D nonincreasing and s*D
nondecreasing. Strict violations throughout a supplied rational box exclude
this model from that box. Passing these tests proves no existence result.
When a finite support-floor bound is available, its direction is UPPER:
the weighted average of spectral mass squared bounds the support infimum
from above. It is never a positive lower bound on a physical mass gap.
"""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
from typing import Any

from omnibias.core.proof.certificate import make_certificate, verify_certificate_digest

_INPUT_NAMES = ("s1", "lower1", "upper1", "s2", "lower2", "upper2")
_TYPE = "stieltjes_pair_box"
_SCOPE = (
    "finite exact arithmetic on supplied rational boxes for an unsubtracted "
    "positive Stieltjes model; physical applicability and true-value coverage unverified"
)


def _q(value: int | Fraction) -> Fraction:
    if type(value) is not int and not isinstance(value, Fraction):
        raise TypeError("use an exact integer or Fraction; bool and float are refused")
    return Fraction(value)


def stieltjes_pair_box(
    s1: int | Fraction,
    lower1: int | Fraction,
    upper1: int | Fraction,
    s2: int | Fraction,
    lower2: int | Fraction,
    upper2: int | Fraction,
) -> dict[str, Any]:
    """Exclude a positive Stieltjes model using strict rational box inequalities.

    Require ``0 <= s1 < s2`` and ordered closed value boxes. Negative box
    endpoints are permitted: positivity is a necessary model condition to
    test, not an input assumption. Only integers and ``Fraction`` are accepted.

    ``INCOMPATIBLE`` means that no pair of model values lies in these boxes;
    it does not reject a physical theory without verified applicability and
    coverage premises. ``INCONCLUSIVE`` is not a positive representation or
    gap certificate, even for point boxes. In particular the necessary checks
    deliberately do not decide all boundary cases.

    For true values ``a = D(s1) > b = D(s2) > 0``, subtracting the integrals
    gives ``R = (s2*b - s1*a)/(a-b)`` as the average of t under the positive
    weight ``rho(dt)/((s1+t)*(s2+t))``. Thus ``inf supp(rho) <= R``. On a box
    with ``lower1 > upper2 > 0``, R decreases in a and increases in b, giving
    the displayed conditional support-floor UPPER bound. All analytic and
    physical premises remain unverified by this finite arithmetic checker.
    """
    values = tuple(map(_q, (s1, lower1, upper1, s2, lower2, upper2)))
    q1, lo1, hi1, q2, lo2, hi2 = values
    if not 0 <= q1 < q2:
        raise ValueError("need 0 <= s1 < s2")
    if lo1 > hi1 or lo2 > hi2:
        raise ValueError("need lower <= upper for both closed value boxes")

    margins = {
        "negative_first_value": -hi1,
        "negative_second_value": -hi2,
        "increasing_D": lo2 - hi1,
        "decreasing_sD": q1 * lo1 - q2 * hi2,
    }
    obstructions = [
        {"code": code, "strict_margin": str(margin)}
        for code, margin in margins.items()
        if margin > 0
    ]
    incompatible = bool(obstructions)
    upper: Fraction | None = None
    if not incompatible and lo1 > hi2 > 0:
        upper = (q2 * hi2 - q1 * lo1) / (lo1 - hi2)

    premises = {
        "observable_has_unsubtracted_positive_stieltjes_representation": "UNVERIFIED",
        "spectral_measure_supported_on_nonnegative_mass_squared": "UNVERIFIED",
        "listed_stieltjes_values_are_finite": "UNVERIFIED",
        "no_subtraction_or_contact_terms": "UNVERIFIED",
        "supplied_boxes_cover_true_values": "UNVERIFIED",
        "momenta_and_spectral_variable_use_one_fixed_physical_scale": "UNVERIFIED",
    }
    honesty = {
        "supplied_boxes_verified": False,
        "physical_applicability_verified": False,
        "physical_exclusion_claim": False,
        "positive_spectral_representation_claim": False,
        "spectral_gap_claim": False,
        "mass_gap_lower_bound_claim": False,
        "yang_mills_claim": False,
        "yang_mills_mass_gap_claim": False,
        "continuum_claim": False,
        "uniform_in_a_claim": False,
        "infinite_volume_claim": False,
        "analytic_implication_formally_verified": False,
        "mathlib_verified": False,
        "unproven_claim": False,
    }
    report: dict[str, Any] = {
        "status": "INCOMPATIBLE" if incompatible else "INCONCLUSIVE",
        "verification_kind": "EXACT_RATIONAL",
        "finite_box_exclusion_verified": incompatible,
        "inputs": {name: str(value) for name, value in zip(_INPUT_NAMES, values, strict=True)},
        "necessary_inequality_margins": {key: str(value) for key, value in margins.items()},
        "obstructions": obstructions,
        "conditional_support_floor_upper": {
            "status": "CONDITIONAL" if upper is not None else "UNAVAILABLE",
            "mass_squared_upper": str(upper) if upper is not None else None,
            "direction": "UPPER",
            "quantity": "infimum of support of the model's spectral mass-squared measure",
            "interpretation": "conditional upper bound; never a mass-gap lower bound",
        },
        "external_premises": premises,
        "scope": _SCOPE,
        "theorem_prover_verified": False,
        **honesty,
    }
    certificate = make_certificate(
        claim="finite rational necessary-condition check for a supplied Stieltjes pair box",
        payload={"type": _TYPE, "report": report},
        honesty=honesty,
        meta={"transcend_backend": "not_used", "scope": _SCOPE},
    )
    return {
        **report,
        "certificate": certificate,
        "digest_verified": verify_certificate_digest(certificate),
    }


def replay_stieltjes_pair_certificate(certificate: Mapping[str, Any]) -> bool:
    """Recompute and compare the complete canonical envelope, including scope.

    A true replay verifies what this checker reported, including an honest
    ``INCONCLUSIVE`` result. It does not establish any external premise or
    promote that result to a positive spectral representation.
    """
    if not isinstance(certificate, Mapping):
        return False
    try:
        if not verify_certificate_digest(certificate):
            return False
        payload = certificate["payload"]
        if payload["type"] != _TYPE:
            return False
        inputs = payload["report"]["inputs"]
        # Canonical strings are required; Fraction(float) must not silently
        # rationalize a tampered numerical input during replay.
        if not all(isinstance(inputs[name], str) for name in _INPUT_NAMES):
            return False
        values = tuple(Fraction(inputs[name]) for name in _INPUT_NAMES)
        replay = stieltjes_pair_box(*values)
        return bool(replay["certificate"] == certificate)
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError):
        return False


__all__ = ["replay_stieltjes_pair_certificate", "stieltjes_pair_box"]
