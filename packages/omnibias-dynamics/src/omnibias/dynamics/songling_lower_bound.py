# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Reproduction-readiness audit for the Songling four-cycle lower bound.

Shi's quadratic system established ``H(2) >= 4`` in 1980.  Galias--Tucker
(2022, DOI ``10.1016/j.amc.2021.126691``) subsequently gave a rigorous
multiple-precision interval proof that the same system has exactly four limit
cycles.  Consequently an omnibias replay would be useful independent
verification, but it would not be the first certificate-backed lower bound.

The four cycles occur at radically different section scales (roughly
``10^-2, 10^-8, 10^-21, 10^-75``), and the smallest published displacement
sign is of order ``10^-295``.  The proof used up to 2048-bit arithmetic.
Omnibias's current :class:`~omnibias.core.verified.interval.Interval` algebra
has binary64 endpoints.  This module gives an exact falsifier for using that
backend unchanged: after forming the Songling ``xy`` coefficient, subtracting
its dominant baseline produces an interval containing zero instead of the
exact ``8*epsilon = -8*10^-52`` perturbation.

This is a backend-readiness result.  It neither imports the published theorem
as an omnibias certificate nor claims that arbitrary precision alone would
automatically reproduce its global return-map proof.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction

from omnibias.core.proof.certificate import (
    Cert,
    make_certificate,
    verify_certificate_digest,
)
from omnibias.core.proof.realization_replay import source_digest
from omnibias.core.realization.polynomial import SparsePolynomial
from omnibias.core.verified.interval import Interval
from omnibias.dynamics.compactify import PlanarPolynomialField
from omnibias.dynamics.honesty_inventory import build_honesty

__all__ = [
    "SonglingPrecisionAudit",
    "SonglingReport",
    "certify_songling_precision_audit",
    "report",
    "songling_field",
    "verify_songling_precision_audit",
]


LAMBDA = -Fraction(1, 10**200)
EPSILON = -Fraction(1, 10**52)
DELTA = -Fraction(1, 10**13)
PUBLISHED_MAX_PRECISION_BITS = 2048


def songling_field() -> PlanarPolynomialField:
    """Return the exact rational Shi--Songling quadratic vector field."""
    x = SparsePolynomial.variable(2, 0)
    y = SparsePolynomial.variable(2, 1)
    p = (
        LAMBDA * x
        - y
        - 10 * x**2
        + (5 + DELTA) * x * y
        + y**2
    )
    q = x + x**2 + (-25 + 8 * EPSILON - 9 * DELTA) * x * y
    return PlanarPolynomialField(p, q, 2)


def _q(value: Fraction) -> list[int]:
    return [value.numerator, value.denominator]


def _published_root_boxes() -> tuple[tuple[Fraction, Fraction], ...]:
    """Section boxes from Galias--Tucker Lemma 2, encoded as exact decimals."""
    return (
        (
            Fraction("0.042689603882075"),
            Fraction("0.042689603882085"),
        ),
        (
            Fraction("6.6666601481e-8"),
            Fraction("6.6666601482e-8"),
        ),
        (
            Fraction("2.247805947e-21"),
            Fraction("2.247805948e-21"),
        ),
        (
            Fraction("7.071067811865475244e-75"),
            Fraction("7.071067811865475245e-75"),
        ),
    )


def _published_displacement_margins() -> tuple[Fraction, ...]:
    """Absolute lower bounds quoted in Galias--Tucker Lemma 2."""
    return (
        Fraction("4.93e-15"),
        Fraction("2.32e-57"),
        Fraction("1.29e-123"),
        Fraction("5.03e-295"),
    )


def _epsilon_recovery_box() -> tuple[Fraction, Interval]:
    xy_coefficient = -25 + 8 * EPSILON - 9 * DELTA
    dominant_baseline = -25 - 9 * DELTA
    exact = xy_coefficient - dominant_baseline
    enclosed = Interval.from_rational(xy_coefficient) - Interval.from_rational(
        dominant_baseline
    )
    return exact, enclosed


@dataclass(frozen=True)
class SonglingPrecisionAudit:
    """Exact source data plus the binary64 cancellation obstruction."""

    field: PlanarPolynomialField
    root_boxes: tuple[tuple[Fraction, Fraction], ...]
    displacement_margins: tuple[Fraction, ...]
    exact_epsilon_effect: Fraction
    recovered_epsilon_box: Interval
    source_digest: str
    seal: Cert


def certify_songling_precision_audit() -> SonglingPrecisionAudit:
    """Seal the exact field and the current backend's precision refusal."""
    field = songling_field()
    roots = _published_root_boxes()
    margins = _published_displacement_margins()
    exact, recovered = _epsilon_recovery_box()
    if exact != 8 * EPSILON or not recovered.contains_zero():
        raise ArithmeticError("Songling precision audit did not reproduce its defining obstruction")
    payload = {
        "type": "songling_binary64_readiness_audit",
        "field": field.to_payload(),
        "parameters": {
            "lambda": _q(LAMBDA),
            "epsilon": _q(EPSILON),
            "delta": _q(DELTA),
        },
        "published_root_boxes": [
            [_q(lower), _q(upper)] for lower, upper in roots
        ],
        "published_displacement_margins": [_q(item) for item in margins],
        "published_max_precision_bits": PUBLISHED_MAX_PRECISION_BITS,
        "omnibias_interval_endpoint_bits": 53,
        "exact_epsilon_effect": _q(exact),
        "binary64_recovery_box_hex": [recovered.lo.hex(), recovered.hi.hex()],
        "scope": (
            "exact source transcription and binary64 precision audit; "
            "published cycle enclosures are metadata, not replayed omnibias "
            "return-map certificates"
        ),
    }
    digest = source_digest(payload)
    seal = make_certificate(
        claim=(
            "The exact Songling source is representable, but the current "
            "binary64 interval backend cannot recover a governing epsilon "
            "perturbation after dominant-coefficient cancellation."
        ),
        payload=payload,
        honesty={
            "songling_exact_source_transcribed": True,
            "published_four_cycle_theorem_replayed": False,
            "four_hyperbolic_returns_certified": False,
            "h2_lower_bound_independently_certified": False,
            "full_hilbert16_solved": False,
        },
        meta={"transcend_backend": "not_used"},
    )
    return SonglingPrecisionAudit(
        field,
        roots,
        margins,
        exact,
        recovered,
        digest,
        seal,
    )


def verify_songling_precision_audit(
    certificate: SonglingPrecisionAudit,
) -> bool:
    """Replay exact source data, digest, and the binary64 cancellation."""
    if not isinstance(certificate, SonglingPrecisionAudit):
        return False
    payload = certificate.seal.get("payload", {})
    exact, recovered = _epsilon_recovery_box()
    return (
        payload.get("type") == "songling_binary64_readiness_audit"
        and verify_certificate_digest(certificate.seal)
        and source_digest(payload) == certificate.source_digest
        and certificate.field == songling_field()
        and certificate.root_boxes == _published_root_boxes()
        and certificate.displacement_margins
        == _published_displacement_margins()
        and certificate.exact_epsilon_effect == exact
        and certificate.recovered_epsilon_box == recovered
        and recovered.contains_zero()
    )


@dataclass(frozen=True)
class SonglingReport:
    """H6 result: known rigorous theorem, but no omnibias replay."""

    exact_source_verified: bool
    published_h2_lower_bound: int
    published_exact_cycle_count: int
    published_certificate_precedes_omnibias: bool
    published_max_precision_bits: int
    omnibias_interval_endpoint_bits: int
    governing_epsilon_sign_resolved: bool
    four_hyperbolic_returns_certified: bool
    h2_lower_bound_independently_certified: bool
    honesty: Mapping[str, object]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": "hilbert16-songling-lower-bound-audit-v1",
            "exact_source_verified": self.exact_source_verified,
            "published_h2_lower_bound": self.published_h2_lower_bound,
            "published_exact_cycle_count": self.published_exact_cycle_count,
            "published_certificate_precedes_omnibias": (
                self.published_certificate_precedes_omnibias
            ),
            "published_max_precision_bits": self.published_max_precision_bits,
            "omnibias_interval_endpoint_bits": (
                self.omnibias_interval_endpoint_bits
            ),
            "governing_epsilon_sign_resolved": (
                self.governing_epsilon_sign_resolved
            ),
            "four_hyperbolic_returns_certified": (
                self.four_hyperbolic_returns_certified
            ),
            "h2_lower_bound_independently_certified": (
                self.h2_lower_bound_independently_certified
            ),
            "full_hilbert16_solved": False,
            "honesty": dict(self.honesty),
            "scope": (
                "Reproduction-readiness audit for an already-published rigorous "
                "four-cycle theorem. Not an omnibias orbit replay, an H(2) "
                "upper bound, finiteness of H(2), or Hilbert XVI."
            ),
        }


def report() -> SonglingReport:
    """Evaluate H6 against the published theorem and current backend."""
    audit = certify_songling_precision_audit()
    exact_source = verify_songling_precision_audit(audit)
    sign_resolved = not audit.recovered_epsilon_box.contains_zero()
    replayed = False
    honesty = build_honesty(
        songling_exact_source_transcribed=exact_source,
        published_four_cycle_theorem_replayed=replayed,
        four_hyperbolic_returns_certified=replayed,
        h2_lower_bound_independently_certified=replayed,
        full_hilbert16_solved=False,
    )
    return SonglingReport(
        exact_source_verified=exact_source,
        published_h2_lower_bound=4,
        published_exact_cycle_count=4,
        published_certificate_precedes_omnibias=True,
        published_max_precision_bits=PUBLISHED_MAX_PRECISION_BITS,
        omnibias_interval_endpoint_bits=53,
        governing_epsilon_sign_resolved=sign_resolved,
        four_hyperbolic_returns_certified=replayed,
        h2_lower_bound_independently_certified=replayed,
        honesty=honesty,
    )
