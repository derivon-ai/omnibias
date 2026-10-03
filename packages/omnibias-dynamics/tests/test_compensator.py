# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
from __future__ import annotations

from fractions import Fraction

from omnibias.dynamics.compensator import (
    certify_compensator_collar_bound,
    certify_compensator_enclosure,
    compensator_derivative,
    verify_compensator_collar_bound,
    verify_compensator_certificate,
)
from omnibias.core.verified.interval import Interval


def test_compensator_derivative_encloses_log_limit() -> None:
    box = Interval(0.1, 0.5)
    derivative = compensator_derivative(Fraction(0), 1, box)
    assert derivative.lo <= 1.0 / box.hi


def test_compensator_collar_bound_certificate_replays() -> None:
    certificate = certify_compensator_collar_bound(
        alpha=Fraction(1, 2),
        s_lo=Fraction(1, 10),
        s_hi=Fraction(1, 2),
        principal_terms=((Fraction(1), 0, 1),),
    )
    assert verify_compensator_collar_bound(certificate)


def test_compensator_certificate_replays() -> None:
    certificate = certify_compensator_enclosure(
        alpha=Fraction(1, 2),
        s_lo=Fraction(1, 10),
        s_hi=Fraction(1, 2),
    )
    assert verify_compensator_certificate(certificate)
    assert certificate.seal["honesty"]["full_hilbert16_solved"] is False
