# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Exact Picard--Fuchs syzygies for the cubic elliptic family."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

from omnibias.core.proof.certificate import verify_certificate_digest
from omnibias.holonomic import (
    certify_hyperelliptic_picard_fuchs,
    certify_picard_fuchs,
    diff_algebra,
    verify_hyperelliptic_picard_fuchs,
    verify_picard_fuchs,
)


def test_cubic_picard_fuchs_operator_is_exactly_certified() -> None:
    certificate = certify_picard_fuchs(
        -1,
        energy_factor=(Fraction(-1, 64), 0, 1),
    )

    assert certificate.verified
    assert certificate.period_operator.coeffs == (
        (Fraction(15),),
        (Fraction(0), Fraction(216)),
        (Fraction(-16), Fraction(0), Fraction(108)),
    )
    assert certificate.area_operator.order == 3
    assert certificate.integral_operator.order >= 3
    assert certificate.seal is not None
    assert verify_certificate_digest(certificate.seal)
    assert verify_picard_fuchs(certificate)


def test_perturbed_picard_fuchs_operator_is_rejected() -> None:
    bad = diff_algebra().operator(
        (
            (16,),
            (0, 216),
            (-16, 0, 108),
        )
    )

    certificate = certify_picard_fuchs(-1, candidate=bad)

    assert not certificate.verified
    assert certificate.seal is None
    assert not verify_picard_fuchs(certificate)


def test_picard_fuchs_refuses_inexact_input() -> None:
    try:
        certify_picard_fuchs(-1.0)
    except TypeError as exc:
        assert "exact rationals" in str(exc)
    else:  # pragma: no cover - explicit refusal is part of the contract
        raise AssertionError("float coefficients must be refused")


def test_general_de_rham_reduction_reproduces_cubic_operator() -> None:
    general = certify_hyperelliptic_picard_fuchs((0, -1, 0, 1))
    specialized = certify_picard_fuchs(-1)

    assert general.basis_dimension == 2
    assert general.period_operator.coeffs == specialized.period_operator.coeffs
    assert general.seal["payload"]["type"] == "integer_matrix_syzygy"
    assert verify_hyperelliptic_picard_fuchs(general)


def test_general_quartic_picard_fuchs_replays_and_rejects_tampering() -> None:
    certificate = certify_hyperelliptic_picard_fuchs((0, -1, 0, 0, 1))

    assert certificate.basis_dimension == 3
    assert certificate.period_operator.order == 3
    assert verify_hyperelliptic_picard_fuchs(certificate)

    bad_operator = certificate.period_operator + diff_algebra().operator(((1,),))
    assert not verify_hyperelliptic_picard_fuchs(
        replace(certificate, period_operator=bad_operator)
    )
