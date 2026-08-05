# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Soundness of the SOS certifier: proved cases are genuinely nonnegative, and
non-SOS / vanishing / negative polynomials are never over-claimed."""

from __future__ import annotations

import random
from fractions import Fraction

import pytest
from omnibias.sos.certify import certify_sos, is_sos, rational_gram
from omnibias.sos.monomials import gram_products
from omnibias.sos.problem import Polynomial, SOSCertificate

X = Polynomial.variable(0, 2)
Y = Polynomial.variable(1, 2)
ONE = Polynomial.constant(1.0, 2)


def _reconstruction_matches(cert: SOSCertificate, poly: Polynomial) -> bool:
    """Exact rational check that ``z(x)^T Q z(x) == poly``."""
    gram = rational_gram(cert)
    assert gram is not None
    products = gram_products(cert.basis)
    reconstructed: dict[tuple[int, ...], Fraction] = {}
    for alpha, pairs in products.items():
        value = sum((mult * gram[i][j] for i, j, mult in pairs), Fraction(0))
        if value != 0:
            reconstructed[alpha] = value
    expected = {alpha: Fraction(poly.coefficient(alpha)) for alpha in poly.support}
    return reconstructed == expected


def _sampled_min(poly: Polynomial, *, seed: int = 1234) -> float:
    """Minimum of ``poly`` over a dense grid plus random samples in a box."""
    rng = random.Random(seed)
    values = []
    for a in range(-32, 33):
        for b in range(-32, 33):
            values.append(poly.evaluate([a / 8.0, b / 8.0]))
    for _ in range(20000):
        values.append(poly.evaluate([rng.uniform(-8.0, 8.0), rng.uniform(-8.0, 8.0)]))
    return min(values)


PROVED_CASES: list[tuple[str, Polynomial]] = [
    ("positive-constant", 3.0 * ONE),
    ("pd-quadratic-plus-one", X * X - X * Y + Y * Y + ONE),
    ("completed-square-shift", 2.0 * X * X + 2.0 * Y * Y + 3.0 * ONE - 2.0 * X - 4.0 * Y),
    ("sum-of-explicit-squares", (X + Y) * (X + Y) + (X - 1.0) * (X - 1.0) + (Y - 1.0) * (Y - 1.0)),
    ("strictly-positive-quartic", X * X * X * X + Y * Y * Y * Y + ONE),
    ("quartic-with-cross", X * X * X * X + X * X * Y * Y + Y * Y * Y * Y + ONE),
]


@pytest.mark.parametrize(("name", "poly"), PROVED_CASES, ids=[c[0] for c in PROVED_CASES])
def test_proved_cases_are_sound(name: str, poly: Polynomial) -> None:
    cert = certify_sos(poly)
    assert cert.certified, f"{name}: expected proved, got {cert.status} ({cert.detail})"
    # The rational Gram matches the polynomial coefficients exactly ...
    assert cert.coeff_residual == 0.0
    assert _reconstruction_matches(cert, poly)
    # ... every certified interval LDL^T pivot is strictly positive ...
    assert cert.pivots
    assert all(lo > 0.0 for lo, _hi in cert.pivots)
    assert cert.pd_margin > 0.0
    assert cert.pd_margin == min(lo for lo, _hi in cert.pivots)
    # ... and the polynomial really is nonnegative (independent numeric cross-check).
    assert _sampled_min(poly) >= -1e-9


def test_motzkin_is_not_overclaimed() -> None:
    # The Motzkin polynomial is nonnegative but provably NOT a sum of squares.
    # A sound certifier must return inconclusive, never a false proof.
    motzkin = Polynomial(2, {(4, 2): 1.0, (2, 4): 1.0, (2, 2): -3.0, (0, 0): 1.0})
    cert = certify_sos(motzkin)
    assert cert.status == "inconclusive"
    assert not cert.certified
    assert cert.gram is None
    assert _sampled_min(motzkin) >= -1e-9  # it *is* nonnegative; SOS just cannot see it


def test_negative_polynomial_is_inconclusive() -> None:
    cert = certify_sos(-ONE - X * X)  # <= -1 everywhere
    assert cert.status == "inconclusive"
    assert not cert.certified


def test_vanishing_form_is_inconclusive_not_false() -> None:
    # x^2 + y^2 is SOS but vanishes at the origin, so it has no strictly-PD Gram.
    # Reporting inconclusive (rather than proved) preserves soundness.
    cert = certify_sos(X * X + Y * Y)
    assert cert.status == "inconclusive"
    assert not cert.certified


def test_is_sos_helper_agrees() -> None:
    assert is_sos(X * X - X * Y + Y * Y + ONE)
    assert not is_sos(Polynomial(2, {(4, 2): 1.0, (2, 4): 1.0, (2, 2): -3.0, (0, 0): 1.0}))


def test_certificate_never_claims_an_unproven_result() -> None:
    # The math container carries no unproven/continuum affordance at all.
    cert = certify_sos(X * X + Y * Y + ONE)
    assert not hasattr(cert, "unproven_claim")
    assert cert.status in {"proved", "inconclusive"}
