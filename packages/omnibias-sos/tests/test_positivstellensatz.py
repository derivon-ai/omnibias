# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Constrained positivity: certified Putinar representations are sound identities."""

from __future__ import annotations

import random
from fractions import Fraction

from omnibias.core.proof.certificate import schema_errors_v1, verify_certificate_digest
from omnibias.sos.monomials import gram_products
from omnibias.sos.positivstellensatz import (
    PositivstellensatzCertificate,
    certify_nonneg_on_set,
    is_nonneg_on_set,
    seal_positivstellensatz_certificate,
)
from omnibias.sos.problem import Polynomial

X = Polynomial.variable(0, 2)
Y = Polynomial.variable(1, 2)
ONE2 = Polynomial.constant(1.0, 2)
X1 = Polynomial.variable(0, 1)
ONE1 = Polynomial.constant(1.0, 1)


def _multiplier_poly(mult) -> Polynomial:  # type: ignore[no-untyped-def]
    n = len(mult.basis[0])
    gram = [[Fraction(mult.gram[i][j]) for j in range(len(mult.basis))] for i in range(len(mult.basis))]
    coeffs: dict[tuple[int, ...], float] = {}
    for alpha, pairs in gram_products(mult.basis).items():
        value = sum((m * gram[i][j] for i, j, m in pairs), Fraction(0))
        if value != 0:
            coeffs[alpha] = float(value)
    return Polynomial(n, coeffs)


def _identity_holds(cert: PositivstellensatzCertificate, p: Polynomial, gs: list[Polynomial]) -> bool:
    ext = [Polynomial.constant(1.0, p.n_vars), *gs]
    acc = Polynomial.zero(p.n_vars)
    for mult in cert.multipliers:
        acc = acc + _multiplier_poly(mult) * ext[mult.constraint_index + 1]
    keys = set(p.support) | set(acc.support)
    return all(Fraction(p.coefficient(k)) == Fraction(acc.coefficient(k)) for k in keys)


def _min_in_set(p: Polynomial, gs: list[Polynomial], *, seed: int = 0) -> float:
    rng = random.Random(seed)
    best = float("inf")
    count = 0
    while count < 30000:
        point = [rng.uniform(-4.0, 4.0) for _ in range(p.n_vars)]
        if all(g.evaluate(point) >= 0.0 for g in gs):
            best = min(best, p.evaluate(point))
            count += 1
    return best


def test_ball_constraint_is_proved_and_exact() -> None:
    p = 2.0 * ONE2 - X * X - Y * Y
    gs = [ONE2 - X * X - Y * Y]  # closed unit disk
    cert = certify_nonneg_on_set(p, gs)
    assert cert.certified
    assert cert.pd_margin > 0.0
    assert all(lo > 0.0 for lo, _hi in cert.pivots)
    assert _identity_holds(cert, p, gs)
    assert _min_in_set(p, gs) >= -1e-9


def test_interval_constraint_is_proved_and_exact() -> None:
    p = 2.0 * ONE1 - X1 * X1
    gs = [ONE1 - X1 * X1]  # [-1, 1]
    cert = certify_nonneg_on_set(p, gs)
    assert cert.certified
    assert _identity_holds(cert, p, gs)
    assert _min_in_set(p, gs) >= -1e-9
    assert is_nonneg_on_set(p, gs)


def test_false_statement_is_inconclusive() -> None:
    # x - 5 <= -4 on [-1, 1]; must never be "proved".
    cert = certify_nonneg_on_set(X1 - 5.0 * ONE1, [ONE1 - X1 * X1])
    assert cert.status == "inconclusive"
    assert not cert.certified


def test_seal_makes_no_unproven_claim_and_is_valid() -> None:
    p = 2.0 * ONE2 - X * X - Y * Y
    gs = [ONE2 - X * X - Y * Y]
    cert = certify_nonneg_on_set(p, gs)
    sealed = seal_positivstellensatz_certificate(
        cert, claim="2 - x^2 - y^2 >= 0 on the closed unit disk"
    )
    assert schema_errors_v1(sealed) == []
    assert verify_certificate_digest(sealed)
    assert sealed["honesty"]["unproven_claim"] is False
    assert sealed["meta"]["sos"]["form"] == "positivstellensatz"
    assert sealed["payload"]["type"] == "positive_definite"
