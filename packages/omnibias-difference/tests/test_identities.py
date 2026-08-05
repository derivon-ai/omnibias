# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""W8: a Lean-checkable corpus of rational special-number identities.

Each builder turns a finite, rational special-number identity into a sealed
certificate whose obligation the Mathlib-free Lean kernel discharges via the new
``enclosed_quantity_eq`` (equality) lemma or the existing sign lemmas. The honesty
invariant from ``omnibias-dev-certificate-lean`` is enforced: ``theorem_prover_verified``
is exactly ``lean.verified`` -- earned only by a genuine ``lake build`` pass.
"""

from __future__ import annotations

import copy
from fractions import Fraction
from math import comb

import pytest
from omnibias.core.proof.certificate import schema_errors_v1, verify_certificate_digest
from omnibias.core.proof.lean_check import (
    check_certificate,
    generate_obligation,
    lean_check_available,
)
from omnibias.core.verified.coeffs import bernoulli_number_exact, euler_number_exact
from omnibias.difference import (
    bernoulli_recurrence_identity,
    bernoulli_sign_certificate,
    check_identity_certificate,
    euler_recurrence_identity,
    rational_value_identity,
    zeta_negative_odd_identity,
)

# Textbook trivial zeta values zeta(1-2m) = -B_2m/(2m).
_ZETA_NEG = {1: Fraction(-1, 12), 2: Fraction(1, 120), 3: Fraction(-1, 252), 4: Fraction(1, 240)}


def _lhs_sum(cert: dict) -> int:
    """Evaluate the integer ``sum_i c_i m_i`` the kernel would check against ``rhs``."""
    payload = cert["payload"]
    return sum(c * m for c, m in payload["lhs_terms"])


class TestIdentitiesHoldExactly:
    """The rational data is correct: the kernel's integer obligation is really 0."""

    @pytest.mark.parametrize("n", [2, 3, 4, 5, 6, 7, 8, 10])
    def test_bernoulli_recurrence_is_zero(self, n: int) -> None:
        cert = bernoulli_recurrence_identity(n)
        assert _lhs_sum(cert) == cert["payload"]["rhs"] == 0
        # Independent check of the classical recurrence in exact rationals.
        assert sum(comb(n, k) * bernoulli_number_exact(k) for k in range(n)) == 0

    @pytest.mark.parametrize("m", [1, 2, 3, 4, 5])
    def test_euler_recurrence_is_zero(self, m: int) -> None:
        cert = euler_recurrence_identity(m)
        assert _lhs_sum(cert) == 0
        assert sum(comb(2 * m, 2 * k) * euler_number_exact(2 * k) for k in range(m + 1)) == 0

    @pytest.mark.parametrize("m", [1, 2, 3, 4])
    def test_zeta_negative_odd_matches_textbook(self, m: int) -> None:
        cert = zeta_negative_odd_identity(m, _ZETA_NEG[m])
        assert _lhs_sum(cert) == 0  # -B_2m/(2m) == textbook value
        assert -bernoulli_number_exact(2 * m) / Fraction(2 * m) == _ZETA_NEG[m]


class TestObligationGeneration:
    """Every identity emits a finite obligation routed through the new kernel lemma."""

    def test_equality_obligations_use_the_new_lemma(self) -> None:
        for cert in (
            bernoulli_recurrence_identity(6),
            euler_recurrence_identity(3),
            zeta_negative_odd_identity(1, _ZETA_NEG[1]),
        ):
            obligation = generate_obligation(cert)
            assert obligation is not None
            assert "eq_of_mem_point" in obligation
            assert "theorem obligation" in obligation

    def test_bernoulli_sign_obligation_alternates(self) -> None:
        # sign(B_2m) = (-1)^{m+1}: +, -, +, - ... drives the EM remainder sign.
        pos = generate_obligation(bernoulli_sign_certificate(1))  # B_2 = 1/6 > 0
        neg = generate_obligation(bernoulli_sign_certificate(2))  # B_4 = -1/30 < 0
        assert pos is not None and "enclosed_quantity_pos" in pos
        assert neg is not None and "enclosed_quantity_neg" in neg


class TestSealingAndHonesty:
    def test_certificates_are_well_formed_and_sealed(self) -> None:
        cert = bernoulli_recurrence_identity(6)
        assert schema_errors_v1(cert) == []
        assert verify_certificate_digest(cert)
        assert cert["payload"]["type"] == "rational_identity"
        assert cert["meta"]["identity"] == "bernoulli_recurrence"

    def test_tampering_breaks_the_digest(self) -> None:
        cert = zeta_negative_odd_identity(1, _ZETA_NEG[1])
        tampered = copy.deepcopy(cert)
        tampered["payload"]["rhs"] = 1  # a false identity
        assert not verify_certificate_digest(tampered)
        result = check_certificate(tampered)
        assert not result.verified
        assert "digest" in result.detail.lower()

    def test_theorem_prover_verified_is_never_forged(self) -> None:
        verdict = check_identity_certificate(bernoulli_recurrence_identity(4))
        assert verdict.theorem_prover_verified == verdict.lean.verified
        assert verdict.obligation_generated and verdict.sealed_ok
        if not lean_check_available():
            assert not verdict.lean.available
            assert not verdict.theorem_prover_verified

    def test_false_identity_would_be_rejected_by_the_kernel(self) -> None:
        # A wrong value (zeta(-1) = -1/13) yields a nonzero integer sum, so the
        # kernel's `omega`/`decide` would fail -- the obligation is not vacuously true.
        cert = rational_value_identity("bad", Fraction(-1, 12), Fraction(-1, 13))
        assert _lhs_sum(cert) != 0


class TestInputGuards:
    def test_domain_guards(self) -> None:
        with pytest.raises(ValueError, match="n >= 2"):
            bernoulli_recurrence_identity(1)
        with pytest.raises(ValueError, match="m >= 1"):
            euler_recurrence_identity(0)
        with pytest.raises(ValueError, match="m must be >= 1"):
            bernoulli_sign_certificate(0)


@pytest.mark.skipif(not lean_check_available(), reason="no Lean toolchain (lake) on PATH")
class TestRealLeanKernelPass:
    """Opt-in: with a real ``lake`` toolchain, a true identity earns the flag."""

    def test_bernoulli_recurrence_earns_theorem_prover_verified(self) -> None:
        verdict = check_identity_certificate(bernoulli_recurrence_identity(6))
        assert verdict.lean.available
        assert verdict.theorem_prover_verified  # a genuine kernel pass

    def test_zeta_value_identity_earns_theorem_prover_verified(self) -> None:
        verdict = check_identity_certificate(zeta_negative_odd_identity(2, _ZETA_NEG[2]))
        assert verdict.theorem_prover_verified

    def test_false_identity_is_rejected_by_the_kernel(self) -> None:
        # Directly hand the bridge a wrong-but-sealed identity; the kernel fails.
        cert = rational_value_identity("bad", Fraction(-1, 12), Fraction(-1, 13))
        verdict = check_identity_certificate(cert)
        assert not verdict.theorem_prover_verified
