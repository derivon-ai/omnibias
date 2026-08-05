# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""The count-certificate bridge: seal a :class:`CountCertificate` and Lean-check it.

These run with or without a Lean toolchain: sealing, obligation *generation*, the
tamper path, and graceful degradation are always exercised; the actual ``lake`` build
(``theorem_prover_verified=True``) is asserted only in the opt-in class that skips when
``lake`` is absent -- mirroring ``omnibias-difference``'s W8 identity tests.
"""

from __future__ import annotations

import copy

import pytest
from omnibias.core.proof import check_certificate, generate_obligation
from omnibias.core.proof.certificate import schema_errors_v1, verify_certificate_digest
from omnibias.core.proof.lean_check import lean_check_available
from omnibias.logic import (
    CountCertificate,
    ProofMachine,
    count_enclosure,
    count_prover,
    exact_model_count,
    model_count,
    model_count_conjecture,
    prove_model_count,
    seal_count_certificate,
)

# Small, deterministic instances (n, clauses) with their exact model counts.
_SAT_CASES = [
    ([[1, 2], [-1, 3], [2, -3]], 3),
    ([[1, 2, 3]], 3),
    ([[1], [2], [3]], 3),
    ([[1, -2], [-1, 2]], 2),  # x1 <-> x2
]


def _tight_cert(clauses: list[list[int]], n: int) -> tuple[CountCertificate, object]:
    """A tight (exact) enclosure + its problem (order >= #clauses counts exactly)."""
    mc = model_count(clauses, n_vars=n)
    cert = count_enclosure(mc, order=len(clauses) + 1)
    return cert, mc


def _lhs_sum(sealed: dict) -> int:
    """The integer ``sum_i c_i m_i`` the Lean kernel checks against ``rhs``."""
    return sum(c * m for c, m in sealed["payload"]["lhs_terms"])


class TestExactCountIdentity:
    """A tight, unweighted enclosure seals to a kernel-checkable integer count identity."""

    @pytest.mark.parametrize("clauses,n", _SAT_CASES)
    def test_identity_reproduces_the_exact_count(self, clauses: list[list[int]], n: int) -> None:
        cert, mc = _tight_cert(clauses, n)
        assert cert.tight
        sealed = seal_count_certificate(cert, problem=mc)
        assert sealed["payload"]["type"] == "rational_identity"
        exact = int(exact_model_count(mc))
        # The finite inclusion-exclusion assembly the kernel re-checks IS the exact count.
        assert sealed["payload"]["rhs"] == exact
        assert _lhs_sum(sealed) == exact

    def test_identity_is_the_real_assembly_not_a_tautology(self) -> None:
        # Z0 + the signed Bonferroni terms: more than one term, with a genuine -1 coefficient.
        cert, mc = _tight_cert([[1, 2], [-1, 3], [2, -3]], 3)
        sealed = seal_count_certificate(cert, problem=mc)
        terms = sealed["payload"]["lhs_terms"]
        assert len(terms) >= 2
        assert terms[0][0] == 1  # the Z0 term
        assert any(c == -1 for c, _ in terms)  # the S_1 (odd) Bonferroni term

    def test_unsat_is_certified_as_count_zero(self) -> None:
        mc = model_count([[1], [-1]], n_vars=1)  # x1 and not x1: no model
        cert = count_enclosure(mc, order=2)
        assert cert.tight and cert.upper == 0.0
        sealed = seal_count_certificate(cert, problem=mc)
        assert sealed["payload"]["type"] == "rational_identity"
        assert sealed["payload"]["rhs"] == 0
        assert _lhs_sum(sealed) == 0
        obligation = generate_obligation(sealed)
        assert obligation is not None and "theorem obligation" in obligation

    def test_obligation_routes_through_the_equality_lemma(self) -> None:
        cert, mc = _tight_cert([[1, 2, 3]], 3)
        sealed = seal_count_certificate(cert, problem=mc)
        obligation = generate_obligation(sealed)
        assert obligation is not None
        assert "eq_of_mem_point" in obligation
        assert "theorem obligation" in obligation


class TestSignObligation:
    """When no exact identity applies, a positive lower bound seals to a sat sign."""

    def test_positive_lower_bound_witnesses_satisfiability(self) -> None:
        # A single wide clause: non-trivial order-1 enclosure with lower > 0.
        mc = model_count([[1, 2, 3, 4]], n_vars=6)
        cert = count_enclosure(mc, order=1)
        assert not cert.tight and cert.lower > 0.0
        sealed = seal_count_certificate(cert)  # no problem -> sign obligation only
        assert sealed["payload"]["type"] == "interval"
        assert sealed["meta"]["obligation"] == "interval_sign"
        obligation = generate_obligation(sealed)
        assert obligation is not None and "enclosed_quantity_pos" in obligation

    def test_seal_without_problem_still_certifies_sat_for_tight_sat(self) -> None:
        cert, _mc = _tight_cert([[1, 2], [-1, 3], [2, -3]], 3)
        sealed = cert.seal()  # tight SAT, but no problem -> sign, not identity
        assert sealed["payload"]["type"] == "interval"
        obligation = generate_obligation(sealed)
        assert obligation is not None and "enclosed_quantity_pos" in obligation

    def test_pure_upper_bound_has_no_finite_obligation(self) -> None:
        mc = model_count([[1, 2], [3]], n_vars=4)
        cert = count_enclosure(mc, order=0)  # trivial [0, Z0]
        assert cert.lower == 0.0
        sealed = seal_count_certificate(cert)
        assert sealed["payload"]["type"] == "interval"
        assert sealed["meta"]["obligation"] == "none"
        assert generate_obligation(sealed) is None
        result = check_certificate(sealed)
        assert not result.verified
        assert "obligation" in result.detail.lower()


class TestClassifierFallbacks:
    """The classifier never widens the claim: it falls back rather than force an identity."""

    def test_weighted_tight_falls_back_to_sign(self) -> None:
        # Weighted count is rational, so no *integer* identity: expect the sign obligation.
        mc = model_count([[1, 2]], weights=[[1.0, 2.0], [1.0, 3.0]], n_vars=2)
        cert = count_enclosure(mc, order=2)
        assert cert.tight and cert.weighted
        sealed = seal_count_certificate(cert, problem=mc)
        assert sealed["payload"]["type"] == "interval"

    def test_non_tight_with_problem_falls_back_to_sign(self) -> None:
        mc = model_count([[1, 2], [-1, 3], [2, -3]], n_vars=3)
        cert = count_enclosure(mc, order=1)  # truncated -> not tight
        assert not cert.tight
        sealed = seal_count_certificate(cert, problem=mc)
        assert sealed["payload"]["type"] == "interval"

    def test_clause_cap_skips_the_exponential_identity(self) -> None:
        cert, mc = _tight_cert([[1, 2], [-1, 3], [2, -3]], 3)
        sealed = seal_count_certificate(cert, problem=mc, identity_max_clauses=0)
        assert sealed["payload"]["type"] == "interval"


class TestSealingAndTamperEvidence:
    """Every sealed count certificate is well-formed, tamper-evident, and honest."""

    def test_sealed_certificate_is_well_formed(self) -> None:
        cert, mc = _tight_cert([[1, 2, 3]], 3)
        sealed = seal_count_certificate(cert, problem=mc)
        assert schema_errors_v1(sealed) == []
        assert verify_certificate_digest(sealed)
        assert sealed["meta"]["kind"] == "model_count"
        assert sealed["meta"]["poly_time_exact"] is False
        assert sealed["honesty"]["unproven_claim"] is False

    def test_method_and_free_function_agree(self) -> None:
        cert, mc = _tight_cert([[1, 2, 3]], 3)
        assert cert.seal(problem=mc)["digest"] == seal_count_certificate(cert, problem=mc)["digest"]

    def test_tampering_breaks_the_digest_and_is_rejected(self) -> None:
        cert, mc = _tight_cert([[1, 2], [-1, 3], [2, -3]], 3)
        sealed = seal_count_certificate(cert, problem=mc)
        tampered = copy.deepcopy(sealed)
        tampered["payload"]["rhs"] = sealed["payload"]["rhs"] + 1  # a false count
        assert not verify_certificate_digest(tampered)
        result = check_certificate(tampered)
        assert not result.verified
        assert "digest" in result.detail.lower()

    def test_a_false_count_would_be_rejected_by_the_kernel(self) -> None:
        # Independently of the digest: a wrong rhs makes the integer sum != rhs, so the
        # kernel's omega/decide would fail (the obligation is not vacuously true).
        cert, mc = _tight_cert([[1, 2], [-1, 3], [2, -3]], 3)
        sealed = seal_count_certificate(cert, problem=mc)
        assert _lhs_sum(sealed) == sealed["payload"]["rhs"]
        forged = copy.deepcopy(sealed)
        forged["payload"]["rhs"] = sealed["payload"]["rhs"] + 1
        assert _lhs_sum(forged) != forged["payload"]["rhs"]


class TestGracefulDegradation:
    """With no Lean toolchain the bridge never raises; the flag simply stays False."""

    def test_check_certificate_degrades_without_lean(self) -> None:
        cert, mc = _tight_cert([[1, 2, 3]], 3)
        sealed = seal_count_certificate(cert, problem=mc)
        result = check_certificate(sealed)
        if not lean_check_available():
            assert not result.available
            assert not result.verified
        else:  # pragma: no cover - only on a Lean-equipped runner
            assert result.available


class TestProofMachineIntegration:
    """A count certificate flows through ``ProofMachine`` and surfaces a full ``Verdict``."""

    def test_enclosure_claim_is_proved_with_all_gates(self) -> None:
        mc = model_count([[1, 2], [-1, 3], [2, -3]], n_vars=3)
        verdict = prove_model_count(mc, claim="enclosure")
        assert verdict.proved
        assert verdict.schema_ok and verdict.honesty_ok
        assert verdict.replay_ok is True  # independent O(2^n) oracle agrees
        assert verdict.kind == "model_count"
        assert verdict.certificate is not None and verify_certificate_digest(verdict.certificate)

    def test_count_claim_proved_disproved_blocked(self) -> None:
        mc = model_count([[1, 2], [-1, 3], [2, -3]], n_vars=3)
        exact = int(exact_model_count(mc))
        assert prove_model_count(mc, claim="count", value=exact).proved
        assert prove_model_count(mc, claim="count", value=exact + 1).disproved
        # A truncated (non-tight) enclosure cannot pin the exact value -> BLOCKED, not wrong.
        blocked = prove_model_count(mc, claim="count", value=exact, order=1)
        assert blocked.blocked

    def test_sat_and_unsat_claims(self) -> None:
        sat = model_count([[1, 2], [-1, 3], [2, -3]], n_vars=3)
        unsat = model_count([[1], [-1]], n_vars=1)
        assert prove_model_count(sat, claim="sat").proved
        assert prove_model_count(unsat, claim="unsat").proved
        assert prove_model_count(unsat, claim="sat").disproved
        assert prove_model_count(sat, claim="unsat").disproved

    def test_strict_mode_needs_digest_schema_and_replay(self) -> None:
        mc = model_count([[1, 2], [-1, 3], [2, -3]], n_vars=3)  # exactly 3 models
        machine = ProofMachine().register(count_prover())
        verdict = machine.evaluate(model_count_conjecture("c", mc, claim="count", value=3), strict=True)
        assert verdict.proved and verdict.replay_ok is True

    def test_missing_problem_blocks(self) -> None:
        from omnibias.logic import Conjecture

        verdict = ProofMachine().register(count_prover()).evaluate(
            Conjecture(name="bad", kind="model_count", data={})
        )
        assert verdict.blocked

    def test_independent_replay_catches_a_bad_count(self) -> None:
        cert, mc = _tight_cert([[1, 2], [-1, 3], [2, -3]], 3)
        sealed = seal_count_certificate(
            cert, problem=mc, meta_extra={"problem": {"n": mc.n, "clauses": [[1, 2], [-1, 3], [2, -3]]}}
        )
        prover = count_prover()
        assert prover.replay(sealed) is True
        forged = copy.deepcopy(sealed)
        forged["payload"]["rhs"] = sealed["payload"]["rhs"] + 1  # breaks sum_i c_i m_i = rhs
        assert prover.replay(forged) is False

    def test_schema_gate_flags_a_tampered_certificate(self) -> None:
        cert, mc = _tight_cert([[1, 2, 3]], 3)
        sealed = seal_count_certificate(cert, problem=mc)
        prover = count_prover()
        assert prover.schema_errors(sealed) == []
        tampered = copy.deepcopy(sealed)
        tampered["payload"]["rhs"] = sealed["payload"]["rhs"] + 1
        assert prover.schema_errors(tampered)  # digest mismatch reported

    def test_theorem_prover_verified_is_never_forged(self) -> None:
        mc = model_count([[1, 2], [-1, 3], [2, -3]], n_vars=3)  # exactly 3 models
        verdict = prove_model_count(mc, claim="count", value=3, lean_check=True)
        assert verdict.certificate is not None
        assert verdict.theorem_prover_verified == check_certificate(verdict.certificate).verified
        if not lean_check_available():
            assert not verdict.theorem_prover_verified

    def test_asserting_the_formal_claim_blocks_without_a_kernel(self) -> None:
        if lean_check_available():  # pragma: no cover - only on a non-Lean runner
            pytest.skip("Lean toolchain present; the formal claim can genuinely pass")
        mc = model_count([[1, 2], [-1, 3], [2, -3]], n_vars=3)  # would be PROVED (3 models)...
        verdict = prove_model_count(mc, claim="count", value=3, assert_theorem_prover=True)
        assert verdict.blocked and not verdict.theorem_prover_verified  # ...but the kernel gate blocks


@pytest.mark.skipif(not lean_check_available(), reason="no Lean toolchain (lake) on PATH")
class TestRealLeanKernelPass:
    """Opt-in: with a real ``lake`` toolchain, a true count identity earns the flag."""

    def test_exact_count_earns_a_kernel_pass(self) -> None:
        cert, mc = _tight_cert([[1, 2], [-1, 3], [2, -3]], 3)
        sealed = seal_count_certificate(cert, problem=mc)
        result = check_certificate(sealed)
        assert result.available
        assert result.verified  # a genuine kernel pass of Z0 - S1 + S2 - ... = #models

    def test_proof_machine_earns_theorem_prover_verified(self) -> None:
        mc = model_count([[1, 2], [-1, 3], [2, -3]], n_vars=3)
        verdict = prove_model_count(mc, claim="count", value=3, assert_theorem_prover=True)
        assert verdict.proved and verdict.theorem_prover_verified

    def test_certified_unsat_earns_a_kernel_pass(self) -> None:
        mc = model_count([[1], [-1]], n_vars=1)
        sealed = seal_count_certificate(count_enclosure(mc, order=2), problem=mc)
        assert check_certificate(sealed).verified

    def test_forged_count_is_rejected_by_the_kernel(self) -> None:
        cert, mc = _tight_cert([[1, 2, 3]], 3)
        sealed = seal_count_certificate(cert, problem=mc)
        forged = copy.deepcopy(sealed)
        forged["payload"]["rhs"] = sealed["payload"]["rhs"] + 1
        forged["payload"]["count"] = forged["payload"]["rhs"]
        from omnibias.core.proof.certificate import seal_certificate

        forged = seal_certificate({k: v for k, v in forged.items() if k != "digest"})
        assert not check_certificate(forged).verified
