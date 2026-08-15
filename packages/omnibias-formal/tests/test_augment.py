# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Tests for the ``mathlib_verified`` verdict tier (``omnibias.formal.augment``).

These run with or without a Lean toolchain.  Without one, the interval obligation
cannot be Mathlib-verified, so an *asserted* ``mathlib_verified`` claim downgrades
a ``PROVED`` verdict to ``BLOCKED`` (the honesty gate); the actual ``verified=True``
path is exercised only by the dedicated ``lean-analytic`` CI job.
"""

from __future__ import annotations

from omnibias.core.proof import (
    Conjecture,
    FunctionProver,
    ProofAttempt,
    ProofMachine,
    ProverRegistry,
)
from omnibias.core.proof.certificate import interval_certificate
from omnibias.core.verified.interval import Interval
from omnibias.formal import (
    MATHLIB_CLAIM_KEY,
    MathlibVerdict,
    evaluate_with_mathlib,
    mathlib_check_available,
)


def _machine() -> ProofMachine:
    """A one-prover machine that returns a sealed, positive interval certificate."""
    cert = interval_certificate("q", Interval(0.5, 2.0))

    def prove(_conjecture: Conjecture) -> ProofAttempt:
        return ProofAttempt(status="PROVED", certificate=cert, detail="stub")

    prover = FunctionProver(name="stub", kinds=frozenset({"stub"}), prove_fn=prove)
    return ProofMachine(ProverRegistry([prover]))


def _conj(**claims: bool) -> Conjecture:
    return Conjecture(name="c", kind="stub", claims=dict(claims))


def test_passthrough_without_request() -> None:
    verdict = evaluate_with_mathlib(_machine(), _conj())
    assert isinstance(verdict, MathlibVerdict)
    assert verdict.status == "PROVED"
    assert verdict.mathlib_verified is False
    assert verdict.mathlib_available is False
    # The core Mathlib-free formal tier is never touched by this wrapper.
    assert verdict.verdict.theorem_prover_verified is False


def test_mathlib_check_param_records_without_gating() -> None:
    verdict = evaluate_with_mathlib(_machine(), _conj(), mathlib_check=True)
    # ``mathlib_check`` records the tier without gating: a PROVED verdict is
    # never downgraded when the claim was not asserted.
    assert verdict.status == "PROVED"
    if not mathlib_check_available():
        assert verdict.mathlib_verified is False
        assert verdict.mathlib_available is False
        assert "enclosed_pos" in verdict.mathlib_obligation
    else:  # pragma: no cover - only on a machine with Lean + Mathlib
        assert verdict.mathlib_available is True
        assert verdict.mathlib_verified is True


def test_asserted_claim_without_verification_blocks() -> None:
    verdict = evaluate_with_mathlib(_machine(), _conj(**{MATHLIB_CLAIM_KEY: True}))
    if not mathlib_check_available():
        assert verdict.status == "BLOCKED"
        assert any("mathlib_verified" in o for o in verdict.obligations)
        # Never sets theorem_prover_verified; the tiers stay distinct.
        assert verdict.verdict.theorem_prover_verified is False


def test_reserved_key_is_stripped_so_block_is_the_mathlib_one() -> None:
    # If the reserved key were NOT stripped before core, core's *generic* honesty
    # gate would block with a different ("external evidence") message. Assert the
    # block, when it happens, is the Mathlib-specific one.
    verdict = evaluate_with_mathlib(_machine(), _conj(**{MATHLIB_CLAIM_KEY: True}))
    if verdict.status == "BLOCKED":
        assert any("Mathlib did not verify" in o for o in verdict.obligations)
        assert not any("external evidence" in o for o in verdict.obligations)


def test_unproven_claim_blocks_and_mathlib_never_rescues() -> None:
    # An unproven-result assertion is blocked by the untouched core honesty gate; the Mathlib
    # tier neither sets nor rescues it.
    verdict = evaluate_with_mathlib(_machine(), _conj(unproven_claim=True), mathlib_check=True)
    assert verdict.status == "BLOCKED"
    assert verdict.mathlib_verified is False


def test_summary_includes_mathlib_tier_and_never_asserts_an_unproven_result() -> None:
    summary = evaluate_with_mathlib(_machine(), _conj()).summary()
    assert summary["mathlib_verified"] is False
    assert "mathlib_available" in summary
    assert summary["status"] == "PROVED"
    assert summary.get("unproven_claim") is None
