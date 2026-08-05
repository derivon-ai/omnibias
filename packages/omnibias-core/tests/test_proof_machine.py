# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Unit tests for the pure-Python prove/disprove orchestration engine."""

from __future__ import annotations

from omnibias.core.proof import (
    FORMAL_CLAIM_KEY,
    Certificate,
    Conjecture,
    FunctionProver,
    ProofAttempt,
    ProofMachine,
    ProverRegistry,
    Verdict,
    formal_claim_forgery_errors,
    honesty_gate,
)


def _sign_prover() -> FunctionProver:
    """A toy prover: positive => PROVED, negative => DISPROVED, zero => BLOCKED."""

    def prove(conjecture: Conjecture) -> ProofAttempt:
        value = float(conjecture.data["value"])
        cert: Certificate = {
            "value": value,
            "honesty": {"unproven_claim": False, "model_only": True},
        }
        if value > 0.0:
            return ProofAttempt(status="PROVED", certificate=cert)
        if value < 0.0:
            return ProofAttempt(status="DISPROVED", certificate=cert)
        return ProofAttempt(
            status="BLOCKED", certificate=cert, obligations=("value is zero",)
        )

    def schema(cert: Certificate) -> list[str]:
        return [] if "value" in cert else ["missing value"]

    def replay(cert: Certificate) -> bool | None:
        # A forged sentinel value 99 fails the independent replay.
        return float(cert["value"]) != 99.0

    return FunctionProver(
        name="sign",
        kinds=frozenset({"sign"}),
        prove_fn=prove,
        schema_fn=schema,
        replay_fn=replay,
    )


def test_machine_proves_disproves_and_blocks() -> None:
    machine = ProofMachine().register(_sign_prover())
    assert machine.kinds() == ("sign",)

    proved = machine.evaluate(Conjecture("pos", "sign", {"value": 3.0}))
    assert proved.status == "PROVED"
    assert proved.proved is True
    assert proved.schema_ok is True
    assert proved.replay_ok is True
    assert proved.honesty_ok is True

    disproved = machine.evaluate(Conjecture("neg", "sign", {"value": -3.0}))
    assert disproved.status == "DISPROVED"
    assert disproved.disproved is True

    blocked = machine.evaluate(Conjecture("zero", "sign", {"value": 0.0}))
    assert blocked.status == "BLOCKED"
    assert "value is zero" in blocked.obligations


def test_machine_honesty_gate_blocks_forged_unproven_claim() -> None:
    machine = ProofMachine().register(_sign_prover())
    verdict = machine.evaluate(
        Conjecture("forge", "sign", {"value": 3.0}, claims={"unproven_claim": True})
    )
    assert verdict.status == "BLOCKED"
    assert verdict.honesty_ok is False
    assert any("honesty gate" in o for o in verdict.obligations)


def test_machine_blocks_when_independent_replay_disagrees() -> None:
    machine = ProofMachine().register(_sign_prover())
    verdict = machine.evaluate(Conjecture("forged-value", "sign", {"value": 99.0}))
    assert verdict.status == "BLOCKED"
    assert verdict.replay_ok is False
    assert any("replay" in o for o in verdict.obligations)


def test_machine_blocks_when_schema_invalid() -> None:
    def prove(conjecture: Conjecture) -> ProofAttempt:
        return ProofAttempt(status="PROVED", certificate={"honesty": {}})  # no 'value'

    def schema(cert: Certificate) -> list[str]:
        return [] if "value" in cert else ["missing value"]

    machine = ProofMachine().register(
        FunctionProver(
            name="bad", kinds=frozenset({"bad"}), prove_fn=prove, schema_fn=schema
        )
    )
    verdict = machine.evaluate(Conjecture("x", "bad", {}))
    assert verdict.status == "BLOCKED"
    assert verdict.schema_ok is False
    assert any("schema" in o for o in verdict.obligations)


def test_machine_blocks_unknown_kind() -> None:
    machine = ProofMachine().register(_sign_prover())
    verdict = machine.evaluate(Conjecture("x", "unknown", {}))
    assert verdict.status == "BLOCKED"
    assert verdict.prover == "<none>"
    assert any("no registered prover" in o for o in verdict.obligations)


def test_honesty_gate_passes_when_no_claims_or_supported() -> None:
    conj = Conjecture("x", "k", {})
    assert honesty_gate(conj, None) is True
    cert: Certificate = {"honesty": {"interval_verified": True}}
    conj2 = Conjecture("y", "k", {}, claims={"interval_verified": True})
    assert honesty_gate(conj2, cert) is True
    conj3 = Conjecture("z", "k", {}, claims={"unproven_claim": True})
    assert honesty_gate(conj3, cert) is False


def test_registry_select_first_match_and_replay_none_keeps_verdict() -> None:
    def prove(conjecture: Conjecture) -> ProofAttempt:
        return ProofAttempt(status="PROVED", certificate={"honesty": {}})

    prover = FunctionProver(name="noreplay", kinds=frozenset({"k"}), prove_fn=prove)
    registry = ProverRegistry([prover])
    assert registry.select(Conjecture("a", "k", {})) is prover
    assert registry.select(Conjecture("a", "other", {})) is None

    machine = ProofMachine(registry)
    verdict = machine.evaluate(Conjecture("a", "k", {}))
    assert isinstance(verdict, Verdict)
    # No replay twin -> replay_ok is None and the PROVED verdict stands.
    assert verdict.replay_ok is None
    assert verdict.status == "PROVED"


def test_evaluate_all_batches() -> None:
    machine = ProofMachine().register(_sign_prover())
    conjectures = [
        Conjecture("a", "sign", {"value": 1.0}),
        Conjecture("b", "sign", {"value": -1.0}),
    ]
    verdicts = machine.evaluate_all(conjectures)
    assert [v.status for v in verdicts] == ["PROVED", "DISPROVED"]
    assert verdicts[0].summary()["status"] == "PROVED"


# --- the formal claim is earned, never asserted -----------------------------
#
# `ProofMachine.evaluate` already adjudicates FORMAL_CLAIM_KEY against a real
# kernel pass. These lock the *stored-artifact* half of that rule: a bundle on
# disk is editable, so a consumer reading the flag out of its `honesty` mapping
# must refuse it rather than take the artifact's word.


def test_honest_artifact_declares_nothing_and_passes() -> None:
    assert formal_claim_forgery_errors({FORMAL_CLAIM_KEY: False}) == []
    assert formal_claim_forgery_errors({}) == []
    assert formal_claim_forgery_errors(None) == []


def test_self_declared_formal_claim_is_refused_without_a_kernel_pass() -> None:
    errors = formal_claim_forgery_errors({FORMAL_CLAIM_KEY: True, "unproven_claim": True})
    assert len(errors) == 1
    assert FORMAL_CLAIM_KEY in errors[0]
    assert "earned" in errors[0]
    # Truthy non-bools must not slip through a `is True` style check.
    assert formal_claim_forgery_errors({FORMAL_CLAIM_KEY: 1})


def test_forgery_error_names_the_offending_field() -> None:
    (error,) = formal_claim_forgery_errors({FORMAL_CLAIM_KEY: True}, context="bundle.honesty")
    assert error.startswith(f"bundle.honesty.{FORMAL_CLAIM_KEY}")


def test_forgery_check_tolerates_a_malformed_honesty_field() -> None:
    """A hand-edited artifact may carry junk; the guard must not raise."""
    for junk in (["not", "a", "mapping"], "string", 42, 0.0):
        assert formal_claim_forgery_errors(junk) == []  # type: ignore[arg-type]


def test_declared_claim_stays_refused_when_the_certificate_cannot_be_verified() -> None:
    """With no Lean toolchain the bridge reports unverified -- that must refuse.

    Graceful degradation of the formal loop must never become silent acceptance:
    supplying a certificate is an *invitation to check*, not a bypass.
    """
    unverifiable: Certificate = {"schema_version": "not-a-real-obligation", "value": 1.0}
    errors = formal_claim_forgery_errors({FORMAL_CLAIM_KEY: True}, certificate=unverifiable)
    assert errors, "an unverifiable certificate must not launder the formal claim"


def test_honesty_gate_still_ignores_the_formal_claim() -> None:
    """The conjecture-side rule and the artifact-side rule must agree."""
    cert: Certificate = {"honesty": {FORMAL_CLAIM_KEY: False}}
    conjecture = Conjecture("x", "k", {}, claims={FORMAL_CLAIM_KEY: True})
    # honesty_gate defers to ProofMachine.evaluate, which consults the kernel.
    assert honesty_gate(conjecture, cert) is True
