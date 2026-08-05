# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Certificate format v1: round-trip, canonical digest, tamper-evidence, schema."""

from __future__ import annotations

import pytest
from omnibias.core.proof import (
    Conjecture,
    FunctionProver,
    ProofAttempt,
    ProofMachine,
    Verdict,
)
from omnibias.core.proof.certificate import (
    CERTIFICATE_SCHEMA_VERSION,
    THEOREM_PROVER_VERIFIED_KEY,
    TRANSCEND_BACKEND_KEY,
    UNCONDITIONAL_CLAIM_KEY,
    UNCONDITIONAL_TRANSCEND_BACKENDS,
    canonical_json,
    certificate_digest,
    certificate_is_unconditional,
    certificate_transcend_backend,
    decode_interval,
    decode_taylor_model,
    encode_interval,
    encode_taylor_model,
    interval_certificate,
    make_certificate,
    schema_errors_v1,
    seal_certificate,
    taylor_model_certificate,
    verify_certificate_digest,
)
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.taylor_model import TaylorModel


def test_interval_roundtrip_is_bit_exact() -> None:
    iv = Interval(0.1, 0.1 + 1e-16)  # endpoints not exactly representable in decimal
    again = decode_interval(encode_interval(iv))
    assert again.lo == iv.lo
    assert again.hi == iv.hi


def test_taylor_model_roundtrip_is_bit_exact() -> None:
    tm = TaylorModel(
        0.3,
        0.25,
        [Interval(1.0, 1.0), Interval(-0.5, -0.4999999999999999), Interval(0.0, 1e-17)],
        Interval(-1e-12, 1e-12),
    )
    again = decode_taylor_model(encode_taylor_model(tm))
    assert again.center == tm.center
    assert again.radius == tm.radius
    assert again.order == tm.order
    assert [(c.lo, c.hi) for c in again.coeffs] == [(c.lo, c.hi) for c in tm.coeffs]
    assert (again.remainder.lo, again.remainder.hi) == (tm.remainder.lo, tm.remainder.hi)


def test_canonical_json_is_order_independent() -> None:
    a = {"b": 1, "a": {"y": 2, "x": 3}}
    b = {"a": {"x": 3, "y": 2}, "b": 1}
    assert canonical_json(a) == canonical_json(b)


def test_seal_and_verify_digest() -> None:
    cert = interval_certificate("x in [0,1]", Interval(0.0, 1.0))
    assert cert["schema_version"] == CERTIFICATE_SCHEMA_VERSION
    assert cert["digest"].startswith("sha256:")
    assert verify_certificate_digest(cert)
    assert schema_errors_v1(cert) == []


def test_tampering_invalidates_digest() -> None:
    cert = interval_certificate("x in [0,1]", Interval(0.0, 1.0))
    tampered = dict(cert)
    # Widen the claimed bound -- the digest must no longer match.
    tampered["payload"] = dict(cert["payload"])
    tampered["payload"]["interval"] = encode_interval(Interval(0.0, 2.0))
    assert not verify_certificate_digest(tampered)
    assert "digest mismatch (tampered or stale certificate)" in schema_errors_v1(tampered)


def test_digest_is_stable_under_reserialization() -> None:
    tm = TaylorModel(0.0, 1.0, [Interval(2.0, 2.0), Interval(3.0, 3.0)], Interval(-0.1, 0.1))
    cert = taylor_model_certificate("f enclosure", tm)
    # Recomputing the digest from the body must reproduce the sealed digest.
    assert certificate_digest(cert) == cert["digest"]
    # Re-sealing is idempotent.
    assert seal_certificate(cert)["digest"] == cert["digest"]


def test_schema_errors_for_missing_fields() -> None:
    errs = schema_errors_v1({"schema_version": "1.0"})
    assert any("claim" in e for e in errs)
    assert any("payload" in e for e in errs)
    assert any("digest" in e for e in errs)
    assert schema_errors_v1({"schema_version": "9.9", "claim": "x", "payload": {}, "honesty": {}, "digest": "sha256:0"})


def test_honesty_defaults_to_no_unproven_claim() -> None:
    cert = make_certificate(claim="benign", payload={"type": "interval"})
    assert cert["honesty"] == {"unproven_claim": False}


def test_reserved_honesty_key_is_rejected_by_make_certificate() -> None:
    # theorem_prover_verified is earned only by the Lean kernel / ProofMachine;
    # a producer must never seal it into the honesty body (even as False).
    with pytest.raises(ValueError, match="theorem_prover_verified"):
        make_certificate(
            claim="forged formal claim",
            payload={"type": "interval"},
            honesty={"theorem_prover_verified": True},
        )
    with pytest.raises(ValueError, match="reserved key"):
        make_certificate(
            claim="still reserved when False",
            payload={"type": "interval"},
            honesty={"unproven_claim": False, "theorem_prover_verified": False},
        )
    sealed = make_certificate(claim="honest", payload={"type": "interval"})
    assert "theorem_prover_verified" not in sealed["honesty"]
    assert THEOREM_PROVER_VERIFIED_KEY not in sealed
    # Hand-built / already-sealed bodies that smuggle the key fail schema too.
    forged = seal_certificate(
        {
            "schema_version": CERTIFICATE_SCHEMA_VERSION,
            "claim": "hand-built",
            "payload": {"type": "interval"},
            "honesty": {"unproven_claim": False, "theorem_prover_verified": True},
        }
    )
    assert any("theorem_prover_verified" in e for e in schema_errors_v1(forged))


def test_verdict_has_new_fields_with_defaults() -> None:
    v = Verdict(
        status="PROVED",
        conjecture="c",
        kind="k",
        prover="p",
        certificate=None,
        obligations=(),
        schema_ok=True,
        replay_ok=None,
        honesty_ok=True,
    )
    assert v.certificate_schema_version is None
    assert v.theorem_prover_verified is False
    assert v.summary()["theorem_prover_verified"] is False


def test_machine_records_certificate_schema_version() -> None:
    cert = interval_certificate("x in [0,1]", Interval(0.0, 1.0))

    def prove(_c: Conjecture) -> ProofAttempt:
        return ProofAttempt(status="PROVED", certificate=cert)

    machine = ProofMachine().register(
        FunctionProver(name="iv", kinds=frozenset({"iv_demo"}), prove_fn=prove)
    )
    verdict = machine.evaluate(Conjecture(name="demo", kind="iv_demo"))
    assert verdict.status == "PROVED"
    assert verdict.certificate_schema_version == CERTIFICATE_SCHEMA_VERSION
    assert verdict.theorem_prover_verified is False


# --------------------------------------------------------------------------- #
# Transcendental-backend provenance (audit item: conditional soundness).
# --------------------------------------------------------------------------- #
def test_every_certificate_records_its_transcendental_backend() -> None:
    # Stamped centrally, so no individual producer can forget the provenance.
    for cert in (
        make_certificate(claim="c", payload={"type": "interval"}),
        interval_certificate("x in [0,1]", Interval(0.0, 1.0)),
        taylor_model_certificate(
            "tm",
            TaylorModel(0.0, 0.25, [Interval(1.0, 1.0), Interval(0.5, 0.5)], Interval(-1e-12, 1e-12)),
        ),
    ):
        recorded = certificate_transcend_backend(cert)
        assert isinstance(recorded, str) and recorded
        assert cert["meta"][TRANSCEND_BACKEND_KEY] == recorded
        assert not schema_errors_v1(cert)


def test_backend_stamp_is_inside_the_sealed_body() -> None:
    # Provenance must be tamper-evident: editing the stamp breaks the digest.
    # Upgrade whatever this machine recorded to a *different* value, so the test
    # is meaningful with or without mpmath installed.
    cert = interval_certificate("x in [0,1]", Interval(0.0, 1.0))
    assert verify_certificate_digest(cert)
    recorded = certificate_transcend_backend(cert)
    upgraded = "mpmath" if recorded != "mpmath" else "arb"
    forged = {**cert, "meta": {**cert["meta"], TRANSCEND_BACKEND_KEY: upgraded}}
    assert not verify_certificate_digest(forged)
    assert any("digest" in e for e in schema_errors_v1(forged))


def test_caller_supplied_backend_provenance_is_preserved() -> None:
    # Re-sealing a decoded certificate must keep the original machine's backend,
    # not silently acquire the re-sealing machine's.
    cert = make_certificate(
        claim="c", payload={"type": "interval"}, meta={TRANSCEND_BACKEND_KEY: "arb"}
    )
    assert certificate_transcend_backend(cert) == "arb"
    assert certificate_is_unconditional(cert)


def test_unknown_provenance_is_treated_as_conditional() -> None:
    # A hand-built or pre-stamp certificate must never pass for rigorous.
    for cert in ({}, {"meta": {}}, {"meta": "not-a-mapping"}, {"meta": {TRANSCEND_BACKEND_KEY: 7}}):
        assert certificate_transcend_backend(cert) is None
        assert not certificate_is_unconditional(cert)
    assert not certificate_is_unconditional({"meta": {TRANSCEND_BACKEND_KEY: "libm"}})
    assert not certificate_is_unconditional(
        {"meta": {TRANSCEND_BACKEND_KEY: "libm_fallback"}}
    )
    for good in UNCONDITIONAL_TRANSCEND_BACKENDS:
        assert certificate_is_unconditional({"meta": {TRANSCEND_BACKEND_KEY: good}})


def test_unconditional_claim_is_refused_at_seal_time_on_conditional_backend() -> None:
    # Seal-time gate: cannot mint an unconditional certificate on libm_fallback.
    with pytest.raises(RuntimeError, match="libm_fallback|rigorous transcendental"):
        make_certificate(
            claim="rests on no libm assumption",
            payload={"type": "interval"},
            honesty={"unproven_claim": False, UNCONDITIONAL_CLAIM_KEY: True},
            meta={TRANSCEND_BACKEND_KEY: "libm_fallback"},
        )
    with pytest.raises(RuntimeError, match="libm_fallback|rigorous transcendental"):
        make_certificate(
            claim="rests on no libm assumption",
            payload={"type": "interval"},
            honesty={"unproven_claim": False, UNCONDITIONAL_CLAIM_KEY: True},
            meta={TRANSCEND_BACKEND_KEY: "libm"},  # legacy alias
        )
    # The same assertion on a rigorous backend is accepted.
    honest = make_certificate(
        claim="rests on no libm assumption",
        payload={"type": "interval"},
        honesty={"unproven_claim": False, UNCONDITIONAL_CLAIM_KEY: True},
        meta={TRANSCEND_BACKEND_KEY: "mpmath"},
    )
    assert not schema_errors_v1(honest)
    # Not asserting the claim is always fine, whatever the backend.
    quiet = make_certificate(
        claim="q",
        payload={"type": "interval"},
        meta={TRANSCEND_BACKEND_KEY: "libm_fallback"},
    )
    assert not schema_errors_v1(quiet)


def test_schema_errors_catch_hand_built_unconditional_on_libm() -> None:
    # Already-sealed / hand-built certificates still fail schema validation.
    from omnibias.core.proof.certificate import seal_certificate

    hand = seal_certificate(
        {
            "schema_version": CERTIFICATE_SCHEMA_VERSION,
            "claim": "forged unconditional",
            "payload": {"type": "interval"},
            "honesty": {"unproven_claim": False, UNCONDITIONAL_CLAIM_KEY: True},
            "meta": {TRANSCEND_BACKEND_KEY: "libm_fallback"},
        }
    )
    errors = schema_errors_v1(hand)
    assert any(UNCONDITIONAL_CLAIM_KEY in e and "libm" in e for e in errors), errors


def test_certificate_cannot_silently_use_libm_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Certificate path records libm_fallback and refuses unconditional sealing."""
    from omnibias.core.verified import transcend

    monkeypatch.setattr(transcend, "_mpmath", lambda: None)
    transcend.clear_libm_fallback_used()
    assert transcend.backend_name() == "libm_fallback"

    # Exploratory use is allowed and marks the sticky flag.
    enc = transcend.exp_iv(Interval.point(0.0))
    assert enc.contains(1.0)
    assert transcend.libm_fallback_used()

    # Sealing records the conditional backend honestly.
    cert = make_certificate(claim="exploratory", payload={"type": "interval"})
    assert certificate_transcend_backend(cert) == "libm_fallback"
    assert not certificate_is_unconditional(cert)

    # Unconditional / strict sealing cannot silently proceed on the fallback.
    with pytest.raises(RuntimeError):
        make_certificate(
            claim="must be rigorous",
            payload={"type": "interval"},
            honesty={"unproven_claim": False, UNCONDITIONAL_CLAIM_KEY: True},
        )
    prev = transcend.set_strict_backend(True)
    try:
        with pytest.raises(RuntimeError):
            make_certificate(claim="strict", payload={"type": "interval"})
    finally:
        transcend.set_strict_backend(prev)
