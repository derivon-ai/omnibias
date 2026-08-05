# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Regression tests for the certificate audit soundness fixes.

These lock in the *modest, honest* guarantees the certificates actually make:

* the rigorous transcendental backend can be made mandatory (the conditionally
  rigorous libm fallback is refusable);
* even integer powers of a sign-straddling interval are non-negative and tight;
* the proof machine rejects a tampered sealed certificate, and ``strict`` mode
  additionally demands a sealed v1 envelope and an *agreeing* replay twin.

None of this "solves" a global-regularity problem -- it only makes the small claims sound.
"""

from __future__ import annotations

import pytest
from omnibias.core.proof import Conjecture, FunctionProver, ProofAttempt, ProofMachine
from omnibias.core.proof.certificate import interval_certificate
from omnibias.core.verified import transcend
from omnibias.core.verified.interval import Interval


# --------------------------------------------------------------------------- #
# transcend: rigorous backend can be made mandatory
# --------------------------------------------------------------------------- #
def test_strict_backend_toggle_roundtrips() -> None:
    prev = transcend.set_strict_backend(True)
    try:
        assert transcend.strict_backend() is True
    finally:
        restored = transcend.set_strict_backend(prev)
    assert restored is True
    assert transcend.strict_backend() is prev


def test_require_rigorous_backend_accepts_mpmath(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(transcend, "_mpmath", lambda: object())
    assert transcend.backend_name() == "mpmath"
    transcend.require_rigorous_backend()  # must not raise


def test_strict_mode_refuses_libm_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    # Simulate mpmath being unavailable so the libm fallback path is taken.
    monkeypatch.setattr(transcend, "_mpmath", lambda: None)
    assert transcend.backend_name() == "libm_fallback"
    with pytest.raises(RuntimeError):
        transcend.require_rigorous_backend()

    prev = transcend.set_strict_backend(True)
    try:
        with pytest.raises(RuntimeError):
            transcend.exp_iv(Interval.point(0.5))
        with pytest.raises(RuntimeError):
            transcend.ln_iv(Interval(0.5, 2.0))
    finally:
        transcend.set_strict_backend(prev)

    # With strict off the (conditionally rigorous) libm fallback still encloses.
    transcend.clear_libm_fallback_used()
    enc = transcend.exp_iv(Interval.point(0.0))
    assert enc.lo <= 1.0 <= enc.hi
    assert transcend.libm_fallback_used()


def test_certificate_mode_requires_mpmath(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(transcend, "_mpmath", lambda: None)
    with pytest.raises(RuntimeError):
        with transcend.certificate_mode():
            pass
    # With mpmath present the context enables strict mode for its duration.
    monkeypatch.setattr(transcend, "_mpmath", lambda: object())
    assert not transcend.strict_backend()
    with transcend.certificate_mode():
        assert transcend.strict_backend()
    assert not transcend.strict_backend()


# --------------------------------------------------------------------------- #
# interval: even powers of a sign-straddling interval are tight & non-negative
# --------------------------------------------------------------------------- #
def test_pow_int_even_straddling_zero_is_tight_and_nonnegative() -> None:
    sq = Interval(-2.0, 1.0).pow_int(2)
    # The tight enclosure of x**2 over [-2, 1] is [0, 4]. The new abs()-folded
    # path gives lo ~ 0 (a single outward-rounding ulp below zero), whereas the
    # old binary expansion produced a loose lo ~ -2. Assert "essentially zero".
    assert -1e-300 < sq.lo <= 0.0
    assert sq.hi >= 4.0
    for x in (-2.0, -1.5, -1.0, 0.0, 0.3, 1.0):
        assert sq.lo <= x * x <= sq.hi


def test_pow_int_nonstraddling_even_and_odd_remain_valid() -> None:
    pos = Interval(2.0, 3.0).pow_int(2)
    assert pos.lo <= 4.0 <= pos.hi and pos.hi >= 9.0
    neg = Interval(-3.0, -2.0).pow_int(2)
    assert neg.lo <= 4.0 <= neg.hi and neg.hi >= 9.0
    cube = Interval(-2.0, 1.0).pow_int(3)
    for x in (-2.0, -1.0, 0.0, 1.0):
        assert cube.lo <= x**3 <= cube.hi
    assert Interval(2.0, 2.0).pow_int(0).lo == 1.0


# --------------------------------------------------------------------------- #
# proof machine: digest tamper-evidence and strict adjudication
# --------------------------------------------------------------------------- #
def _sealed_prover(
    *, tamper: bool = False, replay: bool | None = True
) -> FunctionProver:
    def prove(conjecture: Conjecture) -> ProofAttempt:
        cert = interval_certificate(
            "q in [0.5, 2]", Interval(0.5, 2.0), honesty={"unproven_claim": False}
        )
        if tamper:
            cert = dict(cert)
            cert["claim"] = "TAMPERED"  # body changed; sealed digest now stale
        return ProofAttempt(status="PROVED", certificate=cert)

    replay_fn = None if replay is None else (lambda _cert: replay)
    return FunctionProver(
        name="sealed",
        kinds=frozenset({"sealed"}),
        prove_fn=prove,
        replay_fn=replay_fn,
    )


def test_machine_blocks_tampered_sealed_certificate() -> None:
    machine = ProofMachine().register(_sealed_prover(tamper=True))
    verdict = machine.evaluate(Conjecture("x", "sealed", {}))
    assert verdict.status == "BLOCKED"
    assert verdict.schema_ok is False
    assert any("digest" in o for o in verdict.obligations)


def test_machine_accepts_valid_sealed_certificate() -> None:
    machine = ProofMachine().register(_sealed_prover(tamper=False))
    verdict = machine.evaluate(Conjecture("x", "sealed", {}))
    assert verdict.status == "PROVED"
    assert verdict.schema_ok is True


def test_strict_mode_requires_sealed_cert_and_agreeing_replay() -> None:
    # Valid sealed cert + an agreeing replay twin -> PROVED under strict.
    ok = ProofMachine().register(_sealed_prover(tamper=False, replay=True))
    assert ok.evaluate(Conjecture("x", "sealed", {}), strict=True).status == "PROVED"

    # No replay decision (twin returns None) -> strict blocks.
    none_twin = ProofMachine().register(_sealed_prover(tamper=False, replay=None))
    v_none = none_twin.evaluate(Conjecture("x", "sealed", {}), strict=True)
    assert v_none.status == "BLOCKED"
    assert any("replay" in o for o in v_none.obligations)

    # An unsealed certificate cannot back a strict verdict, even with an agreeing twin.
    def prove_unsealed(_conjecture: Conjecture) -> ProofAttempt:
        return ProofAttempt(
            status="PROVED", certificate={"value": 1.0, "honesty": {"unproven_claim": False}}
        )

    unsealed = ProofMachine().register(
        FunctionProver(
            name="unsealed",
            kinds=frozenset({"sealed"}),
            prove_fn=prove_unsealed,
            replay_fn=lambda _cert: True,
        )
    )
    v_unsealed = unsealed.evaluate(Conjecture("x", "sealed", {}), strict=True)
    assert v_unsealed.status == "BLOCKED"
    assert any("digest" in o or "schema(v1)" in o for o in v_unsealed.obligations)


def test_strict_mode_does_not_change_unsealed_default_behaviour() -> None:
    # Default (non-strict) verdicts are unaffected for unsealed certs without a twin.
    def prove(_conjecture: Conjecture) -> ProofAttempt:
        return ProofAttempt(
            status="PROVED", certificate={"value": 1.0, "honesty": {"unproven_claim": False}}
        )

    machine = ProofMachine().register(
        FunctionProver(name="plain", kinds=frozenset({"plain"}), prove_fn=prove)
    )
    verdict = machine.evaluate(Conjecture("x", "plain", {}))
    assert verdict.status == "PROVED"
    assert verdict.replay_ok is None
