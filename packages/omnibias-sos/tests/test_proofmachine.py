# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""SOS provers behind the omnibias ProofMachine front door.

A PROVED verdict must clear every gate: the sealed v1 schema + digest, an
independent interval-LDL^T replay of the exact Gram, and the honesty gate.  A
forged unproven-result assertion is always downgraded to BLOCKED, and an inconclusive SOS
attempt never yields PROVED.
"""

from __future__ import annotations

from omnibias.core.proof import Conjecture
from omnibias.sos.auxiliary import energy_conserving_triad_system, energy_observable
from omnibias.sos.honesty import GALERKIN_TRUNCATION, SOSScope
from omnibias.sos.problem import Polynomial
from omnibias.sos.proofmachine import (
    SOS_GLOBAL_NONNEG,
    SOS_NONNEG_ON_SET,
    SOS_TIME_AVERAGE_BOUND,
    build_sos_machine,
    replay_sos_certificate,
)


def _x_y() -> tuple[Polynomial, Polynomial, Polynomial]:
    return Polynomial.variable(0, 2), Polynomial.variable(1, 2), Polynomial.constant(1.0, 2)


def test_global_sos_conjecture_is_proved_and_replayed() -> None:
    x, y, one = _x_y()
    poly = x * x - x * y + y * y + one  # strictly positive => SOS
    machine = build_sos_machine()
    verdict = machine.evaluate(Conjecture(name="x^2 - xy + y^2 + 1 >= 0", kind=SOS_GLOBAL_NONNEG, data={"polynomial": poly}))
    assert verdict.proved
    assert verdict.schema_ok
    assert verdict.replay_ok is True
    assert verdict.honesty_ok
    assert verdict.certificate is not None
    assert verdict.certificate["honesty"]["unproven_claim"] is False


def test_motzkin_conjecture_is_blocked_not_proved() -> None:
    # Motzkin: nonnegative but NOT a sum of squares -> the prover must not claim PROVED.
    x, y, _one = _x_y()
    motzkin = (
        x * x * y * y * y * y
        + x * x * x * x * y * y
        - Polynomial.constant(3.0, 2) * x * x * y * y
        + Polynomial.constant(1.0, 2)
    )
    machine = build_sos_machine()
    verdict = machine.evaluate(Conjecture(name="Motzkin", kind=SOS_GLOBAL_NONNEG, data={"polynomial": motzkin}))
    assert verdict.blocked
    assert not verdict.proved


def test_forged_unproven_claim_is_blocked_by_honesty_gate() -> None:
    x, y, one = _x_y()
    poly = x * x + y * y + one
    machine = build_sos_machine()
    verdict = machine.evaluate(
        Conjecture(
            name="forged unproven-result assertion",
            kind=SOS_GLOBAL_NONNEG,
            data={"polynomial": poly},
            claims={"unproven_claim": True},
        )
    )
    assert verdict.blocked
    assert not verdict.honesty_ok


def test_positivstellensatz_conjecture_is_proved() -> None:
    # 2 - x^2 - y^2 >= 0 on the closed unit disk {1 - x^2 - y^2 >= 0}: strictly positive.
    x, y, one = _x_y()
    g = one - x * x - y * y
    poly = Polynomial.constant(2.0, 2) - x * x - y * y
    machine = build_sos_machine()
    verdict = machine.evaluate(
        Conjecture(
            name="2 - x^2 - y^2 >= 0 on the unit disk",
            kind=SOS_NONNEG_ON_SET,
            data={"polynomial": poly, "constraints": [g]},
        )
    )
    assert verdict.proved
    assert verdict.replay_ok is True


def test_time_average_bound_conjecture_is_proved() -> None:
    system = energy_conserving_triad_system(viscosities=[1.0] * 3, forcing=[1.0, 0.0, 0.0])
    phi = energy_observable(3)
    scope = SOSScope(GALERKIN_TRUNCATION, truncation_order=3, system="triad")
    machine = build_sos_machine()
    verdict = machine.evaluate(
        Conjecture(
            name="mean energy of the Galerkin triad is bounded",
            kind=SOS_TIME_AVERAGE_BOUND,
            data={"system": system, "observable": phi, "scope": scope},
        )
    )
    assert verdict.proved
    assert verdict.replay_ok is True
    assert verdict.certificate["meta"]["sos"]["scope"] == GALERKIN_TRUNCATION


def test_tampered_gram_fails_independent_replay() -> None:
    x, y, one = _x_y()
    poly = x * x - x * y + y * y + one
    machine = build_sos_machine()
    verdict = machine.evaluate(Conjecture(name="tamper test", kind=SOS_GLOBAL_NONNEG, data={"polynomial": poly}))
    assert verdict.certificate is not None
    assert replay_sos_certificate(verdict.certificate) is True

    # Corrupt a Gram entry so it no longer factorises to the sealed pivots.
    tampered = {**verdict.certificate, "meta": {**verdict.certificate["meta"]}}
    tampered["meta"]["sos"] = {**tampered["meta"]["sos"]}
    gram = [list(row) for row in tampered["meta"]["sos"]["gram"]]
    gram[0][0] = "-1"  # break positive definiteness
    tampered["meta"]["sos"]["gram"] = gram
    assert replay_sos_certificate(tampered) is False
