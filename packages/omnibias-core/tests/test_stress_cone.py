# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Theory 07-10: admissible stress cone membership."""

from __future__ import annotations

from fractions import Fraction

from omnibias.core.proof.lean_check import generate_obligation
from omnibias.core.proof.obligations.stress_cone import (
    PAYLOAD_CONE,
    check_cone,
    honesty_payload,
    locked_interior_cone,
    opposite_cone_query,
    parallel_generators_cone,
    replay_cone_certificate,
    seal_cone_certificate,
)


def test_g1_locked_cone_discharges() -> None:
    query = locked_interior_cone()
    report = check_cone(query)
    assert report.holds
    assert report.strength == "PROVED"
    assert report.lambda1 == Fraction(1)
    assert report.lambda2 == Fraction(1)
    assert report.reason == "interior"


def test_g2_parallel_and_opposite_are_blocked_and_named() -> None:
    parallel = check_cone(parallel_generators_cone())
    assert parallel.holds is False
    assert parallel.strength == "BLOCKED"
    assert parallel.reason == "degenerate_generators"
    opposite = check_cone(opposite_cone_query())
    assert opposite.holds is False
    assert opposite.reason == "opposite_cone"


def test_g4_seal_digest_replay_no_parent_flag() -> None:
    sealed = seal_cone_certificate(locked_interior_cone(), run_lean=False)
    assert sealed.mathlib_verified is False
    assert sealed.theorem_prover_verified is False
    assert sealed.certificate["payload"]["type"] == PAYLOAD_CONE
    assert sealed.certificate["honesty"]["navier_stokes_proof_claim"] is False
    assert sealed.certificate["honesty"]["forced_blowup_reproof_claim"] is False
    assert replay_cone_certificate(sealed.certificate) is True
    tampered = dict(sealed.certificate)
    payload = dict(tampered["payload"])
    payload["holds"] = False
    tampered["payload"] = payload
    from omnibias.core.proof.certificate import verify_certificate_digest

    assert verify_certificate_digest(sealed.certificate)
    assert verify_certificate_digest(tampered) is False


def test_g5_kernel_emits_allratlt() -> None:
    src = generate_obligation(
        seal_cone_certificate(locked_interior_cone(), run_lean=False).certificate
    )
    assert src is not None
    assert "allRatLt" in src
    assert "import Omnibias.RationalStencil" in src
    failed = generate_obligation(
        seal_cone_certificate(parallel_generators_cone(), run_lean=False).certificate
    )
    assert failed is None


def test_honesty_flags_stay_false() -> None:
    flags = honesty_payload()
    assert flags["navier_stokes_proof_claim"] is False
    assert flags["forced_blowup_reproof_claim"] is False
