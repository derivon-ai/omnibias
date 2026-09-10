# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Theory 07-08: finite rational convergence ledgers."""

from __future__ import annotations

from fractions import Fraction

import pytest
from omnibias.core.proof.certificate import make_certificate, verify_certificate_digest
from omnibias.core.proof.lean_check import generate_obligation, lean_check_available
from omnibias.core.proof.obligations.convergence_ledger import (
    NS_CD_PARENT,
    NS_PARENT,
    PARENT_CLAIM_KEYS,
    PAYLOAD_LEDGER,
    AffineForm,
    ConvergenceLedger,
    MarginObligation,
    MinForm,
    SideCondition,
    StageMap,
    check_ledger,
    curated_convergence_ledgers,
    empty_premise_cd_ledger,
    empty_premise_discharged_ledger,
    failing_margin_ledger,
    honesty_payload,
    ledger_obligation,
    ledger_to_inequality_system,
    navier_stokes_exponent_ledger,
    parent_earns_navier_stokes_claim,
    replay_ledger_certificate,
    seal_ledger_certificate,
    stage_invariant,
    strong_coupling_polymer_ledger,
    unbounded_slope_ledger,
)


def _assert_fraction_tree(obj: object) -> None:
    if isinstance(obj, Fraction):
        return
    if isinstance(obj, bool) or obj is None:
        return
    if isinstance(obj, int):
        return
    if isinstance(obj, str):
        return
    if isinstance(obj, tuple | list):
        for item in obj:
            _assert_fraction_tree(item)
        return
    raise AssertionError(f"non-exact value of type {type(obj).__name__}: {obj!r}")


def test_exactness_no_floats() -> None:
    for ledger in curated_convergence_ledgers():
        report = check_ledger(ledger)
        for row in report.residuals:
            _assert_fraction_tree(row.residual)
            _assert_fraction_tree(row.slope)
            if row.tipping is not None:
                _assert_fraction_tree(row.tipping[2])
        if report.binding_threshold is not None:
            _assert_fraction_tree(report.binding_threshold[2])


def test_ns_ledger_discharges_and_recovers_kappa_threshold() -> None:
    ledger = navier_stokes_exponent_ledger()
    report = check_ledger(ledger)
    assert report.holds
    assert report.strength == "CONDITIONAL"
    assert report.stage_invariant
    names = {obl.name for obl in ledger.obligations}
    assert names == {
        "particular",
        "signed",
        "mean_update_wave",
        "completed_mean",
        "completed_defect",
        "signed_bar",
    }
    assert report.failing == ()
    assert report.binding_threshold is not None
    param, sense, value = report.binding_threshold
    assert param == "kappa"
    assert sense == "lt"
    assert value == Fraction(1, 200)
    flags = honesty_payload(ledger, report)
    assert flags["navier_stokes_proof_claim"] is False
    assert flags["yang_mills_mass_gap_claim"] is False
    assert ledger.external_premises


def test_ab_architecture_ledger_stays_conditional_and_refuses_stamp() -> None:
    from omnibias.core.proof.obligations.convergence_ledger import (
        NS_AB_EXTERNAL_PREMISES,
        NS_EXTERNAL_PREMISES,
        navier_stokes_ab_architecture_ledger,
    )

    ledger = navier_stokes_ab_architecture_ledger()
    report = check_ledger(ledger)
    assert report.holds
    assert report.strength == "CONDITIONAL"
    assert ledger.parent == NS_PARENT
    assert ledger.external_premises == NS_AB_EXTERNAL_PREMISES
    assert NS_AB_EXTERNAL_PREMISES
    joined = " ".join(NS_AB_EXTERNAL_PREMISES).lower()
    assert "bridge" in joined
    assert "taylor-green" in joined or "taylor–green" in joined
    assert "three-dimensional" in joined
    assert "infinity" in joined
    assert "majorant" in joined
    flags = honesty_payload(ledger, report)
    assert flags["navier_stokes_proof_claim"] is False
    assert stage_invariant(ledger)
    sealed = seal_ledger_certificate(ledger, run_lean=False)
    assert sealed.certificate["honesty"]["navier_stokes_proof_claim"] is False
    with pytest.raises(ValueError, match="navier_stokes_proof_claim"):
        make_certificate(
            claim="forged A/B architecture",
            payload=ledger_obligation(ledger).payload,
            honesty={"navier_stokes_proof_claim": True},
        )
    # C/D exponent premises stay the construction spine; emptying them
    # is not an A/B earn path.
    assert NS_EXTERNAL_PREMISES
    assert "analytic classes of the slow base" in NS_EXTERNAL_PREMISES[0]


def test_scale_ledger_discharges_with_binding_h() -> None:
    from omnibias.core.proof.obligations.convergence_ledger import (
        navier_stokes_scale_ledger,
    )

    ledger = navier_stokes_scale_ledger()
    report = check_ledger(ledger)
    assert report.holds
    assert report.strength == "CONDITIONAL"
    assert report.binding_threshold is not None
    param, sense, value = report.binding_threshold
    assert param == "h"
    assert sense == "lt"
    assert value == Fraction(1, 100)
    flags = honesty_payload(ledger, report)
    assert flags["navier_stokes_proof_claim"] is False
    assert ledger.external_premises


def test_polymer_ledger_matches_check_polymer() -> None:
    ledger = strong_coupling_polymer_ledger()
    report = check_ledger(ledger)
    assert report.holds
    assert report.strength == "CONDITIONAL"
    names = {obl.name for obl in ledger.obligations}
    assert names == {"backtrack_lt_first_step", "backtrack_lt_crude"}
    by_name = {row.obligation: row for row in report.residuals if row.binding}
    # residual = 15 - 20 and 15 - 24, matching Check/Polymer.lean.
    assert by_name["backtrack_lt_first_step"].residual == Fraction(-5)
    assert by_name["backtrack_lt_crude"].residual == Fraction(-9)
    flags = honesty_payload(ledger, report)
    assert flags["yang_mills_mass_gap_claim"] is False
    assert flags["navier_stokes_proof_claim"] is False


def test_failing_margin_is_blocked_and_named() -> None:
    ledger = failing_margin_ledger()
    report = check_ledger(ledger)
    assert report.holds is False
    assert report.strength == "BLOCKED"
    assert "particular" in report.failing


def test_unbounded_interval_decided_by_slope() -> None:
    good = check_ledger(unbounded_slope_ledger(holds=True))
    bad = check_ledger(unbounded_slope_ledger(holds=False))
    assert good.holds
    assert good.strength == "PROVED"
    assert bad.holds is False
    assert bad.strength == "BLOCKED"
    assert any(row.slope > 0 for row in bad.residuals)


def test_open_upper_bound_uses_endpoint_value() -> None:
    """``s - 1 < 0`` on ``[0, hi)`` holds iff ``hi <= 1``, not merely because hi is open."""

    def ledger(hi: Fraction) -> ConvergenceLedger:
        return ConvergenceLedger(
            name="open_hi",
            stage=StageMap(var="s", step=Fraction(0), initial=Fraction(0)),
            parameters={},
            side_conditions=(
                SideCondition(AffineForm.variable("s"), "ge", Fraction(0), name="lo"),
                SideCondition(AffineForm.variable("s"), "lt", hi, name="hi"),
            ),
            obligations=(
                MarginObligation(
                    "residual",
                    AffineForm.constant(0),
                    MinForm.singleton(AffineForm.constant(1)),
                    next_quantity=AffineForm.variable("s"),
                ),
            ),
            parent="",
            external_premises=(),
        )

    assert check_ledger(ledger(Fraction(1))).holds
    assert check_ledger(ledger(Fraction(2))).holds is False


def _empty_premise_ledger(*, parent: str) -> ConvergenceLedger:
    return ConvergenceLedger(
        name="empty_premise_control",
        stage=StageMap(var="s", step=Fraction(1), initial=Fraction(0)),
        parameters={},
        side_conditions=(
            SideCondition(AffineForm.variable("s"), "ge", Fraction(0), name="s_nonneg"),
        ),
        obligations=(
            MarginObligation(
                "constant_gain",
                AffineForm.variable("s"),
                MinForm.singleton(AffineForm.constant(2)),
            ),
        ),
        parent=parent,
        external_premises=(),
    )


def test_empty_premises_earn_parent_flag() -> None:
    ledger = empty_premise_discharged_ledger()
    report = check_ledger(ledger)
    assert report.holds
    assert report.strength == "PROVED"
    assert ledger.parent == NS_PARENT
    sealed = seal_ledger_certificate(ledger, run_lean=False)
    assert sealed.certificate["honesty"]["navier_stokes_proof_claim"] is True
    assert sealed.certificate["honesty"]["yang_mills_mass_gap_claim"] is False


def test_cd_empty_premises_do_not_earn_ab_flag() -> None:
    ledger = empty_premise_cd_ledger()
    report = check_ledger(ledger)
    assert report.holds
    assert report.strength == "PROVED"
    assert parent_earns_navier_stokes_claim(ledger.parent) is False
    flags = honesty_payload(ledger, report)
    assert flags["navier_stokes_proof_claim"] is False
    sealed = seal_ledger_certificate(ledger, run_lean=False)
    assert sealed.certificate["honesty"]["navier_stokes_proof_claim"] is False
    with pytest.raises(ValueError, match="navier_stokes_proof_claim"):
        make_certificate(
            claim="forged C/D as A/B",
            payload=ledger_obligation(ledger).payload,
            honesty={"navier_stokes_proof_claim": True},
        )


def test_navier_or_euler_substring_does_not_earn() -> None:
    for parent in (
        "finite-time singularity of 3D Euler / Navier-Stokes",
        "Navier-Stokes forced blowup (Clay C/D)",
        "some euler paper",
    ):
        ledger = _empty_premise_ledger(parent=parent)
        report = check_ledger(ledger)
        assert report.holds
        assert honesty_payload(ledger, report)["navier_stokes_proof_claim"] is False


def test_cannot_forge_parent_flag_on_nonempty_premises() -> None:
    ledger = navier_stokes_exponent_ledger()
    obligation = ledger_obligation(ledger)
    with pytest.raises(ValueError, match="navier_stokes_proof_claim"):
        make_certificate(
            claim="forged",
            payload=obligation.payload,
            honesty={"navier_stokes_proof_claim": True},
        )
    sealed = seal_ledger_certificate(ledger, run_lean=False)
    assert sealed.certificate["honesty"]["navier_stokes_proof_claim"] is False


def test_cannot_forge_parent_flag_on_unrelated_payload() -> None:
    with pytest.raises(ValueError, match="navier_stokes_proof_claim"):
        make_certificate(
            claim="forged",
            payload={"type": "toy"},
            honesty={"navier_stokes_proof_claim": True},
        )
    with pytest.raises(ValueError, match="yang_mills_mass_gap_claim"):
        make_certificate(
            claim="forged",
            payload={"type": "toy"},
            honesty={"yang_mills_mass_gap_claim": True},
        )
    for key in PARENT_CLAIM_KEYS:
        cert = make_certificate(
            claim="plain",
            payload={"type": "toy"},
            honesty={"unproven_claim": False},
        )
        assert cert["honesty"].get(key, False) is False


def test_seal_digest_and_replay() -> None:
    ledger = navier_stokes_exponent_ledger()
    report = seal_ledger_certificate(ledger, run_lean=False)
    assert report.obligation.kind == PAYLOAD_LEDGER
    assert report.obligation.holds
    assert report.mathlib_verified is False
    assert "theorem_prover_verified" not in report.certificate.get("honesty", {})
    assert verify_certificate_digest(report.certificate)
    assert replay_ledger_certificate(report.certificate) is True
    tampered = dict(report.certificate)
    payload = dict(tampered["payload"])
    payload["holds"] = False
    tampered["payload"] = payload
    assert verify_certificate_digest(tampered) is False


def test_generate_obligation_emits_all_rat_lt() -> None:
    cert = seal_ledger_certificate(
        navier_stokes_exponent_ledger(), run_lean=False
    ).certificate
    src = generate_obligation(cert)
    assert src is not None
    assert "allRatLt" in src
    assert "import Omnibias.RationalStencil" in src
    assert "sorry" not in src.lower()
    assert "limit" not in src.lower()


def test_failing_ledger_emits_no_kernel_obligation() -> None:
    cert = seal_ledger_certificate(failing_margin_ledger(), run_lean=False).certificate
    assert generate_obligation(cert) is None


def test_inequality_system_round_trip_shape() -> None:
    system = ledger_to_inequality_system(navier_stokes_exponent_ledger())
    assert system.sort == "linear"
    assert system.data["type"] == PAYLOAD_LEDGER
    assert system.data["A"] == [["1"], ["-1"]]
    assert system.data["b"] == ["0", "0"]
    bad = ledger_to_inequality_system(failing_margin_ledger())
    assert bad.data["b"] == ["-1", "-1"]


def test_stage_invariant_on_curated() -> None:
    for ledger in curated_convergence_ledgers():
        assert stage_invariant(ledger)


def test_affine_form_hashable_and_exact() -> None:
    form = AffineForm(Fraction(1, 2), (("sigma", Fraction(1)),))
    assert form.eval({"sigma": Fraction(1, 5)}) == Fraction(7, 10)
    shifted = form.shift("sigma", Fraction(1, 10))
    assert shifted.eval({"sigma": Fraction(1, 5)}) == Fraction(4, 5)


def test_catalog_kind_is_registered() -> None:
    from omnibias.core.proof.catalog import catalog_entry, list_catalog

    kinds = {item.kind for item in list_catalog()}
    if PAYLOAD_LEDGER not in kinds:
        from omnibias.core.proof.obligations.convergence_ledger import _register

        _register()
    entry = catalog_entry(PAYLOAD_LEDGER)
    assert entry.mode == "exact_replay"
    assert entry.complete is True


def test_g3_no_toolchain_degrades() -> None:
    report = seal_ledger_certificate(navier_stokes_exponent_ledger(), run_lean=True)
    assert verify_certificate_digest(report.certificate)
    assert report.mathlib_verified is False
    if lean_check_available():
        assert report.theorem_prover_verified is True
        assert report.lean is not None and report.lean.verified is True
    else:
        assert report.theorem_prover_verified is False
        assert report.lean is not None and report.lean.verified is False
        assert report.lean.available is False
