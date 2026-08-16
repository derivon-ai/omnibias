# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Sealed finite-gauge report: connect engines, measure G1, no Clay claim."""

from __future__ import annotations

import pytest
from omnibias.core.proof import Conjecture, seal_certificate
from omnibias.core.proof.certificate import verify_certificate_digest
from omnibias.geometry.gauge.proofmachine import build_gauge_machine
from omnibias.geometry.gauge.transfer.certificates import (
    FINITE_GAUGE_REPORT_KIND,
    FINITE_GAUGE_REPORT_SCHEMA_VERSION,
    finite_gauge_report_schema_errors,
    replay_finite_gauge_report,
    seal_finite_gauge_report_certificate,
)
from omnibias.geometry.gauge.transfer.report import (
    FiniteGaugeReport,
    FiniteGaugeSpec,
    finite_gauge_report,
    finite_gauge_spec_from_mapping,
    finite_gauge_spec_to_mapping,
)


@pytest.fixture(scope="module")
def report() -> FiniteGaugeReport:
    return finite_gauge_report()


def test_default_pack_certifies_and_measures_g1(report: FiniteGaugeReport) -> None:
    assert report.certified is True
    assert report.continuum_claim is False
    assert report.yang_mills_claim is False
    assert report.scaling.continuum_claim is False
    assert report.haar.certified is True
    assert report.su3_gap.dimension == 4
    assert report.g1.ge_generic is True
    assert report.g1.factor + 1e-12 >= 1.0
    assert "measured" in report.g1.note
    assert "5x" in report.g1.note
    assert report.torus_rp is None
    assert len(report.polymer) == 2
    assert all(item.certified for item in report.polymer)
    assert report.wilson_character.spectral_gap_lower > report.polymer[0].spectral_gap_lower


def test_spec_round_trips() -> None:
    spec = FiniteGaugeSpec(name="round-trip")
    again = finite_gauge_spec_from_mapping(finite_gauge_spec_to_mapping(spec))
    assert again.name == spec.name
    assert again.polymer_beta == spec.polymer_beta
    assert again.include_torus is False
    assert again.polymer_countings == spec.polymer_countings


def test_seal_replay_and_honesty_flags(report: FiniteGaugeReport) -> None:
    cert = seal_finite_gauge_report_certificate(report)
    assert cert["schema_version"] == FINITE_GAUGE_REPORT_SCHEMA_VERSION
    assert cert["continuum_claim"] is False
    assert cert["honesty"]["yang_mills_claim"] is False
    assert cert["g1_target_5x"] is False
    assert cert["g1_ge_generic"] is True
    assert finite_gauge_report_schema_errors(cert) == []
    assert verify_certificate_digest(cert)
    assert replay_finite_gauge_report(cert) is True


def test_schema_rejects_continuum_and_yang_mills_flags(report: FiniteGaugeReport) -> None:
    cert = dict(seal_finite_gauge_report_certificate(report))
    cert["continuum_claim"] = True
    errors = finite_gauge_report_schema_errors(cert)
    assert any("continuum_claim must be False" in item for item in errors)
    cert = dict(seal_finite_gauge_report_certificate(report))
    honesty = dict(cert["honesty"])
    honesty["yang_mills_claim"] = True
    cert["honesty"] = honesty
    errors = finite_gauge_report_schema_errors(cert)
    assert any("yang_mills_claim must be False" in item for item in errors)


def test_forged_tighter_hamiltonian_gap_fails_replay(report: FiniteGaugeReport) -> None:
    forged = dict(seal_finite_gauge_report_certificate(report))
    forged.pop("digest")
    forged["hamiltonian_gap"] = float(report.hamiltonian.spectral_gap_lower) + 1.0
    resealed = dict(seal_certificate(forged))
    assert verify_certificate_digest(resealed)
    assert replay_finite_gauge_report(resealed) is False


def test_proofmachine_proves_the_pack_and_blocks_yang_mills() -> None:
    machine = build_gauge_machine()
    proved = machine.evaluate(Conjecture("pack", FINITE_GAUGE_REPORT_KIND, {}))
    assert proved.status == "PROVED"
    assert proved.schema_ok is True
    assert proved.replay_ok is True
    assert proved.honesty_ok is True
    blocked = machine.evaluate(
        Conjecture(
            "overclaim",
            FINITE_GAUGE_REPORT_KIND,
            {},
            claims={"yang_mills_claim": True},
        )
    )
    assert blocked.status == "BLOCKED"
    assert blocked.honesty_ok is False
