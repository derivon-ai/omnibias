# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""The gauge gap certificate must be sealed, replayable and honest.

Three properties, each of which has failed somewhere in this repo before:

* **sealed** -- ``check_certificate`` refuses an unsealed payload before emitting
  any Lean, so a certificate carrying a kernel-checkable obligation but no digest
  can never earn ``theorem_prover_verified``;
* **replayable** -- the replay rebuilds the matrix from the recorded inputs and
  re-derives the bound, so a forged number is caught even though the arithmetic
  that produced the original was sound;
* **honest** -- ``continuum_claim`` and ``yang_mills_claim`` are hard-wired
  ``False``, and asserting either is downgraded rather than believed.
"""

from __future__ import annotations

from typing import Any

import pytest
from omnibias.core.proof import Conjecture, ProofMachine, lean_check_available
from omnibias.core.proof.certificate import verify_certificate_digest
from omnibias.core.proof.lean_check import check_certificate, generate_obligation
from omnibias.geometry.gauge.proofmachine import build_gauge_machine, gauge_provers
from omnibias.geometry.gauge.transfer.certificates import (
    STRONG_COUPLING_KIND,
    TRANSFER_GAP_KIND,
    TRANSFER_GAP_SCHEMA_VERSION,
    replay_transfer_matrix_gap,
    seal_transfer_gap_certificate,
    transfer_gap_schema_errors,
)
from omnibias.geometry.gauge.transfer.gap import certified_transfer_matrix_gap
from omnibias.geometry.gauge.transfer.matrices import su3_heat_kernel_transfer

SU3_PARAMETERS: dict[str, Any] = {
    "builder": "su3_heat_kernel_transfer",
    "coupling": "4/5",
    "max_dynkin": 2,
    "lattice_spacing": 1.0,
}


@pytest.fixture
def machine() -> ProofMachine:
    return build_gauge_machine()


def _certificate() -> dict[str, Any]:
    transfer = su3_heat_kernel_transfer(0.8, max_dynkin=2)
    result = certified_transfer_matrix_gap(transfer)
    return dict(seal_transfer_gap_certificate(result, transfer, claim="su(3) gap"))


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
def test_the_gauge_machine_registers_exactly_the_expected_kinds(
    machine: ProofMachine,
) -> None:
    """Frozen so a new prover cannot slip in unannounced."""
    assert sorted(machine.kinds()) == [STRONG_COUPLING_KIND, TRANSFER_GAP_KIND]
    assert [p.name for p in gauge_provers()] == [
        "transfer_matrix_spectral_gap",
        "strong_coupling_glueball_gap",
    ]


def test_every_gauge_prover_wires_a_schema_and_a_replay() -> None:
    for prover in gauge_provers():
        assert prover.schema_fn is not None
        assert prover.replay_fn is not None


# --------------------------------------------------------------------------- #
# the happy path
# --------------------------------------------------------------------------- #
def test_a_gap_conjecture_is_proved_with_schema_and_replay(
    machine: ProofMachine,
) -> None:
    verdict = machine.evaluate(
        Conjecture("su(3) gap", TRANSFER_GAP_KIND, {"parameters": SU3_PARAMETERS})
    )
    assert verdict.status == "PROVED"
    assert verdict.schema_ok is True
    assert verdict.replay_ok is True
    assert verdict.honesty_ok is True
    assert verdict.certificate is not None
    assert verdict.certificate["spectral_gap_lower"] == pytest.approx(4 * 0.8 / 3, rel=1e-9)


def test_the_certificate_is_clean_against_its_own_schema() -> None:
    assert transfer_gap_schema_errors(_certificate()) == []


def test_the_certificate_carries_the_expected_schema_version() -> None:
    assert _certificate()["schema_version"] == TRANSFER_GAP_SCHEMA_VERSION


# --------------------------------------------------------------------------- #
# sealed, and therefore able to reach the kernel
# --------------------------------------------------------------------------- #
def test_the_certificate_is_sealed_so_the_kernel_is_reachable() -> None:
    cert = _certificate()
    assert verify_certificate_digest(cert)
    assert generate_obligation(cert) is not None
    assert "unsealed" not in check_certificate(cert).detail


def test_the_obligation_is_the_kernels_spectral_gap_lemma() -> None:
    source = generate_obligation(_certificate())
    assert source is not None
    assert "spectral_gap_pos" in source
    assert "gapNumerator" in source


def test_the_lean_flag_mirrors_the_toolchain(machine: ProofMachine) -> None:
    """No toolchain means the flag stays ``False``, never that the check is skipped."""
    verdict = machine.evaluate(
        Conjecture("su(3) gap", TRANSFER_GAP_KIND, {"parameters": SU3_PARAMETERS}),
        lean_check=True,
    )
    assert verdict.theorem_prover_verified is lean_check_available()


# --------------------------------------------------------------------------- #
# tamper evidence
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("spectral_gap_lower", 99.0),
        ("subdominant_ratio_upper", 1e-12),
        ("dimension", 2),
        ("lattice_spacing", 0.5),
    ],
)
def test_editing_any_sealed_field_breaks_the_digest(key: str, value: object) -> None:
    forged = _certificate()
    forged[key] = value
    assert not verify_certificate_digest(forged)
    assert any("digest" in e for e in transfer_gap_schema_errors(forged))
    assert "digest mismatch" in check_certificate(forged).detail


def test_a_forged_bound_fails_the_replay_even_with_a_valid_digest() -> None:
    """The replay never reads the sealed numbers, so re-sealing a lie does not help.

    This is the property a digest alone cannot give: the digest proves nobody edited
    the file, while the replay proves the numbers were derivable in the first place.
    """
    from omnibias.core.proof import seal_certificate

    forged = _certificate()
    forged.pop("digest")
    forged["subdominant_ratio_upper"] = 1e-12  # far tighter than the matrix supports
    resealed = dict(seal_certificate(forged))
    assert verify_certificate_digest(resealed)  # the lie is internally consistent
    assert replay_transfer_matrix_gap(resealed) is False


def test_an_inflated_gap_also_fails_the_replay() -> None:
    from omnibias.core.proof import seal_certificate

    forged = _certificate()
    forged.pop("digest")
    forged["spectral_gap_lower"] = 99.0
    resealed = dict(seal_certificate(forged))
    assert replay_transfer_matrix_gap(resealed) is False


def test_a_replay_with_nothing_to_rebuild_abstains() -> None:
    cert = _certificate()
    cert["parameters"] = {}
    assert replay_transfer_matrix_gap(cert) is None


def test_a_replay_of_unbuildable_parameters_fails_rather_than_raising() -> None:
    cert = _certificate()
    cert["parameters"] = {"builder": "no_such_builder", "coupling": "1.0"}
    assert replay_transfer_matrix_gap(cert) is False


# --------------------------------------------------------------------------- #
# honesty
# --------------------------------------------------------------------------- #
def test_the_certificate_refuses_to_claim_a_continuum_limit() -> None:
    cert = _certificate()
    assert cert["continuum_claim"] is False
    assert cert["honesty"]["continuum_claim"] is False
    assert cert["honesty"]["yang_mills_claim"] is False
    assert cert["honesty"]["fixed_matrix"] is True
    assert "NOT a continuum" in cert["honesty"]["note"]


@pytest.mark.parametrize(
    "claim", ["continuum_claim", "yang_mills_claim", "unproven_claim"]
)
def test_asserting_a_claim_the_certificate_does_not_support_is_blocked(
    machine: ProofMachine, claim: str
) -> None:
    verdict = machine.evaluate(
        Conjecture(
            "overclaim",
            TRANSFER_GAP_KIND,
            {"parameters": SU3_PARAMETERS},
            claims={claim: True},
        )
    )
    assert verdict.status == "BLOCKED"
    assert verdict.honesty_ok is False


@pytest.mark.parametrize(
    ("key", "value", "match"),
    [
        ("continuum_claim", True, "continuum_claim must be False"),
        ("schema_version", "bogus-1", "schema_version must be"),
        ("subdominant_ratio_upper", 1.5, r"must lie in \[0, 1\)"),
        ("spectral_gap_lower", -1.0, "spectral_gap_lower must be > 0"),
        ("lattice_spacing", 0.0, "lattice_spacing must be > 0"),
    ],
)
def test_the_schema_rejects_a_malformed_field(
    key: str, value: object, match: str
) -> None:
    import re

    forged = _certificate()
    forged[key] = value
    errors = transfer_gap_schema_errors(forged)
    assert any(re.search(match, e) for e in errors), errors


def test_the_schema_reports_every_missing_key() -> None:
    errors = transfer_gap_schema_errors({})
    assert any("digest" in e for e in errors)
    assert any("subdominant_ratio_upper" in e for e in errors)
    assert any("honesty" in e for e in errors)


# --------------------------------------------------------------------------- #
# prover gates
# --------------------------------------------------------------------------- #
def test_a_gap_below_a_requested_threshold_is_blocked(machine: ProofMachine) -> None:
    verdict = machine.evaluate(
        Conjecture(
            "too tight",
            TRANSFER_GAP_KIND,
            {"parameters": SU3_PARAMETERS, "min_spectral_gap": 99.0},
        )
    )
    assert verdict.status == "BLOCKED"
    assert any("below the requested threshold" in o for o in verdict.obligations)


def test_a_conjecture_without_parameters_is_blocked(machine: ProofMachine) -> None:
    verdict = machine.evaluate(Conjecture("bare", TRANSFER_GAP_KIND, {}))
    assert verdict.status == "BLOCKED"
    assert any("parameters" in o for o in verdict.obligations)


def test_an_unbuildable_matrix_is_blocked(machine: ProofMachine) -> None:
    verdict = machine.evaluate(
        Conjecture(
            "nonsense",
            TRANSFER_GAP_KIND,
            {"parameters": {"builder": "no_such_builder", "coupling": "1.0"}},
        )
    )
    assert verdict.status == "BLOCKED"
    assert any("could not build" in o for o in verdict.obligations)
