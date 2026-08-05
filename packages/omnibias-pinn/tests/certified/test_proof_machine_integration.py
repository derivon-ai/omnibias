# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Integration tests for the default omnibias prove/disprove machine."""

from __future__ import annotations

import pytest
from omnibias.core.proof import Conjecture, ProofMachine, lean_check_available
from omnibias.core.verified.pde_certificate import (
    aposteriori_error_certificate,
    laplace,
)
from omnibias.pinn.certified import build_default_machine
from omnibias.pinn.certified.machine import (
    PERRON_GAP_SCHEMA_VERSION,
    perron_spectral_gap_schema_errors,
)


@pytest.fixture
def machine() -> ProofMachine:
    return build_default_machine()


def test_default_machine_registers_every_kind(machine: ProofMachine) -> None:
    assert set(machine.kinds()) == {
        "clm_blowup",
        "clm_multizero_blowup",
        "ccf_selfsimilar_blowup",
        "gclm_selfsimilar_blowup",
        "gclm_gradient_amplification",
        "perron_spectral_gap",
        "pinn_aposteriori_error",
        "navier_stokes_periodic_residual",
        "navier_stokes_streamfunction_residual",
        "navier_stokes_rollout_diagnostics",
    }


def test_ccf_selfsimilar_proved_with_refine_and_replay(machine: ProofMachine) -> None:
    # refine -> certify: the radii polynomial closes (collocation profile exists).
    verdict = machine.evaluate(
        Conjecture(
            "ccf",
            "ccf_selfsimilar_blowup",
            {
                "coeffs": [1.0, -0.5, 0.3],
                "scales": [0.6, 1.3, 2.1],
                "lam": 0.6,
                "refine": True,
            },
        )
    )
    assert verdict.status == "PROVED"
    assert verdict.schema_ok is True
    assert verdict.replay_ok is True  # numpy line-Hilbert twin agrees
    assert verdict.honesty_ok is True
    assert verdict.certificate is not None
    assert verdict.certificate["closure_certified"] is True
    enc = verdict.certificate["lambda_enclosure"]
    assert enc["lower"] <= verdict.certificate["lambda_candidate"] <= enc["upper"]


def test_ccf_selfsimilar_blocks_on_generic_candidate(machine: ProofMachine) -> None:
    # A non-refined candidate cannot close; the gap is reported, replay agrees.
    verdict = machine.evaluate(
        Conjecture(
            "ccf0",
            "ccf_selfsimilar_blowup",
            {"coeffs": [1.0, 0.4, -0.2], "scales": [0.6, 1.3, 2.1], "lam": 0.5},
        )
    )
    assert verdict.status == "BLOCKED"
    assert verdict.replay_ok is True  # the failure-to-close is independently confirmed
    assert any("closure failed" in o for o in verdict.obligations)


def test_ccf_forged_unproven_claim_is_blocked(machine: ProofMachine) -> None:
    verdict = machine.evaluate(
        Conjecture(
            "ccf_forge",
            "ccf_selfsimilar_blowup",
            {
                "coeffs": [1.0, -0.5, 0.3],
                "scales": [0.6, 1.3, 2.1],
                "lam": 0.6,
                "refine": True,
            },
            claims={"unproven_claim": True},
        )
    )
    assert verdict.status == "BLOCKED"
    assert verdict.honesty_ok is False


def test_clm_multizero_proved_with_independent_replay(machine: ProofMachine) -> None:
    verdict = machine.evaluate(
        Conjecture("clm", "clm_multizero_blowup", {"coeffs": [-1.0], "scales": [1.0]})
    )
    assert verdict.status == "PROVED"
    assert verdict.schema_ok is True
    assert verdict.replay_ok is True  # numpy twin agrees
    assert verdict.honesty_ok is True
    assert verdict.certificate is not None
    assert verdict.certificate["honesty"]["unproven_claim"] is False


def test_clm_multizero_disproved_when_no_zero_has_positive_hilbert(
    machine: ProofMachine,
) -> None:
    # omega0 = +q_1 has H omega0(0) = -1 < 0 and no other zeros => no blow-up.
    verdict = machine.evaluate(
        Conjecture("noblow", "clm_multizero_blowup", {"coeffs": [1.0], "scales": [1.0]})
    )
    assert verdict.status == "DISPROVED"
    assert verdict.replay_ok is True  # the negative result is independently confirmed


def test_clm_single_point_blocks_instead_of_disproving(machine: ProofMachine) -> None:
    # The origin-only certificate cannot disprove blow-up (other zeros unchecked).
    verdict = machine.evaluate(
        Conjecture("clm1", "clm_blowup", {"coeffs": [1.0], "scales": [1.0]})
    )
    assert verdict.status == "BLOCKED"
    assert any("cannot disprove" in o for o in verdict.obligations)


def test_clm_single_point_proved(machine: ProofMachine) -> None:
    verdict = machine.evaluate(
        Conjecture("clm1", "clm_blowup", {"coeffs": [-1.0], "scales": [1.0]})
    )
    assert verdict.status == "PROVED"


def test_gclm_selfsimilar_proved(machine: ProofMachine) -> None:
    verdict = machine.evaluate(
        Conjecture("gclm", "gclm_selfsimilar_blowup", {"a": 0.5})
    )
    assert verdict.status == "PROVED"


def test_gclm_gradient_amplification_proved(machine: ProofMachine) -> None:
    verdict = machine.evaluate(
        Conjecture(
            "gclmg",
            "gclm_gradient_amplification",
            {"a": 0.5, "coeffs": [-1.0], "scales": [1.0]},
        )
    )
    assert verdict.status == "PROVED"


def test_perron_spectral_gap_proved_and_schema_clean(machine: ProofMachine) -> None:
    verdict = machine.evaluate(
        Conjecture("gap", "perron_spectral_gap", {"matrix": [[2.0, 1.0], [1.0, 2.0]]})
    )
    assert verdict.status == "PROVED"
    assert verdict.certificate is not None
    assert verdict.certificate["spectral_gap_lower"] > 0.0
    assert perron_spectral_gap_schema_errors(verdict.certificate) == []
    assert verdict.certificate["schema_version"] == PERRON_GAP_SCHEMA_VERSION


def test_perron_formal_claim_runs_through_lean_gate(machine: ProofMachine) -> None:
    # The real Birkhoff-Hopf certificate carries a rational subdominant-ratio upper
    # bound, so it is a finite Lean-checkable obligation. Asserting the formal claim
    # makes the verdict depend on the Lean kernel.
    conj = Conjecture(
        "gap",
        "perron_spectral_gap",
        {"matrix": [[2.0, 1.0], [1.0, 2.0]]},
        claims={"theorem_prover_verified": True},
    )
    verdict = machine.evaluate(conj)
    if lean_check_available():  # pragma: no cover - Lean-equipped environment only
        assert verdict.status == "PROVED"
        assert verdict.theorem_prover_verified is True
    else:
        assert verdict.status == "BLOCKED"
        assert verdict.theorem_prover_verified is False
        assert any("theorem_prover_verified" in o for o in verdict.obligations)


def test_perron_lean_check_flag_mirrors_kernel(machine: ProofMachine) -> None:
    verdict = machine.evaluate(
        Conjecture("gap", "perron_spectral_gap", {"matrix": [[2.0, 1.0], [1.0, 2.0]]}),
        lean_check=True,
    )
    # No formal claim asserted: requesting the kernel check never blocks; the flag
    # mirrors kernel availability.
    assert verdict.status == "PROVED"
    assert verdict.theorem_prover_verified is lean_check_available()


def test_perron_spectral_gap_blocks_on_nonpositive_matrix(machine: ProofMachine) -> None:
    # Birkhoff-Hopf requires an entrywise-positive matrix; a zero entry => BLOCKED.
    verdict = machine.evaluate(
        Conjecture("gap0", "perron_spectral_gap", {"matrix": [[1.0, 0.0], [1.0, 1.0]]})
    )
    assert verdict.status == "BLOCKED"
    assert any("could not build" in o for o in verdict.obligations)


def test_forged_unproven_claim_is_blocked(machine: ProofMachine) -> None:
    verdict = machine.evaluate(
        Conjecture(
            "forge",
            "clm_multizero_blowup",
            {"coeffs": [-1.0], "scales": [1.0]},
            claims={"unproven_claim": True},
        )
    )
    assert verdict.status == "BLOCKED"
    assert verdict.honesty_ok is False
    assert any("honesty gate" in o for o in verdict.obligations)


def test_unknown_kind_is_blocked(machine: ProofMachine) -> None:
    verdict = machine.evaluate(Conjecture("x", "no_such_kind", {}))
    assert verdict.status == "BLOCKED"
    assert verdict.prover == "<none>"


def test_supported_honest_claim_passes(machine: ProofMachine) -> None:
    # interval_verified IS a flag the CLM certificate genuinely sets to True.
    verdict = machine.evaluate(
        Conjecture(
            "clm",
            "clm_multizero_blowup",
            {"coeffs": [-1.0], "scales": [1.0]},
            claims={"interval_verified": True},
        )
    )
    assert verdict.status == "PROVED"
    assert verdict.honesty_ok is True


def test_pinn_aposteriori_certificate_proved_with_replay(machine: ProofMachine) -> None:
    layers = [([[2.0, -3.0]], [1.0], None)]
    cert = aposteriori_error_certificate(
        layers, [(-1.0, 1.0), (-1.0, 1.0)], laplace(2), max_error=1e-6, splits=2
    ).certificate
    verdict = machine.evaluate(
        Conjecture(
            "harmonic affine",
            "pinn_aposteriori_error",
            {"certificate": cert, "max_error": 1e-6},
        )
    )
    assert verdict.status == "PROVED"
    assert verdict.schema_ok is True
    assert verdict.replay_ok is True
    assert verdict.honesty_ok is True
    assert verdict.certificate is not None
    assert verdict.certificate["honesty"]["unproven_claim"] is False


def test_pinn_aposteriori_blocks_forged_unproven_claim(machine: ProofMachine) -> None:
    layers = [([[2.0, -3.0]], [1.0], None)]
    cert = aposteriori_error_certificate(
        layers, [(-1.0, 1.0), (-1.0, 1.0)], laplace(2), splits=2
    ).certificate
    verdict = machine.evaluate(
        Conjecture(
            "forge pde",
            "pinn_aposteriori_error",
            {"certificate": cert},
            claims={"unproven_claim": True},
        )
    )
    assert verdict.status == "BLOCKED"
    assert verdict.honesty_ok is False


def test_pinn_aposteriori_blocks_threshold_miss(machine: ProofMachine) -> None:
    layers = [([[2.0, -3.0]], [1.0], None)]
    cert = aposteriori_error_certificate(
        layers,
        [(-1.0, 1.0), (-1.0, 1.0)],
        laplace(2),
        stability_interior=1.0,
        max_error=0.0,
        splits=2,
    ).certificate
    # Tamper-free but impossible strict threshold once rounded margin is checked by
    # the prover threshold gate.
    verdict = machine.evaluate(
        Conjecture(
            "too tight",
            "pinn_aposteriori_error",
            {"certificate": cert, "max_error": -1.0},
        )
    )
    assert verdict.status == "BLOCKED"


@pytest.mark.parametrize(
    "data",
    [
        {"name": "taylor_green_vortex", "n": 32, "viscosity": 0.1},
        {"name": "kolmogorov_flow", "n": 32, "viscosity": 0.1, "wavenumber": 4},
    ],
)
def test_navier_stokes_periodic_residual_proved_with_replay(
    machine: ProofMachine, data: dict
) -> None:
    verdict = machine.evaluate(
        Conjecture("periodic NS residual", "navier_stokes_periodic_residual", data)
    )
    assert verdict.status == "PROVED"
    assert verdict.schema_ok is True
    assert verdict.replay_ok is True  # numpy spectral twin agrees
    assert verdict.honesty_ok is True
    assert verdict.certificate is not None
    assert verdict.certificate["honesty"]["unproven_claim"] is False
    assert verdict.certificate["honesty"]["interval_verified"] is False
    assert verdict.certificate["exact_solution_claim"] is True


def test_navier_stokes_periodic_residual_blocks_forged_chaos_claim(
    machine: ProofMachine,
) -> None:
    verdict = machine.evaluate(
        Conjecture(
            "forged chaos",
            "navier_stokes_periodic_residual",
            {"name": "kolmogorov_flow", "n": 32, "viscosity": 0.1, "wavenumber": 4},
            claims={"chaotic_tracking_claim": True},
        )
    )
    assert verdict.status == "BLOCKED"
    assert verdict.honesty_ok is False


def test_navier_stokes_periodic_residual_blocks_tolerance_miss(
    machine: ProofMachine,
) -> None:
    verdict = machine.evaluate(
        Conjecture(
            "too tight",
            "navier_stokes_periodic_residual",
            {
                "name": "taylor_green_vortex",
                "n": 32,
                "viscosity": 0.1,
                "residual_tolerance": 1e-20,
            },
        )
    )
    assert verdict.status == "BLOCKED"
    assert any("exceeds tolerance" in o for o in verdict.obligations)
