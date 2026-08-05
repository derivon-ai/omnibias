# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Phase-2 proof-carrying fluid dynamics: 3-D fixtures, rigorous interval
streamfunction residuals, genuine time-integration rollout, between-node spectral
bounds, method-independent replay, and the digest gate."""

from __future__ import annotations

import copy

import numpy as np
import pytest
from omnibias.core.proof import Conjecture, ProofMachine
from omnibias.core.proof.certificate import make_certificate, verify_certificate_digest
from omnibias.core.verified.pde_certificate import (
    certified_streamfunction_divergence,
    certified_vorticity_transport_residual,
)
from omnibias.pinn.certified import (
    NS_ROLLOUT_DIAGNOSTICS_SCHEMA_VERSION,
    NS_STREAMFUNCTION_RESIDUAL_SCHEMA_VERSION,
    beltrami_abc_flow,
    build_default_machine,
    cellular_streamfunction,
    certified_periodic_flow_residual,
    certified_periodic_flow_residual_schema_errors,
    certified_rollout_diagnostics,
    certified_shear_streamfunction_residual,
    certified_streamfunction_residual,
    fourier_mode_vorticity,
    integrate_vorticity_2d,
    periodic_residual_digest_ok,
    rollout_diagnostics_schema_errors,
    shear_streamfunction,
    streamfunction_residual_schema_errors,
)


@pytest.fixture
def machine() -> ProofMachine:
    return build_default_machine()


# --------------------------------------------------------------------------- #
# WS1: exact 3-D Beltrami / ABC flow.
# --------------------------------------------------------------------------- #
def test_beltrami_abc_flow_is_exact_3d_solution() -> None:
    sample = beltrami_abc_flow(16, viscosity=0.05, wavenumber=1, time=0.3)
    assert sample.dimension == 3
    assert sample.grid_shape == (16, 16, 16)
    cert = certified_periodic_flow_residual(sample)
    assert cert["dimension"] == 3
    assert cert["momentum_residual_sup"] < 1e-10
    assert cert["continuity_residual_sup"] < 1e-10
    assert cert["pressure_poisson_residual_sup"] < 1e-10
    assert cert["exact_solution_claim"] is True
    assert certified_periodic_flow_residual_schema_errors(cert) == []


def test_beltrami_decays_and_replays() -> None:
    from omnibias.symbolic.fluid import verify_periodic_flow_residual

    e = [
        certified_periodic_flow_residual(beltrami_abc_flow(16, viscosity=0.1, time=t))["kinetic_energy"]
        for t in (0.0, 0.5, 1.0)
    ]
    assert e[0] > e[1] > e[2] > 0.0  # viscous decay
    cert = certified_periodic_flow_residual(beltrami_abc_flow(16, viscosity=0.1, time=0.2))
    rep = verify_periodic_flow_residual(cert)
    assert rep["replay_match"] is True
    assert rep["unproven_claim"] is False


# --------------------------------------------------------------------------- #
# WS3 + WS7: spectral l1 between-node bound + tamper-evident digest.
# --------------------------------------------------------------------------- #
def test_spectral_l1_bound_dominates_on_node_sup() -> None:
    cert = certified_periodic_flow_residual(beltrami_abc_flow(16, viscosity=0.05))
    assert cert["spectral_l1_residual_bound"] >= cert["residual_sup"] - 1e-12
    # resolved band-limited field: negligible spectral tail
    assert cert["velocity_spectral_tail_ratio"] < 1e-6


def test_digest_detects_tampering() -> None:
    cert = certified_periodic_flow_residual(beltrami_abc_flow(16, viscosity=0.05))
    assert periodic_residual_digest_ok(cert) is True
    assert certified_periodic_flow_residual_schema_errors(cert) == []
    tampered = copy.deepcopy(cert)
    tampered["residual_sup"] = 1e-30
    assert periodic_residual_digest_ok(tampered) is False
    assert any("digest" in e for e in certified_periodic_flow_residual_schema_errors(tampered))


# --------------------------------------------------------------------------- #
# WS4: methodologically-independent (finite-difference) replay.
# --------------------------------------------------------------------------- #
def test_finite_difference_replay_is_independent() -> None:
    from omnibias.symbolic.fluid import verify_periodic_flow_residual

    cert = certified_periodic_flow_residual(beltrami_abc_flow(16, viscosity=0.1))
    rep = verify_periodic_flow_residual(cert)
    assert rep["methodologically_independent"] is True
    assert rep["finite_difference_consistent"] is True
    assert "finite_difference_momentum_residual_sup" in rep["finite_difference"]


# --------------------------------------------------------------------------- #
# WS2: rigorous interval streamfunction residual (the centerpiece).
# --------------------------------------------------------------------------- #
def test_core_shear_streamfunction_euler_residual_is_machine_zero() -> None:
    layers = shear_streamfunction(seed=0).layers
    domain = ((0.0, 2.0 * np.pi), (0.0, 2.0 * np.pi))
    residual = certified_vorticity_transport_residual(layers, domain, viscosity=0.0, splits=4)
    divergence = certified_streamfunction_divergence(layers, domain, splits=4)
    assert residual.mag < 1e-12
    assert divergence.mag < 1e-12


def test_shear_streamfunction_certificate_is_interval_verified_exact() -> None:
    cert = certified_shear_streamfunction_residual(splits=4)
    payload = cert["payload"]
    assert payload["schema_version"] == NS_STREAMFUNCTION_RESIDUAL_SCHEMA_VERSION
    assert payload["residual_sup"] < 1e-10
    assert payload["divergence_sup"] < 1e-10
    assert payload["exact_steady_euler_claim"] is True
    assert cert["honesty"]["interval_verified"] is True
    assert cert["honesty"]["incompressible_by_construction"] is True
    assert cert["honesty"]["unproven_claim"] is False
    assert verify_certificate_digest(cert) is True
    assert streamfunction_residual_schema_errors(cert) == []


def test_streamfunction_subdivision_tightens_the_bound() -> None:
    field = cellular_streamfunction(seed=1)
    coarse = certified_streamfunction_residual(field, splits=2)["payload"]["residual_sup"]
    fine = certified_streamfunction_residual(field, splits=8)["payload"]["residual_sup"]
    # certified enclosures are nested: finer splits never loosen the bound
    assert fine <= coarse + 1e-12
    # a general (non-shear) field is not an exact steady state
    assert certified_streamfunction_residual(field, splits=4)["payload"]["exact_steady_euler_claim"] is False


def test_streamfunction_twin_catches_forged_small_residual() -> None:
    from omnibias.symbolic.fluid import verify_streamfunction_residual

    cert = certified_streamfunction_residual(cellular_streamfunction(seed=1), splits=4)
    assert verify_streamfunction_residual(cert)["replay_match"] is True
    forged = make_certificate(
        claim=cert["claim"],
        payload={**cert["payload"], "residual_sup": 1e-14, "exact_steady_euler_claim": True},
        honesty={**cert["honesty"], "exact_steady_euler_claim": True},
        meta=cert["meta"],
    )
    assert verify_streamfunction_residual(forged)["replay_match"] is False


def test_streamfunction_schema_requires_interval_verified() -> None:
    cert = certified_shear_streamfunction_residual(splits=3)
    bad = make_certificate(
        claim=cert["claim"],
        payload=dict(cert["payload"]),
        honesty={**cert["honesty"], "interval_verified": False},
        meta=cert["meta"],
    )
    assert any("interval_verified" in e for e in streamfunction_residual_schema_errors(bad))


def test_viscous_streamfunction_residual_uses_order_four() -> None:
    cert = certified_streamfunction_residual(shear_streamfunction(seed=0), viscosity=0.05, splits=4)
    assert cert["payload"]["jet_order"] == 4
    assert cert["payload"]["model"] == "incompressible_navier_stokes"
    # unforced viscous shear is NOT an exact steady NS state
    assert cert["payload"]["exact_steady_euler_claim"] is False


# --------------------------------------------------------------------------- #
# WS5: genuine pseudo-spectral rollout + statistical diagnostics.
# --------------------------------------------------------------------------- #
def _modes() -> list[dict[str, float]]:
    return [
        {"kx": 1, "ky": 0, "amp": 1.0, "phase": 0.0},
        {"kx": 0, "ky": 2, "amp": 0.7, "phase": 0.3},
        {"kx": 1, "ky": 1, "amp": 0.5, "phase": 1.1},
    ]


def test_inviscid_rollout_conserves_energy_and_enstrophy() -> None:
    omega0, _ = fourier_mode_vorticity(48, _modes())
    result = integrate_vorticity_2d(omega0, viscosity=0.0, dt=2e-3, steps=200)
    assert result.max_divergence < 1e-12  # incompressible by construction
    assert result.energy_drift < 1e-4  # low numerical diffusion over the window
    assert result.enstrophy_drift < 1e-4


def test_rollout_certificate_and_independent_reintegration(machine: ProofMachine) -> None:
    from omnibias.symbolic.fluid import verify_rollout_diagnostics

    _, desc = fourier_mode_vorticity(48, _modes())
    cert = certified_rollout_diagnostics(desc, viscosity=0.0, dt=2e-3, steps=200, drift_tol=1e-2)
    assert cert["schema_version"] == NS_ROLLOUT_DIAGNOSTICS_SCHEMA_VERSION
    assert rollout_diagnostics_schema_errors(cert) == []
    assert cert["incompressibility_maintained"] is True
    assert cert["conserved_invariant_drift_bounded"] is True
    rep = verify_rollout_diagnostics(cert)
    assert rep["replay_match"] is True
    assert rep["methodologically_independent"] is True


def test_viscous_rollout_dissipates_energy() -> None:
    _, desc = fourier_mode_vorticity(48, _modes())
    cert = certified_rollout_diagnostics(desc, viscosity=0.02, dt=2e-3, steps=150)
    assert cert["energy_final"] < cert["energy_initial"]
    assert cert["conservative_case"] is False


# --------------------------------------------------------------------------- #
# Proof-machine integration for the two new kinds + honesty gate.
# --------------------------------------------------------------------------- #
def test_machine_proves_streamfunction_and_rollout(machine: ProofMachine) -> None:
    v1 = machine.evaluate(
        Conjecture(name="exact steady Euler shear", kind="navier_stokes_streamfunction_residual",
                   data={"kind": "shear", "splits": 4})
    )
    assert v1.status == "PROVED"
    assert v1.schema_ok and v1.replay_ok and v1.honesty_ok

    v2 = machine.evaluate(
        Conjecture(name="2D NS conservative rollout", kind="navier_stokes_rollout_diagnostics",
                   data={"n": 48, "viscosity": 0.0, "dt": 2e-3, "steps": 150, "drift_tol": 1e-2})
    )
    assert v2.status == "PROVED"
    assert v2.replay_ok is True


def test_machine_blocks_forged_unproven_claim_on_new_kinds(machine: ProofMachine) -> None:
    for kind, data in (
        ("navier_stokes_streamfunction_residual", {"kind": "shear"}),
        ("navier_stokes_rollout_diagnostics", {"n": 48, "steps": 100}),
    ):
        verdict = machine.evaluate(
            Conjecture(name="forged unproven", kind=kind, data=data, claims={"unproven_claim": True})
        )
        assert verdict.status == "BLOCKED"
        assert verdict.honesty_ok is False
