# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Residual-only periodic Navier-Stokes certified-evidence certificate tests."""

from __future__ import annotations

import json

import numpy as np
import pytest
from omnibias.pinn.certified import (
    NS_PERIODIC_RESIDUAL_SCHEMA_VERSION,
    available_fixtures,
    certified_kolmogorov_residual,
    certified_periodic_flow_residual,
    certified_periodic_flow_residual_schema_errors,
    certified_taylor_green_residual,
    kolmogorov_flow,
    regenerate_periodic_flow,
    save_periodic_flow_sample,
    taylor_green_vortex,
)


def test_taylor_green_fixture_is_divergence_free_and_exact() -> None:
    sample = taylor_green_vortex(48, viscosity=0.1)
    assert sample.dimension == 2
    assert sample.grid_shape == (48, 48)
    # independent finite divergence on the analytic field is ~0 (solenoidal)
    assert float(np.max(np.abs(sample.forcing))) == 0.0
    cert = certified_periodic_flow_residual(sample)
    assert cert["momentum_residual_sup"] < 1e-10
    assert cert["continuity_residual_sup"] < 1e-10
    assert cert["pressure_poisson_residual_sup"] < 1e-10
    assert cert["exact_solution_claim"] is True


def test_kolmogorov_fixture_is_forced_and_exact() -> None:
    sample = kolmogorov_flow(48, viscosity=0.1, wavenumber=4)
    assert float(np.max(np.abs(sample.forcing))) > 0.0  # genuinely forced
    cert = certified_periodic_flow_residual(sample)
    assert cert["forced"] is True
    assert cert["residual_sup"] < 1e-10
    assert cert["exact_solution_claim"] is True


def test_taylor_green_decay_rollout_is_drift_free() -> None:
    for t in (0.0, 1.0, 4.0, 8.0):
        cert = certified_taylor_green_residual(48, viscosity=0.1, time=t)
        assert cert["residual_sup"] < 1e-8
        assert cert["exact_solution_claim"] is True
    # kinetic energy follows exp(-4 nu t): strictly decreasing across snapshots
    energies = [certified_taylor_green_residual(48, viscosity=0.1, time=t)["kinetic_energy"]
                for t in (0.0, 1.0, 2.0)]
    assert energies[0] > energies[1] > energies[2] > 0.0


def test_schema_accepts_well_formed_certificate() -> None:
    cert = certified_kolmogorov_residual(32, viscosity=0.1, wavenumber=4)
    assert cert["schema_version"] == NS_PERIODIC_RESIDUAL_SCHEMA_VERSION
    assert certified_periodic_flow_residual_schema_errors(cert) == []


def test_schema_rejects_forged_honesty_claims() -> None:
    cert = certified_taylor_green_residual(32, viscosity=0.1)
    for flag in (
        "unproven_claim",
        "continuum_navier_stokes_claim",
        "chaotic_tracking_claim",
        "perfect_weather_claim",
        "turbulence_closure_claim",
        "interval_verified",
    ):
        forged = json.loads(json.dumps(cert))
        forged["honesty"][flag] = True
        errors = certified_periodic_flow_residual_schema_errors(forged)
        assert any(flag in e for e in errors)


def test_schema_rejects_exact_claim_with_large_residual() -> None:
    cert = certified_taylor_green_residual(32, viscosity=0.1)
    forged = json.loads(json.dumps(cert))
    forged["momentum_residual_sup"] = 1.0
    forged["residual_sup"] = 1.0
    forged["exact_solution_claim"] = True
    errors = certified_periodic_flow_residual_schema_errors(forged)
    assert any("exact_solution_claim requires" in e for e in errors)


def test_certificate_sha256_is_deterministic_and_json_native() -> None:
    a = certified_taylor_green_residual(32, viscosity=0.1)
    b = certified_taylor_green_residual(32, viscosity=0.1)
    assert a["provenance"]["sha256"] == b["provenance"]["sha256"]
    restored = json.loads(json.dumps(a))
    assert restored["provenance"]["sha256"] == a["provenance"]["sha256"]
    assert restored["fixture"] == a["fixture"]


def test_symbolic_replay_matches() -> None:
    from omnibias.symbolic.fluid import verify_periodic_flow_residual

    for cert in (
        certified_taylor_green_residual(48, viscosity=0.1),
        certified_kolmogorov_residual(48, viscosity=0.1, wavenumber=4),
    ):
        rep = verify_periodic_flow_residual(cert)
        assert rep["replay_match"] is True
        assert rep["exact_solution_holds"] is True
        assert rep["divergence_free"] is True
        assert rep["unproven_claim"] is False


def test_symbolic_replay_catches_forged_residual_sup() -> None:
    from omnibias.symbolic.fluid import verify_periodic_flow_residual

    cert = certified_taylor_green_residual(48, viscosity=0.1)
    forged = json.loads(json.dumps(cert))
    forged["momentum_residual_sup"] = 0.5  # impossibly large vs the recomputed ~1e-13
    rep = verify_periodic_flow_residual(forged)
    assert rep["sups_match"] is False
    assert rep["replay_match"] is False


def test_regenerate_periodic_flow_round_trip() -> None:
    assert set(available_fixtures()) == {
        "taylor_green_vortex",
        "kolmogorov_flow",
        "beltrami_abc_flow",
    }
    sample = kolmogorov_flow(32, viscosity=0.2, wavenumber=3, amplitude=1.5)
    again = regenerate_periodic_flow(sample.descriptor)
    assert np.allclose(sample.velocity, again.velocity)
    assert np.allclose(sample.forcing, again.forcing)
    assert again.descriptor == sample.descriptor


def test_save_periodic_flow_sample_writes_artifacts(tmp_path) -> None:
    sample = taylor_green_vortex(16, viscosity=0.1)
    paths = save_periodic_flow_sample(sample, str(tmp_path / "fluid"))
    assert (tmp_path / "fluid" / "periodic_flow_sample.npz").exists()
    assert (tmp_path / "fluid" / "periodic_flow_descriptor.json").exists()
    with open(paths["descriptor"], encoding="utf-8") as handle:
        descriptor = json.load(handle)
    assert descriptor["name"] == "taylor_green_vortex"


def test_fixture_input_validation() -> None:
    with pytest.raises(ValueError):
        taylor_green_vortex(2, viscosity=0.1)
    with pytest.raises(ValueError):
        kolmogorov_flow(8, viscosity=0.1, wavenumber=5)  # 2k >= n
    with pytest.raises(ValueError):
        kolmogorov_flow(32, viscosity=-1.0)
    with pytest.raises(ValueError):
        certified_periodic_flow_residual(taylor_green_vortex(16, viscosity=0.1), residual_tol=0.0)
