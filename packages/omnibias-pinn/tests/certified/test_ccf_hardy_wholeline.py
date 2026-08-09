# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Hardy whole-line CAP + dissipation + spectrum + pipeline smoke tests."""

from __future__ import annotations

import numpy as np
import pytest

from omnibias.pinn.certified.ccf_hardy import (
    certified_ccf_hardy_wholeline_blowup_attempt,
    certified_ccf_hardy_wholeline_blowup_attempt_schema_errors,
    refine_ccf_hardy_profile,
)
from omnibias.pinn.certified.dissipation_threshold import (
    certified_fractional_dissipation_threshold,
    verify_fractional_dissipation_threshold,
)
from omnibias.pinn.certified.machine import build_default_machine
from omnibias.core.proof import Conjecture


def test_hardy_wholeline_cap_schema_and_honesty() -> None:
    cert = certified_ccf_hardy_wholeline_blowup_attempt(
        coeffs=[1.0, -0.2, 0.05],
        scales=[0.7, 1.4, 2.2],
        lam=0.6,
        residual_gate=1e-6,
    )
    assert certified_ccf_hardy_wholeline_blowup_attempt_schema_errors(cert) == []
    assert cert["honesty"]["navier_stokes_proof_claim"] is False
    assert cert["three_d_claim"] is False
    # Unrefined random candidate should typically not close whole-line.
    if cert["honesty"]["whole_line_certified"]:
        assert cert["closure_certified"] is True
    else:
        assert "quantified_gap" in cert["closure_report"]


def test_refine_hardy_reduces_collocation_residual() -> None:
    refined = refine_ccf_hardy_profile(
        coeffs=[1.0, -0.3, 0.1],
        scales=[0.6, 1.3, 2.1],
        lam=0.6,
        iters=40,
    )
    assert refined["residual_max_abs"] < 1.0
    assert refined["alpha"] == pytest.approx(1.0 / (1.0 + refined["lam"]))


def test_dissipation_threshold_closes_and_replays() -> None:
    # lambda_2 = 0.4703 => alpha_crit >= 1/1.4703 ≈ 0.6801
    cert = certified_fractional_dissipation_threshold(
        lambda_lo=0.47, lambda_hi=0.4703, alpha_claimed=0.69
    )
    assert cert["threshold_closed"] is True
    assert cert["honesty"]["navier_stokes_proof_claim"] is False
    report = verify_fractional_dissipation_threshold(cert)
    assert report["replay_match"] is True
    assert "digest" in cert["certificate"]


def test_machine_registers_hardy_and_dissipation_kinds() -> None:
    machine = build_default_machine()
    kinds = {k for p in machine.provers for k in getattr(p, "kinds", ())}
    assert "ccf_hardy_wholeline_blowup" in kinds
    assert "ccf_fractional_dissipation" in kinds
    verdict = machine.evaluate(
        Conjecture(
            name="alpha threshold",
            kind="ccf_fractional_dissipation",
            data={"lambda_lo": 0.47, "lambda_hi": 0.4703, "alpha_claimed": 0.69},
        )
    )
    assert verdict.status == "PROVED"


def test_sealed_spectrum_hardy() -> None:
    from omnibias.pinn.jax.discovery.spectrum import sealed_ccf_unstable_mode_count

    out = sealed_ccf_unstable_mode_count(
        coeffs=[1.0, -0.2, 0.05],
        scales=[0.7, 1.4, 2.2],
        lam=0.6,
        claimed_order=1,
    )
    assert out["honesty"]["navier_stokes_proof_claim"] is False
    assert "eigenvalue_enclosures" in out
    assert "truncation_hides_no_unstable_modes" in out


def test_pipeline_smoke() -> None:
    from omnibias.pinn.jax.discovery.pipeline import (
        CCFHardyAdapter,
        PipelineConfig,
        run_singularity_pipeline,
    )

    result = run_singularity_pipeline(
        CCFHardyAdapter(n_terms=3, n_grid=24, steps=8),
        PipelineConfig(seed=0),
    )
    assert result.honesty["navier_stokes_proof_claim"] is False
    assert result.certificate["schema_version"].startswith("navier-stokes-ccf-hardy")
    assert np.isfinite(result.discovery["max_abs_residual"])
