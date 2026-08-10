# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Tests for pipeline adapters, Phase 5 gates, and Adam-forbid doctrine."""

from __future__ import annotations

import numpy as np
import pytest


def test_ccf_hardy_adapter_forbids_adam() -> None:
    from omnibias.pinn.jax.discovery.pipeline import CCFHardyAdapter

    ad = CCFHardyAdapter(n_scales=2, n_gamma_multiples=1, n_grid=17, steps=2)
    with pytest.raises(ValueError, match="forbids Adam"):
        ad.discover(seed=0, optimizer="adam")


def test_ccf_hardy_adapter_martens_grosse_smoke() -> None:
    from omnibias.pinn.jax.discovery.pipeline import (
        CCF_RUNG1_RESIDUAL_GATE,
        CCFHardyAdapter,
        PipelineConfig,
        run_singularity_pipeline,
    )

    ad = CCFHardyAdapter(n_scales=2, n_gamma_multiples=2, n_grid=17, steps=3, y_max=8.0)
    out = run_singularity_pipeline(
        ad, PipelineConfig(seed=0, residual_gate=CCF_RUNG1_RESIDUAL_GATE)
    )
    assert out.discovery["optimizer"] == "martens_grosse_gn"
    assert out.discovery["train_hilbert"] == "hardy_exact_omega"
    assert out.discovery["gn_solver"] == "qr"
    assert out.honesty["navier_stokes_proof_claim"] is False
    assert "coeffs" in out.discovery
    assert out.certificate["honesty"]["navier_stokes_proof_claim"] is False
    # Earn-path discovery extra must lock exact Hardy Hilbert + QR MG.
    result = out.discovery["result"]
    assert result.extra["train_hilbert"] == "hardy_exact_omega"
    assert result.extra["gn_solver"] == "qr"
    assert result.extra["hilbert_convention"] == "hardy_exact_omega"


def test_ipm_boussinesq_adapters_smoke() -> None:
    from omnibias.pinn.jax.discovery.pipeline import (
        BoussinesqAdapter,
        IPMAdapter,
        run_singularity_pipeline,
    )

    ipm = run_singularity_pipeline(IPMAdapter(n=8, steps=5), None)
    assert ipm.certificate["honesty"]["navier_stokes_proof_claim"] is False
    assert np.isfinite(ipm.discovery["max_abs_residual"])

    bq = run_singularity_pipeline(BoussinesqAdapter(n=8, steps=5), None)
    assert bq.certificate["honesty"]["lambda_n_hypothesis_is_theorem"] is False
    assert bq.honesty["navier_stokes_proof_claim"] is False


def test_phase5_blocked_without_rung2() -> None:
    from omnibias.pinn.jax.discovery import phase5_beyond as p5

    gate = p5.phase5_entry_from_status({"gates": {"rung2_earned": False}})
    assert gate.allowed is False
    blocked = p5.blocked_phase5_bundle("rung2_not_earned")
    assert blocked["earned"] is False
    assert blocked["honesty"]["navier_stokes_proof_claim"] is False


def test_phase5_gate_helpers() -> None:
    from omnibias.pinn.jax.discovery import phase5_beyond as p5

    part = p5.partitioned_near_far_residual_report(
        single_patch_residual=1e-2,
        partitioned_residual=1e-3,
        residual_threshold=1e-2,
    )
    assert part["skill"] > 0.0
    assert part["earned"] is True

    feats = p5.ansatz_router_meta_features(
        family="ccf", order=1, residual_scale=1e-3, n_scales=8, gauge_ok=True
    )
    assert feats.shape == (5,)

    rout = p5.router_skill_report(
        fixed_schedule_failed_ticks=10, router_failed_ticks=4
    )
    assert rout["earned"] is True

    obl = p5.obligation_planner_report(
        hand_ordered_calls=5,
        planned_calls=4,
        obligations_cleared=5,
        obligations_total=5,
    )
    assert obl["earned"] is True
    assert "residual_margin" in obl["atoms"]
    assert obl["honesty"]["continuum_literals_forbidden"] is True
