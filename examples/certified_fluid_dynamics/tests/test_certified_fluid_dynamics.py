# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Smoke tests for the certified fluid-dynamics example."""

from __future__ import annotations

from examples.certified_fluid_dynamics.benchmark import evaluate_benchmark


def test_certified_fluid_dynamics_benchmark() -> None:
    result = evaluate_benchmark(n=32)
    assert result["all_proved"] is True
    assert result["all_replayed"] is True
    assert result["rollout_residual_drift_free"] is True
    for case in result["cases"]:
        assert case["verdict"] == "PROVED"
        assert case["schema_ok"] is True
        assert case["replay_ok"] is True
        assert case["honesty_ok"] is True
        assert case["unproven_claim"] is False
        assert case["chaotic_tracking_claim"] is False
        assert case["interval_verified"] is False
        assert case["residual_sup"] < 1e-8


def test_certified_fluid_dynamics_rollout_is_drift_free() -> None:
    result = evaluate_benchmark(n=32, rollout_times=(0.0, 1.0, 4.0, 8.0))
    times = [snap["time"] for snap in result["taylor_green_rollout"]]
    assert times == [0.0, 1.0, 4.0, 8.0]
    for snap in result["taylor_green_rollout"]:
        assert snap["verdict"] == "PROVED"
        assert snap["residual_sup"] < 1e-8


def test_certified_fluid_dynamics_scratch_dir(tmp_path) -> None:
    out = tmp_path / "fluid"
    result = evaluate_benchmark(n=16, scratch_dir=str(out))
    saved = result["saved_artifacts"]
    assert "arrays" in saved and "descriptor" in saved
    assert (out / "periodic_flow_sample.npz").exists()
    assert (out / "periodic_flow_descriptor.json").exists()
