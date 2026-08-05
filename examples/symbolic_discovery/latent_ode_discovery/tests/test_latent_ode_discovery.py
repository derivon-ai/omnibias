# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Tests for the latent-ODE discovery demo."""

from __future__ import annotations

from examples.symbolic_discovery.latent_ode_discovery.benchmark import evaluate_benchmark


def test_benchmark_recovers_oscillator_spectra_from_one_coordinate() -> None:
    results = evaluate_benchmark()

    undamped = results["undamped"]
    assert undamped["frequency_abs_error"] < 5e-2
    assert abs(undamped["recovered_growth_rate"]) < 5e-2
    assert undamped["reconstruction_rmse"] < 1e-6

    damped = results["damped"]
    assert damped["frequency_abs_error"] < 5e-2
    assert damped["growth_rate_abs_error"] < 5e-2
    assert "diffeomorphism" in results["honesty_note"]
