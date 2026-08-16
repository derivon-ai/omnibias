# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Synthetic Lotka-Volterra recovery and offline public-CSV gates."""

from __future__ import annotations

import numpy as np

from examples.symbolic_discovery.public_csv_discovery.discover import (
    CSV_PATH,
    INTERPOLANT_QUALITY_RATIO,
    evaluate_public_csv,
    evaluate_synthetic,
    load_lynx_hare,
    spline_values_and_deriv,
)


def test_cubic_spline_derivative_is_exact_on_a_line() -> None:
    t = np.linspace(0.0, 2.0, 21)
    _, deriv = spline_values_and_deriv(t, 3.0 * t + 1.0, t)
    np.testing.assert_allclose(deriv, np.full_like(t, 3.0), rtol=1e-10, atol=1e-10)


def test_cubic_spline_derivative_beats_fd_on_a_cubic() -> None:
    t = np.linspace(0.0, 2.0, 21)
    values = t**3
    _, deriv = spline_values_and_deriv(t, values, t)
    true = 3.0 * t**2
    fd = np.gradient(values, t)
    spline_rmse = float(np.sqrt(np.mean((deriv - true) ** 2)))
    fd_rmse = float(np.sqrt(np.mean((fd - true) ** 2)))
    assert spline_rmse < fd_rmse


def test_synthetic_recovers_xy_signs_and_rollout() -> None:
    result = evaluate_synthetic(hidden=48, n=41, seed=0)
    assert result["xy_signs_ok"] is True
    assert result["hare"]["xy"] < 0.0
    assert result["lynx"]["xy"] > 0.0
    assert result["interpolant"] == "spline"
    assert result["spline_dot_rmse"] <= INTERPOLANT_QUALITY_RATIO * result["fd_dot_rmse"]
    assert result["rollout_vs_linear"] > 0.0
    assert result["gates"]["all_passed"] is True


def test_public_csv_loads_offline_and_passes_rollout_gates() -> None:
    table = load_lynx_hare()
    assert CSV_PATH.is_file()
    assert table["year"][0] == 1900.0
    assert table["hare"].shape[0] == 21
    result = evaluate_public_csv(hidden=48, seed=0)
    assert result["source"] == "hudson_bay_lynx_hare"
    assert "stan-dev" in result["provenance"]["url"]
    assert result["rollout_vs_zero"] > 0.0
    assert result["rollout_vs_linear"] > 0.0
    assert result["gates"]["all_passed"] is True
    assert result["honesty"].startswith("Spline-interpolant STLSQ")
    assert result["xy_signs_ok"] is True
