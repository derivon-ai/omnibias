# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Synthetic Lotka-Volterra recovery and offline public-CSV gates."""

from __future__ import annotations

from examples.symbolic_discovery.public_csv_discovery.discover import (
    CSV_PATH,
    evaluate_public_csv,
    evaluate_synthetic,
    load_lynx_hare,
)


def test_synthetic_recovers_xy_signs() -> None:
    result = evaluate_synthetic(hidden=48, n=41, seed=0)
    assert result["xy_signs_ok"] is True
    assert result["hare"]["xy"] < 0.0
    assert result["lynx"]["xy"] > 0.0
    assert result["gates"]["all_passed"] is True


def test_public_csv_loads_offline_and_passes_gates() -> None:
    table = load_lynx_hare()
    assert CSV_PATH.is_file()
    assert table["year"][0] == 1900.0
    assert table["hare"].shape[0] == 21
    result = evaluate_public_csv(hidden=48, seed=0)
    assert result["source"] == "hudson_bay_lynx_hare"
    assert "stan-dev" in result["provenance"]["url"]
    assert result["gates"]["all_passed"] is True
    assert result["honesty"].startswith("Recovers a famous ecological ODE")
