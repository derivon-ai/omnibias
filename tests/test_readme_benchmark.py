# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Public performance claims must stay attached to reproducible evidence."""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "docs/benchmarks/readme_derivatives.json"


def test_committed_measurement_retains_baselines_accuracy_and_compile_cost():
    payload = json.loads(ARTIFACT.read_text())
    assert payload["config"]["reference_points"] >= 9
    assert "mpmath" in payload["config"]["reference"]
    assert [row["order"] for row in payload["rows"]] == [2, 4, 6]
    methods = {"torch_nested_ad", "torch_omnibias", "jax_nested_ad", "jax_taylor_ad", "jax_omnibias"}
    for row in payload["rows"]:
        assert set(row["methods"]) == methods
        for name, result in row["methods"].items():
            samples = result["samples_ms"]
            assert len(samples) == payload["config"]["repeats"]
            assert all(math.isfinite(t) and t > 0 for t in samples)
            assert result["median_ms"] == statistics.median(samples)
            assert result["first_execution_ms"] > 0
            assert result["error_vs_mpmath"]["max_abs"] <= payload["config"]["max_abs_error_cap"]
            if name.startswith("jax_"):
                assert result["compile_ms"] > 0
            else:
                assert result["compile_ms"] is None


def test_deep_jet_guide_and_chart_match_the_measurement():
    payload = json.loads(ARTIFACT.read_text())
    methods = next(row["methods"] for row in payload["rows"] if row["order"] == 6)
    expected = round(methods["torch_nested_ad"]["median_ms"] / methods["torch_omnibias"]["median_ms"], 1)
    readme = (ROOT / "docs/performance.md").read_text().split(
        "## General deep-network jets\n", 1
    )[1]
    claim = re.search(r"\*\*([0-9.]+)× faster than nested PyTorch autograd\*\*", readme)
    assert claim is not None
    assert float(claim.group(1)) == pytest.approx(expected)
    assert hashlib.sha256(ARTIFACT.read_bytes()).hexdigest() in (
        ROOT / "docs/img/derivative-benchmark.svg"
    ).read_text()
    columns = ["torch_nested_ad", "torch_omnibias", "jax_nested_ad", "jax_taylor_ad", "jax_omnibias"]
    for row in payload["rows"]:
        table_line = next(line for line in readme.splitlines() if line.startswith(f"| {row['order']} |"))
        cells = [cell.strip().replace("**", "") for cell in table_line.split("|")[2:-1]]
        assert cells == [f"{row['methods'][name]['median_ms']:.3f} ms" for name in columns]


def test_benchmark_sources_match_the_recorded_provenance():
    payload = json.loads(ARTIFACT.read_text())
    expected_kind = "base_commit" if payload["source_worktree_dirty"] else "exact_commit"
    assert payload["source_revision_kind"] == expected_kind
    for name, digest in payload["source_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, (
            f"{name} changed: refresh the public measurement, chart and README claims together"
        )
