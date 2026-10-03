# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Published precision examples retain an independent, high-precision oracle."""

import hashlib
import json
from pathlib import Path

import mpmath as mp
import pytest


def test_published_precision_reference_is_independent_and_errors_are_visible():
    root = Path(__file__).resolve().parents[1]
    artifact = json.loads((root / "docs/benchmarks/precision_sweep.json").read_text())
    assert artifact["schema"] == "omnibias/precision-sweep/v1"
    assert artifact["config"]["orders"] == [12, 16]
    assert artifact["config"]["points"] == [0.1, 3.0]
    assert len(artifact["results"]) == 4
    for path, digest in artifact["source_sha256"].items():
        assert hashlib.sha256((root / path).read_bytes()).hexdigest() == digest
    for row in artifact["results"]:
        with mp.workdps(100):
            expected = float(mp.diff(mp.tanh, mp.mpf(row["point"]), row["order"]))
        assert row["reference"] == expected
    difficult = next(row for row in artifact["results"]
                     if row["point"] == 3.0 and row["order"] == 16)
    for name in ("torch_closed_form", "jax_closed_form"):
        result = difficult["methods"][name]
        error = abs(result["value"] - difficult["reference"]) / abs(difficult["reference"])
        assert result["relative_error"] == pytest.approx(error, rel=1e-14)
        assert error > 1e-8
    assert difficult["methods"]["jax_taylor_ad"]["relative_error"] < 1e-10
