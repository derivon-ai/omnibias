# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Independent operator references and provenance for published measurements."""

from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path

import mpmath as mp
import numpy as np
import pytest
from _operator_reference import tanh_field_laplacian_reference

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ("derivative_order", "laplacian_scaling", "polylaplacian_order")


def test_operator_oracle_agrees_with_direct_multivariate_differentiation():
    """Compare the ridge identity with ∂xxxx + 2∂xxyy + ∂yyyy directly."""
    points = np.array([[0.2, -0.3], [0.8, 0.1]])
    weights = np.array([[1.3, -0.7], [0.4, 0.2]])
    biases, coefficients = np.array([0.1, -0.2]), np.array([0.8, -0.6])
    actual = tanh_field_laplacian_reference(points, weights, biases, coefficients, 2)
    with mp.workdps(80):
        def field(x, y):
            return sum(mp.mpf(float(c)) * mp.tanh(
                mp.mpf(float(w[0])) * x + mp.mpf(float(w[1])) * y + mp.mpf(float(b))
            ) for w, b, c in zip(weights, biases, coefficients, strict=True))

        expected = [float(
            mp.diff(field, tuple(point), (4, 0))
            + 2 * mp.diff(field, tuple(point), (2, 2))
            + mp.diff(field, tuple(point), (0, 4))
        ) for point in points]
    np.testing.assert_allclose(actual, expected, rtol=2e-15, atol=1e-14)
    with pytest.raises(ValueError, match="positive"):
        tanh_field_laplacian_reference(points, weights, biases, coefficients, 0)


@pytest.mark.parametrize("name", ARTIFACTS)
def test_published_specialized_artifacts_match_source_and_figure(name):
    source = ROOT / "docs/benchmarks" / f"{name}.json"
    payload = json.loads(source.read_text())
    assert payload["config"]["dtype"] == "float64"
    assert "80" in payload["config"]["reference"]
    for relative, expected in payload["source_sha256"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected, relative
    figure = (ROOT / "docs/img/specialized-derivatives.svg").read_text()
    assert hashlib.sha256(source.read_bytes()).hexdigest() in figure
    for row in payload["rows"]:
        if name == "derivative_order":
            methods = row["measurements"]
        elif name == "laplacian_scaling":
            methods = row["methods"]
        else:
            methods = {method: row[method] for method in ("omnibias", "folx_nested", "dense_nested")}
        for result in methods.values():
            if result.get("status", "ok") != "ok":
                assert result["status"] in ("memory_budget", "timeout", "error")
                continue
            assert len(result["samples_ms"]) == payload["config"]["repeats"]
            assert result["median_ms"] == statistics.median(result["samples_ms"])
            assert result["first_execution_ms"] > 0
            if "compile_ms" in result and result["compile_ms"] is not None:
                assert result["compile_ms"] > 0


def test_spatial_evidence_records_runtime_arguments_and_compiler_audit():
    for name in ARTIFACTS[1:]:
        payload = json.loads((ROOT / "docs/benchmarks" / f"{name}.json").read_text())
        assert payload["config"]["runtime_arguments"] == ["X", "W", "beta", "c"]
    audit = json.loads((ROOT / "docs/benchmarks/laplacian_scaling.json").read_text())["constant_folding_audit"]
    assert audit["zero_argument_tanh_operations"] == 0
    assert "constant({...})" in audit["zero_argument_entry_hlo"]
    assert "ROOT %copy" in audit["zero_argument_entry_hlo"]
    assert audit["runtime_argument_tanh_operations"] > 0
    assert "parameter(0)" in audit["runtime_argument_entry_hlo"]


def test_readme_specialized_headlines_match_measured_workloads():
    data = {name: json.loads((ROOT / "docs/benchmarks" / f"{name}.json").read_text())
            for name in ARTIFACTS}
    activation = data["derivative_order"]["rows"][-1]["measurements"]
    laplacian = data["laplacian_scaling"]["rows"][-1]["methods"]
    repeated = data["polylaplacian_order"]["rows"]
    cases = [
        ("Activation derivative · `n = 8`", activation["closed_form"],
         activation["nested_autograd"], "Torch nested autograd"),
        ("Laplacian · `D = 60`", laplacian["omnibias"],
         laplacian["jax_hessian"], "JAX dense Hessian"),
        ("Repeated Laplacian · `Δ³`", repeated[2]["omnibias"],
         repeated[2]["dense_nested"], "JAX dense nested"),
        ("Repeated Laplacian · `Δ⁴`", repeated[3]["omnibias"],
         repeated[3]["folx_nested"], "folx nested"),
    ]
    readme = (ROOT / "README.md").read_text()
    for label, fast, slow, baseline in cases:
        ratio = slow["median_ms"] / fast["median_ms"]
        formatted = f"{ratio:,.0f}" if ratio >= 100 else f"{ratio:.1f}"
        expected = (f"| {label} | {fast['median_ms']:.4f} ms | "
                    f"{slow['median_ms']:.4f} ms · {baseline} | **{formatted}×** |")
        assert expected in readme
    assert repeated[3]["dense_nested"]["status"] == "memory_budget"
    assert "no speedup is claimed for that unfinished run" in readme
