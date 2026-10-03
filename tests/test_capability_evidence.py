# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Agent evidence follows measured workloads and never turns failures into wins."""

from __future__ import annotations

import copy
import importlib.util
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "generate_capability_evidence", ROOT / "scripts/generate_capability_evidence.py"
)
assert SPEC and SPEC.loader
generator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(generator)


def test_committed_guidance_matches_artifacts_and_links_resolve() -> None:
    outputs = generator.render(ROOT)
    for path, content in outputs.items():
        assert path.read_text() == content, f"Regenerate {path.relative_to(ROOT)}"
        generated = content.split(generator.BEGIN)[1].split(generator.END)[0]
        for link in re.findall(r"\]\(([^)]+)\)", generated):
            assert (path.parent / link).is_file(), (path, link)


def test_ratios_use_raw_medians_and_follow_changed_workload_metadata() -> None:
    data = generator.read_artifacts(ROOT)
    row = generator.row_at(data["derivative_order"], "n", 8)
    repeats = data["derivative_order"]["config"]["repeats"]
    row["measurements"]["closed_form"] = {"median_ms": 2.0, "samples_ms": [2.0] * repeats}
    row["measurements"]["nested_autograd"] = {"median_ms": 7.0, "samples_ms": [7.0] * repeats}
    row["speedup_vs_closed_form"] = 9999  # The redundant rounded summary is not authoritative.
    data["derivative_order"]["config"]["n_points"] = 12345
    data["laplacian_scaling"]["config"]["H"] = 17
    text = generator.evidence(data, "../")
    assert "**3.5×**" in text and "9999" not in text
    assert "12,345" in text and "H=17" in text
    # A winning baseline is retained as a ratio below one, not hidden or inverted.
    row["measurements"]["nested_autograd"] = {"median_ms": 1.0, "samples_ms": [1.0] * repeats}
    assert "**0.5×**" in generator.evidence(data, "../")


@pytest.mark.parametrize("status", ["memory_budget", "timeout", "error"])
@pytest.mark.parametrize("method", ["omnibias", "folx_nested"])
def test_failed_selected_measurements_never_generate_speedups(status: str, method: str) -> None:
    data = generator.read_artifacts(ROOT)
    row = generator.row_at(data["polylaplacian_order"], "k", 4)
    row[method] = {"status": status, "median_ms": 0.000001}
    assert generator.comparison(row["omnibias"], row["folx_nested"], 9) == (
        f"unavailable ({status}; no speedup)"
    )
    assert f"unavailable ({status}; no speedup)" in generator.evidence(data, "")


@pytest.mark.parametrize("change", [
    {"median_ms": 0.0, "samples_ms": [0.0]},
    {"median_ms": float("nan"), "samples_ms": [float("nan")]},
    {"median_ms": 2.0, "samples_ms": [1.0]},
    {"median_ms": 1.0, "samples_ms": []},
    {"status": "unrecognized", "median_ms": 1.0, "samples_ms": [1.0]},
])
def test_invalid_successful_measurements_are_rejected(change: dict) -> None:
    with pytest.raises(ValueError):
        generator.median(change, 1)


def test_incompatible_protocol_and_ambiguous_workload_fail() -> None:
    original = generator.read_artifacts(ROOT)
    mutations = [
        ("laplacian_scaling", "dtype", "float32"),
        ("polylaplacian_order", "runtime_arguments", []),
        ("derivative_order", "execution", "GPU eager"),
    ]
    for artifact, key, value in mutations:
        changed = copy.deepcopy(original)
        changed[artifact]["config"][key] = value
        with pytest.raises(ValueError):
            generator.evidence(changed, "")
    original["laplacian_scaling"]["rows"].append(
        copy.deepcopy(generator.row_at(original["laplacian_scaling"], "D", 60))
    )
    with pytest.raises(ValueError, match="exactly one"):
        generator.evidence(original, "")


def test_write_check_and_marker_failure_preserve_authored_guidance(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(generator, "ROOT", tmp_path)
    data = generator.read_artifacts(ROOT)
    artifacts = tmp_path / "docs/benchmarks"
    artifacts.mkdir(parents=True)
    for name, payload in data.items():
        (artifacts / f"{name}.json").write_text(json.dumps(payload))
    for relative in generator.TARGETS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"authored start\n{generator.BEGIN}\nstale\n{generator.END}\nauthored end\n")
    assert generator.main(["--check"]) == 1
    assert generator.main([]) == 0
    assert generator.main(["--check"]) == 0
    for relative in generator.TARGETS:
        text = (tmp_path / relative).read_text()
        assert text.startswith("authored start\n") and text.endswith("\nauthored end\n")
    first, last = (tmp_path / generator.TARGETS[index] for index in (0, -1))
    first.write_text(first.read_text().replace("**220×**", "drift"))
    assert generator.main(["--check"]) == 1
    unchanged = first.read_text()
    last.write_text(generator.END + generator.BEGIN)
    with pytest.raises(SystemExit):
        generator.main([])
    assert first.read_text() == unchanged  # Validate every target before any write.
