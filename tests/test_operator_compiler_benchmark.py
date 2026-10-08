# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Benchmark protocol tests: accuracy, budgets, fixed operands and provenance."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "benchmarks/operator_compiler.py"
SPEC = importlib.util.spec_from_file_location("operator_compiler_benchmark", SOURCE)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


def test_protocol_defaults_and_determinism():
    args = benchmark.parser().parse_args([])
    benchmark.validate(args)
    assert (args.repeats, args.replicates, args.batch) == (20, 3, 32)
    assert args.activation_method == "recurrence"
    assert args.kernel_lowering == "auto" and args.precision == "strict"
    data = benchmark.model_data("separable", 24, 2, 16, 16, 2, 7)
    assert data["dimension"] == 24
    assert data == benchmark.model_data("separable", 24, 2, 16, 16, 2, 7)
    assert data != benchmark.model_data("separable", 24, 2, 16, 16, 2, 8)


def test_independent_reference_and_small_derivative_accuracy_gate():
    data = {"family": "coupled", "points": [[.25]], "layers": [([[3.]], [2.], False)]}
    assert float(benchmark.high_precision(data, 1, [0])[0]) == 3
    assert benchmark.check_values([1e-30], ["1e-30"])["points"][0]["absolute_error"] < 1e-45
    with pytest.raises(ArithmeticError, match="reference failed"):
        benchmark.check_values([0.], ["1e-30"])
    with pytest.raises(ArithmeticError):
        benchmark.check_values([float("inf")], ["1"])


def test_baseline_applicability_and_order_budgets():
    assert benchmark.admissible("ridge", 8, "folx", 8) is None
    assert benchmark.admissible("separable", 24, "compiler", 8) is None
    assert "budget" in benchmark.admissible("separable", 24, "nested", 8)
    assert "directional" in benchmark.admissible("ridge", 4, "jax-jet", 8)


def test_memory_summary_accounts_for_preexisting_and_version_events():
    events = [[-1, 1, 80, 2], [0, 2, 16, 2], [1, 3, -80, 2],
              [1, 3, 80, 3], [2, 4, -16, 2], [3, 2, 32, 2]]
    assert benchmark.memory_summary(events) == {
        "preexisting_bytes": 80, "peak_live_bytes": 112, "peak_above_preexisting_bytes": 32}


def test_ratios_require_all_accuracy_gated_fresh_processes_and_samples():
    rows = []
    for method, time in (("compiler", .024), ("manual", .02), ("nested", 2.)):
        for replicate in range(3):
            rows.append({"family": "ridge", "order": 4, "method": method, "replicate": replicate,
                         "status": "ok", "source_changed_during_cell": False,
                         "source_provenance": {"source_sha256": {"a.py": "hash"}},
                         "evaluation": {"median_ms": time, "samples_ms": [time]*20}})
    result = benchmark.comparisons(rows)[0]
    assert result["ridge_release_gate_pass"]
    assert result["fastest_measured_baseline"] == "manual"
    rows[0]["status"] = "timeout"
    assert benchmark.comparisons(rows) == []


def test_timeout_retains_completed_stage_without_success(monkeypatch):
    args = benchmark.parser().parse_args(["--timeout", "0.2"])
    row = {"family": "ridge", "order": 2, "method": "compiler", "completed_stage": "forward_accuracy"}
    script = "import time; print(" + repr(benchmark.PROGRESS+json.dumps(row)) + ", flush=True); time.sleep(5)"
    monkeypatch.setattr(benchmark, "worker_command", lambda *args: [sys.executable, "-c", script])
    result = benchmark.run_case(args, "ridge", 2, "compiler")
    assert result["status"] == "timeout"
    assert result["completed_stage"] == "forward_accuracy"
    assert "speedup" not in result


def test_exact_source_snapshots(tmp_path, monkeypatch):
    monkeypatch.setenv("OMNIBIAS_SCRATCH", str(tmp_path))
    paths = ["benchmarks/operator_compiler.py", "benchmarks/_operator_models.py"]
    provenance = benchmark.snapshots(paths)
    for path, snapshot in provenance["source_snapshots"].items():
        assert Path(snapshot).read_bytes() == (SOURCE.parents[1] / path).read_bytes()


def test_reference_identity_and_integrity(monkeypatch):
    data = benchmark.model_data("separable", 2, 2, 16, 16, 2, 7)
    reference = benchmark.prepare_reference(data, 2)
    benchmark.validate_reference(json.loads(json.dumps(reference)), data, 2)
    assert reference["generation_seconds"] >= 0
    altered = json.loads(json.dumps(data))
    altered["points"][0][0] += 1e-16
    with pytest.raises(ValueError, match="exact operands"):
        benchmark.validate_reference(reference, altered, 2)
    with pytest.raises(ValueError, match="exact operands"):
        benchmark.validate_reference(reference, data, 3)
    damaged = reference | {"expected": ["0", "0"]}
    with pytest.raises(ValueError, match="checksum"):
        benchmark.validate_reference(damaged, data, 2)
    monkeypatch.setattr(benchmark.importlib.metadata, "version", lambda _: "changed")
    with pytest.raises(ValueError, match="oracle revision"):
        benchmark.validate_reference(reference, data, 2)


def test_engine_consumes_reference_without_running_oracle(tmp_path, monkeypatch, capsys):
    args = benchmark.parser().parse_args(["--worker", "compiler", "--family", "separable", "--order", "2", "--batch", "2"])
    data = benchmark.model_data("separable", 2, 2, 16, 16, 2, 7)
    reference = benchmark.prepare_reference(data, 2)
    args.reference_file = tmp_path / "reference.json"
    args.reference_file.write_text(json.dumps(reference))
    monkeypatch.setenv("OMNIBIAS_SCRATCH", str(tmp_path))
    monkeypatch.setattr(benchmark, "_configure_process", lambda _: None)
    monkeypatch.setattr(benchmark, "high_precision", lambda *args: pytest.fail("oracle ran inside engine"))
    monkeypatch.setattr(benchmark, "_torch", lambda args, data, row: row | {"status": "ok"})
    result = benchmark.worker(args)
    assert result["status"] == "ok" and result["reference"] == reference
    assert result["protocol"] == benchmark.SCHEMA
    assert "excluding separately generated oracle" in result["rss_scope"]
    events = [json.loads(line[len(benchmark.PROGRESS):]) for line in capsys.readouterr().out.splitlines()]
    assert events[0]["active_stage"] == "preparation"
    assert events[-1]["completed_stage"] == "preparation"


def test_stage_progress_preserves_active_and_completed(capsys):
    row = {}
    benchmark.progress(row, "capture")
    benchmark.progress(row, "compile_and_first_execution", "started")
    latest = json.loads(capsys.readouterr().out.splitlines()[-1][len(benchmark.PROGRESS):])
    assert latest["completed_stage"] == "capture"
    assert latest["active_stage"] == "compile_and_first_execution"
    assert [event["state"] for event in latest["stage_events"]] == ["completed", "started"]


def test_runtime_scopes_have_distinct_comparison_identities():
    row = {"family": "ridge", "order": 2, "execution": "compiled"}
    base = benchmark.query_signature(row)
    for field, value in (("protocol", benchmark.SCHEMA), ("cache_policy", "cold"),
                         ("kernel_lowering", "regional"), ("precision", "strict")):
        assert benchmark.query_signature(row | {field: value}) != base
    old_cold = row | {"cache_policy": "cold"}
    new_cold = old_cold | {"cache_protocol": benchmark.cache_protocol("cold")}
    assert benchmark.query_signature(old_cold) != benchmark.query_signature(new_cold)


def test_cold_cache_preserves_within_worker_reuse_and_thread_provenance(tmp_path, monkeypatch):
    monkeypatch.setenv("TORCHINDUCTOR_FORCE_DISABLE_CACHES", "1")
    monkeypatch.setenv("TORCHINDUCTOR_COMPILE_THREADS", "2")
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", "/unused-parent-cache")
    monkeypatch.setenv("JAX_ENABLE_COMPILATION_CACHE", "false")
    cold = benchmark.cache_environment("cold", tmp_path)
    assert cold["TORCHINDUCTOR_CACHE_DIR"] == str(tmp_path / "inductor")
    assert cold["JAX_COMPILATION_CACHE_DIR"] == str(tmp_path / "jax")
    assert "TORCHINDUCTOR_FORCE_DISABLE_CACHES" not in cold
    assert cold["JAX_ENABLE_COMPILATION_CACHE"] == "true"
    assert cold["TORCHINDUCTOR_COMPILE_THREADS"] == "2"
    assert not (tmp_path / "inductor").exists()  # Fresh parent-owned directory.
    disabled = benchmark.cache_environment("disabled", tmp_path)
    assert disabled["TORCHINDUCTOR_FORCE_DISABLE_CACHES"] == "1"
    assert disabled["JAX_ENABLE_COMPILATION_CACHE"] == "false"
    assert benchmark.cache_protocol("cold") != benchmark.cache_protocol("disabled")
    assert benchmark.parser().parse_args(["--cache", "disabled"]).cache == "disabled"


def test_persistent_cache_uses_existing_paths_with_normal_reuse(tmp_path, monkeypatch):
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path / "shared"))
    monkeypatch.setenv("TORCHINDUCTOR_FORCE_DISABLE_CACHES", "1")
    environment = benchmark.cache_environment("persistent", tmp_path / "isolated")
    assert environment["TORCHINDUCTOR_CACHE_DIR"] == str(tmp_path / "shared")
    assert "TORCHINDUCTOR_FORCE_DISABLE_CACHES" not in environment
    assert benchmark.cache_protocol("persistent")["initial_state"] == "shared persistent cache"


def test_v2_reference_evidence_remains_tied_to_measured_snapshot():
    data = benchmark.model_data("separable", 2, 2, 16, 16, 2, 7)
    reference = benchmark.prepare_reference(data, 2)
    row = {"protocol": benchmark.SCHEMA, "family": "separable", "order": 2,
           "operands": data, "reference": reference,
           "source_provenance": {"source_sha256": {"benchmarks/_operator_models.py": reference["request"]["oracle_source_sha256"]}},
           "accuracy": {"points": [{"status": "ok"}]*2}}
    assert benchmark.reference_evidence_valid(row)
    assert not benchmark.reference_evidence_valid(row | {"order": 3})
    assert not benchmark.reference_evidence_valid(row | {"reference": reference | {"expected": ["0", "0"]}})
    assert not benchmark.reference_evidence_valid(row | {"source_provenance": {"source_sha256": {}}})


def test_all_backend_sources_are_in_snapshot_inventory():
    sources = benchmark.source_paths()
    assert "benchmarks/operator_report.py" in sources
    for package, backend in (("omnibias-torch", "torch"), ("omnibias-keras", "keras")):
        root = SOURCE.parents[1] / "packages" / package / "src/omnibias" / backend
        expected = {str(path.relative_to(SOURCE.parents[1])) for path in root.rglob("*.py")}
        assert expected.issubset(sources)


@pytest.mark.parametrize("flags", [["--repeats", "0"], ["--orders", "33"], ["--output", "../bad.json"]])
def test_invalid_configuration(flags):
    with pytest.raises(ValueError):
        benchmark.validate(benchmark.parser().parse_args(flags))
