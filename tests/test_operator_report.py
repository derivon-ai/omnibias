# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Evidence verification, complete-group gating and revision isolation."""

import hashlib
import importlib.util
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "benchmarks/operator_report.py"
SPEC = importlib.util.spec_from_file_location("operator_evidence_report", SOURCE)
assert SPEC is not None and SPEC.loader is not None
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def _payload(tmp_path, revision="one", replicates=3):
    source = tmp_path / f"{revision}.py"
    source.write_text(f"# {revision}\n")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    provenance = {"source_revision": revision, "source_worktree_dirty": True,
                  "source_sha256": {"packages/example.py": digest},
                  "source_snapshots": {"packages/example.py": str(source)}}
    rows = []
    for method, median in (("compiler", 1.), ("manual", 2.)):
        for replicate in range(replicates):
            rows.append({"family": "ridge", "order": 4, "method": method, "replicate": replicate,
                         "execution": "compiled", "activation_method": "recurrence",
                         "operands": {"fixed": [1., 2.]}, "backend_version": "test",
                         "status": "ok", "source_changed_during_cell": False,
                         "source_provenance": provenance,
                         "accuracy": {"points": [{"status": "ok", "relative_error": 1e-15}]},
                         "capture_ms": 5., "compilation_ms": 10.,
                         "evaluation": {"status": "ok", "median_ms": median, "samples_ms": [median]*20},
                         "training_step": {"status": "ok", "median_ms": median*3, "samples_ms": [median*3]*20}})
    return {"schema": "omnibias/operator-compiler/v1", "rows": rows,
            "config": {"repeats": 20}, "environment": {"test": True}}


@pytest.fixture
def licenses(tmp_path, monkeypatch):
    path = tmp_path / "LICENSE.baseline"
    path.write_text("Baseline license text\n")
    monkeypatch.setattr(report, "_distribution_licenses", lambda required: (
        [("folx", path)], [{"name": "folx", "version": "1"}]))
    return path


def _write(tmp_path, name, payload):
    path = tmp_path / name
    path.write_text(json.dumps(payload))
    return path


def test_packaged_sources_reports_licenses_and_manifest_hashes(tmp_path, licenses):
    payload = _payload(tmp_path)
    snapshot = tmp_path / "folx.py"
    snapshot.write_text("# baseline source\n")
    payload["rows"][0]["baseline_sources"] = {"/installed/folx/api.py": {
        "snapshot": str(snapshot), "sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest()}}
    for row in payload["rows"]:
        row["baseline_version"] = "1"
    raw = _write(tmp_path, "run.json", payload)
    accuracy = _write(tmp_path, "accuracy.json", {"rows": [], "source_provenance": payload["rows"][0]["source_provenance"]})
    output, archive = tmp_path / "summary.json", tmp_path / "evidence.tar.gz"
    summary = report.package([raw], output, archive, accuracy)
    assert summary["reports"][0]["comparisons"][0]["baseline_over_compiler"]["manual"] == 2
    stages = summary["reports"][0]["method_groups"][0]["median_of_process_stage_medians"]
    assert stages["training_step_median_ms"] == 3
    with tarfile.open(archive) as bundle:
        manifest = json.load(bundle.extractfile("manifest.json"))
        assert json.load(bundle.extractfile("reports/summary.json")) == json.loads(output.read_text())
        for name, metadata in manifest["members"].items():
            assert hashlib.sha256(bundle.extractfile(name).read()).hexdigest() == metadata["sha256"]
        assert "LICENSE" in bundle.getnames()
        assert "tools/operator_report.py" in bundle.getnames()
        assert any(name.startswith("licenses/folx/") for name in bundle.getnames())
        assert b"RESTORE THE SNAPSHOTS" in bundle.extractfile("README.md").read()
        assert manifest["sources"][0]["original"] == "packages/example.py"
        assert len(manifest["sources"][0]["reports"]) == 2


@pytest.mark.parametrize("kind", ["source", "baseline"])
def test_sha_mismatch_refuses_before_writing_outputs(tmp_path, kind):
    payload = _payload(tmp_path)
    if kind == "source":
        snapshot = Path(payload["rows"][0]["source_provenance"]["source_snapshots"]["packages/example.py"])
    else:
        snapshot = tmp_path / "baseline.py"
        snapshot.write_text("baseline\n")
        payload["rows"][0]["baseline_sources"] = {
            "/installed/baseline.py": {"snapshot": str(snapshot), "sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest()}}
    snapshot.write_text("changed after measurement\n")
    raw = _write(tmp_path, "bad.json", payload)
    output, archive = tmp_path / "summary.json", tmp_path / "evidence.tar.gz"
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        report.package([raw], output, archive)
    assert not output.exists() and not archive.exists()


def test_never_pool_partial_inputs_or_source_revisions(tmp_path, licenses):
    first = _write(tmp_path, "first.json", _payload(tmp_path, "first", 2))
    second = _write(tmp_path, "second.json", _payload(tmp_path, "second", 2))
    summary = report.package([first, second], tmp_path / "summary.json", tmp_path / "evidence.tar.gz")
    assert len(summary["reports"]) == 2
    assert all(item["comparisons"] == [] for item in summary["reports"])
    assert all(not group["eligible"] for item in summary["reports"] for group in item["method_groups"])
    # Even within a single input, a method whose replicates changed revision
    # cannot earn timing aggregates or ratios.
    payload = _payload(tmp_path, "third")
    payload["rows"][0]["source_provenance"] = json.loads(first.read_text())["rows"][0]["source_provenance"]
    result = report.summarize(payload)
    assert result["comparisons"] == []
    assert not result["method_groups"][0]["eligible"]


def test_failures_and_unverified_accuracy_never_get_stage_medians(tmp_path):
    payload = _payload(tmp_path)
    payload["rows"][0]["status"] = "timeout"
    payload["rows"][0]["reason"] = "resource budget"
    payload["rows"][0]["completed_stage"] = "forward_accuracy"
    result = report.summarize(payload)
    assert result["status_counts"]["timeout"] == 1
    assert result["failures"][0] == {"row_index": 0, "family": "ridge", "order": 4,
                                     "method": "compiler", "replicate": 0, "status": "timeout",
                                     "reason": "resource budget", "completed_stage": "forward_accuracy"}
    assert "median_of_process_stage_medians" not in result["method_groups"][0]
    assert result["comparisons"] == []
    payload = _payload(tmp_path)
    del payload["rows"][0]["accuracy"]
    assert report.summarize(payload)["comparisons"] == []


def test_v2_failure_stages_and_reference_totals_are_preserved(tmp_path):
    payload = _payload(tmp_path)
    payload.update(schema="omnibias/operator-compiler/v2", reference_generation_total_seconds=4.5,
                   wall_seconds_through_measurements=122.5)
    payload["rows"][0].update(status="timeout", completed_stage="capture", active_stage="compile_and_first_execution",
                               stage_events=[{"stage": "compile_and_first_execution", "state": "started"}])
    result = report.summarize(payload)
    assert result["reference_generation_total_seconds"] == 4.5
    assert result["wall_seconds_through_measurements"] == 122.5
    assert result["failures"][0]["active_stage"] == "compile_and_first_execution"
    assert result["failures"][0]["stage_events"] == payload["rows"][0]["stage_events"]
    # The synthetic v1 rows lack v2 oracle evidence. A v2 report must not admit
    # them merely because its row-level protocol tag is missing.
    assert not result["comparisons"]
    assert not any(group["eligible"] for group in result["method_groups"])


def test_missing_snapshot_metadata_is_not_replaced_with_current_checkout(tmp_path):
    payload = _payload(tmp_path)
    payload["rows"][0]["source_provenance"]["source_snapshots"] = {}
    raw = _write(tmp_path, "missing.json", payload)
    with pytest.raises(ValueError, match="exactly the same"):
        report.package([raw], tmp_path / "summary.json", tmp_path / "evidence.tar.gz")


def test_compiler_only_needs_no_optional_license_metadata(tmp_path, monkeypatch):
    payload = _payload(tmp_path)
    payload["rows"] = [row for row in payload["rows"] if row["method"] == "compiler"]

    def forbidden(*args, **kwargs):
        raise AssertionError("Optional distribution metadata must not be queried")

    monkeypatch.setattr(report, "_distribution_licenses", forbidden)
    raw = _write(tmp_path, "compiler.json", payload)
    archive = tmp_path / "evidence.tar.gz"
    report.package([raw], tmp_path / "summary.json", archive)
    with tarfile.open(archive) as bundle:
        assert json.load(bundle.extractfile("manifest.json"))["distributions"] == []
        assert not any(name.startswith("licenses/") for name in bundle.getnames())


def test_only_bundled_baselines_require_licenses_and_versions_are_checked(tmp_path, monkeypatch):
    values = {"rows": [
        {"method": "folx", "baseline_version": "1", "baseline_sources": {"/package/api.py": {}}},
        {"method": "jax-jet", "backend_version": "2", "baseline_sources": {"/package/jax/jet.py": {}}},
        {"method": "torch-jet", "baseline_version": "3"},  # no bundled code
    ]}
    assert report._baseline_requirements(values) == {"folx": {"1"}, "jax": {"2"}}
    license_file = tmp_path / "LICENSE"
    license_file.write_text("test license\n")
    installed = SimpleNamespace(metadata={"Name": "folx"}, version="1", files=[Path("LICENSE")],
                                locate_file=lambda path: tmp_path / path)
    monkeypatch.setattr(report.importlib.metadata, "distributions", lambda **kwargs: [installed])
    files, records = report._distribution_licenses({"folx": {"1"}})
    assert files == [("folx", license_file)]
    assert records[0]["version"] == "1"
    with pytest.raises(ValueError, match="License version mismatch"):
        report._distribution_licenses({"folx": {"2"}})


def test_attachments_roundtrip_hashes_and_overwrite_protection(tmp_path):
    raw = _write(tmp_path, "run.json", _payload(tmp_path))
    diagnostic = tmp_path / "diagnostic.bin"
    content = b"Diagnostic bytes are not timing input\x00\xff\n"
    diagnostic.write_bytes(content)
    output, archive = tmp_path / "summary.json", tmp_path / "evidence.tar.gz"
    summary = report.package([raw], output, archive, attachments=[diagnostic])
    with tarfile.open(archive) as bundle:
        manifest = json.load(bundle.extractfile("manifest.json"))
        attachment = manifest["attachments"][0]
        assert attachment == summary["attachments"][0]
        assert attachment["original"] == str(diagnostic)
        assert attachment["archive_path"] == "attachments/000-diagnostic.bin"
        assert attachment["sha256"] == hashlib.sha256(content).hexdigest()
        assert bundle.extractfile(attachment["archive_path"]).read() == content
    for output, archive in ((diagnostic, tmp_path / "other.tar.gz"), (tmp_path / "other.json", diagnostic)):
        with pytest.raises(ValueError, match="attachments"):
            report.package([raw], output, archive, attachments=[diagnostic])
    assert diagnostic.read_bytes() == content


def test_public_location_redaction_keeps_exact_raw_archive(tmp_path):
    payload = _payload(tmp_path)
    other_home = str(Path("/") / "home" / "synthetic-user")
    payload["environment"].update(checkout=str(report.ROOT), user_home=str(Path.home()),
                                   old_host=other_home + "/project/run.py")
    raw = _write(tmp_path, "private-paths.json", payload)
    original_bytes = raw.read_bytes()
    output, archive = tmp_path / "summary.json", tmp_path / "evidence.tar.gz"
    summary = report.package([raw], output, archive)
    published = output.read_text()
    assert str(report.ROOT) not in published and str(Path.home()) not in published
    assert other_home not in published
    assert summary["reports"][0]["environment"]["checkout"] == "<workspace>"
    assert summary["reports"][0]["environment"]["old_host"] == "<home>/project/run.py"
    assert summary["reports"][0]["comparisons"][0]["baseline_over_compiler"]["manual"] == 2
    assert raw.read_bytes() == original_bytes
    with tarfile.open(archive) as bundle:
        manifest = json.load(bundle.extractfile("manifest.json"))
        record = manifest["reports"][0]
        archived = bundle.extractfile(record["archive_path"]).read()
        assert archived == original_bytes
        assert record["sha256"] == hashlib.sha256(original_bytes).hexdigest()
        assert json.load(bundle.extractfile("reports/summary.json")) == summary


def test_public_summary_is_detached_and_preserves_numeric_hash_fields():
    digest = "a0" * 32
    evidence = {"path": str(report.ROOT / "artifacts"), "sha256": digest,
                "nested": [{"median_ms": 0.003123456789, "count": 20, "passed": True, "absent": None}]}
    original = json.loads(json.dumps(evidence))
    public = report.public_summary(evidence)
    assert evidence == original
    assert public["path"] == "<workspace>/artifacts"
    assert public["sha256"] == digest and public["nested"] == evidence["nested"]
