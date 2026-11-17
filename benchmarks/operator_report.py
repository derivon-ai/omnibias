# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Verify and package operator benchmark evidence without pooling revisions.

Only the standard library and the sibling benchmark's comparison gate are used.
No numerical backend is imported and no measurements are executed by this tool.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import math
import os
import re
import statistics
import sys
import tarfile
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from operator_compiler import comparisons, query_signature  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "omnibias/operator-evidence/v1"


def _digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def public_summary(value: Any) -> Any:
    """Redact location metadata in a detached public view of exact evidence.

    Raw reports, attachments, source snapshots and manifest provenance remain
    byte-for-byte intact in the evidence archive. This view changes only text;
    timings, hashes, comparison keys and numerical evidence are unchanged.
    """
    if isinstance(value, dict):
        return {key: public_summary(child) for key, child in value.items()}
    if isinstance(value, list):
        return [public_summary(child) for child in value]
    if isinstance(value, str):
        value = value.replace(str(ROOT), "<workspace>")
        value = value.replace(str(Path.home()), "<home>")
        # Reports may originate on a different host from the packager.
        value = re.sub(r"/(?:home|Users|u)/[A-Za-z0-9._-]+", "<home>", value)
        return re.sub(r"[A-Za-z]:\\Users\\[^\\\s\"']+", "<home>", value)
    return value


def _signature(row: dict[str, Any]) -> str:
    return query_signature(row)


def _eligible(rows: list[dict[str, Any]]) -> bool:
    """Use the benchmark's gate for individual method groups, including baselines.

    Renaming this isolated group does not create or publish a ratio; it lets
    the authoritative gate check all methods, even groups lacking a compiler.
    """
    if any(not row.get("accuracy", {}).get("points") or
           any(point.get("status") != "ok" for point in row["accuracy"]["points"])
           for row in rows):
        return False
    return bool(comparisons([dict(row, method="compiler") for row in rows]))


def summarize(payload: dict[str, Any]) -> dict[str, Any]:
    """Keep failed cells visible; aggregate only complete homogeneous groups."""
    rows = payload.get("rows", [])
    if payload.get("schema") == "omnibias/operator-compiler/v2":
        # A missing row tag must not bypass the v2 reference-integrity gate.
        rows = [dict(row, protocol=payload["schema"]) for row in rows]
    groups: dict[tuple[str, int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["family"], row["order"], row["method"], _signature(row)].append(row)
    summaries = []
    for (family, order, method, query), cells in sorted(groups.items()):
        eligible = _eligible(cells)
        entry: dict[str, Any] = {
            "family": family, "order": order, "method": method,
            "query_configuration_sha256": query, "cells": len(cells),
            "status_counts": dict(Counter(cell["status"] for cell in cells)),
            "eligible": eligible,
        }
        if eligible:
            stages: dict[str, float] = {}
            for stage in ("capture_ms", "frontend_capture_ms", "compilation_ms",
                          "compile_setup_ms", "first_execution_ms", "compile_and_first_execution_ms",
                          "gradient_validation_first_call_ms", "engine_wall_seconds",
                          "reference_generation_seconds", "end_to_end_seconds"):
                values = [cell.get(stage) for cell in cells]
                if all(isinstance(value, (int, float)) and math.isfinite(value) for value in values):
                    stages[stage] = statistics.median(values)
            for stage in ("evaluation", "backward", "training_step"):
                values = [cell.get(stage, {}) for cell in cells]
                if all(value.get("status") == "ok" and len(value.get("samples_ms", [])) >= 20
                       and math.isfinite(value.get("median_ms", math.nan)) for value in values):
                    stages[stage + "_median_ms"] = statistics.median(value["median_ms"] for value in values)
            entry["median_of_process_stage_medians"] = stages
        summaries.append(entry)
    # Missing/malformed accuracy evidence cannot earn a ratio even if a hand-
    # edited row retains status=ok. Preserve it in the raw input regardless.
    gated_rows = [dict(row, status="unverified") if not row.get("accuracy", {}).get("points") or
                  any(point.get("status") != "ok" for point in row["accuracy"]["points"])
                  else row for row in rows]
    return {
        "schema": payload.get("schema"), "config": payload.get("config"),
        "environment": payload.get("environment"), "scope": payload.get("scope"),
        "cache_policy": payload.get("cache_policy"),
        "cache_protocol": payload.get("cache_protocol"),
        "resource_budget_scope": payload.get("resource_budget_scope"),
        "reference_generation_total_seconds": payload.get("reference_generation_total_seconds"),
        "wall_seconds_through_measurements": payload.get("wall_seconds_through_measurements"),
        "status_counts": dict(Counter(row["status"] for row in rows)),
        "method_groups": summaries, "comparisons": comparisons(gated_rows),
        "failures": [{"row_index": i, **{key: row[key] for key in (
            "family", "order", "method", "replicate", "status", "reason", "error", "completed_stage", "active_stage", "stage_events")
            if key in row}} for i, row in enumerate(rows) if row.get("status") != "ok"],
    }


def _snapshot_path(value: str, report: Path) -> Path:
    path = Path(value)
    candidates = [path] if path.is_absolute() else [path, ROOT / path, report.parent / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Missing source snapshot {value!r}, referenced by {report}")


def _snapshot_references(value: Any):
    """Walk report/row metadata, including optional standalone accuracy reports."""
    if isinstance(value, dict):
        provenance = value.get("source_provenance")
        if provenance is not None:
            hashes, snapshots = provenance.get("source_sha256", {}), provenance.get("source_snapshots", {})
            if set(hashes) != set(snapshots):
                raise ValueError("Source hashes and source snapshots must identify exactly the same files")
            for original, digest in hashes.items():
                yield "source", original, snapshots[original], digest
        for original, item in value.get("baseline_sources", {}).items():
            yield "baseline_source", original, item["snapshot"], item["sha256"]
        for key, child in value.items():
            if key not in {"source_provenance", "baseline_sources"}:
                yield from _snapshot_references(child)
    elif isinstance(value, list):
        for child in value:
            yield from _snapshot_references(child)


def _baseline_requirements(value: Any) -> dict[str, set[str | None]]:
    """Infer license requirements only from baseline source files being bundled."""
    required: dict[str, set[str | None]] = defaultdict(set)
    if isinstance(value, dict):
        for original in value.get("baseline_sources", {}):
            parts = Path(original.replace("\\", "/")).parts
            name = next((distribution for component, distribution in (
                ("folx", "folx"), ("jax", "jax"), ("jet", "jet-for-pytorch")) if component in parts), None)
            if name is None:
                name = {"folx": "folx", "torch-jet": "jet-for-pytorch", "jax-jet": "jax"}.get(value.get("method"))
            if name is None:
                raise ValueError(f"Cannot infer installed distribution for baseline snapshot {original!r}")
            version = value.get("backend_version" if name == "jax" else "baseline_version")
            required[name].add(str(version) if version is not None else None)
        for key, child in value.items():
            if key not in {"source_provenance", "baseline_sources"}:
                for name, versions in _baseline_requirements(child).items():
                    required[name].update(versions)
    elif isinstance(value, list):
        for child in value:
            for name, versions in _baseline_requirements(child).items():
                required[name].update(versions)
    return dict(required)


def _distribution_licenses(required: dict[str, set[str | None]]) -> tuple[list[tuple[str, Path]], list[dict[str, Any]]]:
    """Find installed license metadata without importing optional backends."""
    search = [str(path) for path in ROOT.glob(".venv*/lib/python*/site-packages")]
    search.append("/tmp/omnibias-jet-comparison")
    if not required:
        return [], []
    discovered: dict[str, dict[str, Any]] = defaultdict(dict)
    distributions = list(importlib.metadata.distributions())
    distributions.extend(importlib.metadata.distributions(path=search))
    for distribution in distributions:
        name = distribution.metadata.get("Name", "").lower().replace("_", "-")
        discovered[name].setdefault(distribution.version, distribution)
    files, records = [], []
    for name, versions in sorted(required.items()):
        if name not in discovered:
            raise FileNotFoundError(f"Installed distribution metadata needed to bundle {name} licenses was not found")
        selected = {version for version in versions if version is not None}
        if not selected:
            selected.add(next(iter(discovered[name])))
        for version in sorted(selected):
            if version not in discovered[name]:
                raise ValueError(f"License version mismatch for {name}: recorded {version}, installed {sorted(discovered[name])}")
            distribution = discovered[name][version]
            found = []
            for relative in distribution.files or []:
                if relative.name.upper().startswith(("LICENSE", "LICENCE", "COPYING", "AUTHORS", "NOTICE")):
                    path = Path(distribution.locate_file(relative)).resolve()
                    if path.is_file():
                        found.append(path)
                        files.append((name, path))
            if not found:
                raise FileNotFoundError(f"No LICENSE/AUTHORS/NOTICE files found in installed {name} metadata")
            records.append({"name": name, "version": distribution.version,
                            "recorded_versions": sorted(version for version in versions if version is not None),
                            "license_files": [str(path) for path in found]})
    return files, records


README = """# Operator compiler evidence

This archive contains the original JSON reports (including every raw timing
sample and failure), a regenerated summary, exact content-addressed source
snapshots, and license notices. It contains no compiler caches.
Optional attachments are copied verbatim for diagnostics, hardware information
or verification logs. They are not interpreted as measurements or timing evidence.
Optional dependency licenses are included only when their source snapshots are
bundled, and recorded dependency versions must match installed license metadata.

The summary treats each input report independently. A timing aggregate requires
three successful, unchanged-source, accuracy-gated processes with at least twenty
samples, matching query configuration and backend versions. Failed, refused or
timed-out cells receive no ratios; their completed stages remain in raw reports.
Numbers are medians of per-process medians. Compilation/first-call and memory
fields retain the original report scopes; RSS is not a tensor allocation peak.

Protocol v1 includes reference generation inside each worker's resource budget.
Protocol v2 precomputes the independent mp80 references outside engine workers,
keyed by exact runtime operands, derivative request, oracle source and version.
Each v2 row embeds the verified values and key. Engine wall/RSS excludes reference
generation; end_to_end_seconds adds one reference generation to that worker.
This per-cell accounting is not additive across rows: a report reuses references.
reference_generation_total_seconds records actual shared oracle work, while
wall_seconds_through_measurements records actual elapsed report execution.
Stage events identify both the last completed and the active interrupted phase.
Protocol, cache policy, precision and lowering variants never pool for ratios.
Cache protocol version 2 distinguishes cold compilation (a fresh process and
empty isolated cache directories, with normal reuse during that process) from
the explicit disabled diagnostic. Older reports labelled cold also forced Torch
cache disabling; they retain their original label and environment as historical
evidence and cannot pool with the corrected cache protocol.
The public summary replaces developer checkout/home locations with placeholders.
Raw report, attachment, source and manifest contents retain their exact provenance
inside this archive; public redaction never changes numerical results or hashes.

To reproduce a report, first inspect its config, environment and source_revision.
Check out that base repository revision and RESTORE THE SNAPSHOTS: the measured
working tree was dirty, so checkout alone does not reproduce the implementation.
manifest.json maps each original path and revision to sources/<sha256>. Select
the source hashes from the particular report/cell, copy repository files to
those relative paths, and verify SHA-256. Baseline paths are installed-package
paths; restore them only into a suitable isolated environment. Do not blindly
write absolute paths from metadata. Different reports may use different revisions.

Install matching recorded Python/framework/optional-baseline versions, preserve
the CPU/thread/cache policy, and rerun benchmarks/operator_compiler.py with that
report's config. Failed resource-limited cells are bounded experiments, not proof
that the baseline can never run. Accuracy probes are not exhaustive error bounds.

To rebuild a package from available original reports/snapshots:
python benchmarks/operator_report.py --inputs REPORT.json [REPORT2.json ...] \\
  --accuracy ACCURACY.json --output summary.json --archive evidence.tar.gz
The optional accuracy argument may be omitted. The packager verifies source
content before writing either output. It never merges measurements across inputs.
Use --attachments FILE [FILE ...] to include additional files without interpreting
their contents. Every attachment's original path and SHA-256 are in the manifest.
"""


def package(inputs: list[Path], output: Path, archive: Path, accuracy: Path | None = None,
            attachments: list[Path] | None = None) -> dict[str, Any]:
    """Verify every referenced source, then atomically write summary and archive."""
    if not inputs:
        raise ValueError("At least one input report is required")
    inputs = [path.resolve() for path in inputs]
    attachments = [path.resolve() for path in attachments or []]
    output, archive = output.resolve(), archive.resolve()
    if output == archive or output in inputs or archive in inputs or output in attachments or archive in attachments or accuracy and (
        output == accuracy.resolve() or archive == accuracy.resolve()
    ):
        raise ValueError("Output paths must be distinct from each other, input reports and attachments")
    members: dict[str, bytes] = {}
    mappings: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    input_records = []
    summaries = []
    required_licenses: dict[str, set[str | None]] = defaultdict(set)

    def add_report(path: Path, member: str, *, is_accuracy: bool = False) -> None:
        content = path.read_bytes()
        payload = json.loads(content)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected a JSON object in {path}")
        for kind, original, snapshot, digest in _snapshot_references(payload):
            located = _snapshot_path(snapshot, path)
            source = located.read_bytes()
            if _digest(source) != digest:
                raise ValueError(f"SHA-256 mismatch for {original!r}: {located}")
            if located in {output, archive}:
                raise ValueError("Outputs may not overwrite a referenced source snapshot")
            target = "sources/" + digest
            members[target] = source
            key = kind, original, snapshot, digest
            record = mappings.setdefault(key, {"kind": kind, "original": original, "snapshot": snapshot,
                                               "sha256": digest, "archive_path": target, "reports": []})
            if member not in record["reports"]:
                record["reports"].append(member)
        for distribution, versions in _baseline_requirements(payload).items():
            required_licenses[distribution].update(versions)
        members[member] = content
        input_records.append({"original": str(path), "archive_path": member, "sha256": _digest(content),
                              "kind": "accuracy_report" if is_accuracy else "benchmark_report"})
        if not is_accuracy:
            summaries.append({"input": str(path), "archive_path": member, **summarize(payload)})

    for index, path in enumerate(inputs):
        add_report(path, f"reports/{index:03d}-{path.name}")
    if accuracy is not None:
        add_report(accuracy.resolve(), "reports/accuracy.json", is_accuracy=True)
    assets = []
    attachment_records = []
    for index, path in enumerate(attachments):
        content = path.read_bytes()
        target = f"attachments/{index:03d}-{path.name}"
        members[target] = content
        attachment_records.append({"original": str(path), "archive_path": target,
                                   "sha256": _digest(content), "size_bytes": len(content)})
    license_files, distributions = _distribution_licenses(dict(required_licenses)) if required_licenses else ([], [])
    for kind, path in [("repository_license", ROOT / "LICENSE"), ("packager", Path(__file__).resolve())]:
        content = path.read_bytes()
        target = "LICENSE" if kind == "repository_license" else "tools/operator_report.py"
        members[target] = content
        assets.append({"kind": kind, "original": str(path), "archive_path": target, "sha256": _digest(content)})
    for distribution, path in license_files:
        content = path.read_bytes()
        target = f"licenses/{distribution}/{_digest(content)}-{path.name}"
        members[target] = content
        assets.append({"kind": "distribution_license", "distribution": distribution,
                       "original": str(path), "archive_path": target, "sha256": _digest(content)})
    summary = public_summary({"schema": SCHEMA, "aggregation_scope": "Each input is independent; revisions are never pooled.",
               "reports": summaries, "accuracy_report": str(accuracy.resolve()) if accuracy else None,
               "attachments": attachment_records})
    summary_bytes = _json(summary)
    members["reports/summary.json"] = summary_bytes
    members["README.md"] = README.encode()
    manifest = {"schema": SCHEMA, "reports": input_records, "sources": list(mappings.values()),
                "assets": assets, "distributions": distributions, "attachments": attachment_records,
                "members": {name: {"sha256": _digest(content), "size_bytes": len(content)}
                            for name, content in sorted(members.items())}}
    members["manifest.json"] = _json(manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary_archive = temporary_summary = None
    try:
        with tempfile.NamedTemporaryFile(dir=archive.parent, delete=False) as stream:
            temporary_archive = Path(stream.name)
        with tarfile.open(temporary_archive, "w:gz") as bundle:
            for name, content in sorted(members.items()):
                info = tarfile.TarInfo(name)
                info.size, info.mode, info.mtime = len(content), 0o644, 0
                bundle.addfile(info, io.BytesIO(content))
        with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as stream:
            temporary_summary = Path(stream.name)
            stream.write(summary_bytes)
        os.replace(temporary_archive, archive)
        os.replace(temporary_summary, output)
    finally:
        for path in (temporary_archive, temporary_summary):
            if path is not None:
                path.unlink(missing_ok=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--accuracy", type=Path)
    parser.add_argument("--attachments", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    summary = package(args.inputs, args.output, args.archive, args.accuracy, args.attachments)
    print(f"Packaged {len(summary['reports'])} reports: {args.output} and {args.archive}")


if __name__ == "__main__":
    main()
