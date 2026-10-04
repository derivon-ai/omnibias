# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Benchmark outputs follow the scratch-directory contract."""
from __future__ import annotations

import json

import _common


def test_benchmark_output_defaults_to_repository_artifacts(tmp_path, monkeypatch):
    monkeypatch.delenv("OMNIBIAS_SCRATCH", raising=False)
    monkeypatch.setattr(_common, "REPO_ROOT", tmp_path)
    payload = {"configuration": {"order": 4}, "max_error": 1e-12}
    output = _common.write_json("derivatives.json", payload)
    assert output == tmp_path / "artifacts" / "derivatives.json"
    assert json.loads(output.read_text()) == payload
    assert not (tmp_path / "docs").exists()


def test_benchmark_output_honors_scratch_override(tmp_path, monkeypatch):
    destination = tmp_path / "results" / "derivatives"
    monkeypatch.setenv("OMNIBIAS_SCRATCH", str(destination))
    output = _common.write_json("timing.json", {"elapsed_ms": 0.5})
    assert output == destination / "timing.json"
    assert json.loads(output.read_text()) == {"elapsed_ms": 0.5}
