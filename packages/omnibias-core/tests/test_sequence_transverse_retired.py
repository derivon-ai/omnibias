# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""05-02 G5 / Wave-0 A5 retirement guard.

The causal transverse filter lost to a named S4D baseline. The sequence
submodule must stay unshipped. Stdlib + pytest only (numpy-free core job).
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def test_no_torch_sequence_submodule() -> None:
    root = REPO / "packages" / "omnibias-torch" / "src" / "omnibias" / "torch"
    assert not (root / "sequence.py").exists()
    assert not (root / "sequence").is_dir()


def test_no_jax_sequence_submodule() -> None:
    root = REPO / "packages" / "omnibias-jax" / "src" / "omnibias" / "jax"
    assert not (root / "sequence.py").exists()
    assert not (root / "sequence").is_dir()


def test_smoke_artifact_records_g5_unearned() -> None:
    import json

    path = REPO / "docs" / "benchmarks" / "sequence_transverse_smoke.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema"] == "sequence-transverse-v1"
    assert payload["g5_earned"] is False
    assert payload["sequence_submodule_shipped"] is False
    assert payload["gates"]["all_passed"] is False
    assert str(payload["baseline"].get("name") or "").startswith("S4D")
    assert payload["seeds"] == [0, 1, 2, 3, 4]
    assert len(payload["per_seed"]) == 5
