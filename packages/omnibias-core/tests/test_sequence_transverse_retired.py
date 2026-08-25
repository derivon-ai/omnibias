# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""05-02 G5 / Wave-0 A5 shipping guard.

G5 is earned on the order-0 logistic tail at width = horizon. Stdlib +
pytest only (numpy-free core job).
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def test_torch_sequence_submodule_shipped() -> None:
    root = REPO / "packages" / "omnibias-torch" / "src" / "omnibias" / "torch"
    assert (root / "sequence.py").is_file()


def test_jax_sequence_submodule_shipped() -> None:
    root = REPO / "packages" / "omnibias-jax" / "src" / "omnibias" / "jax"
    assert (root / "sequence.py").is_file()


def test_smoke_artifact_records_g5_earned() -> None:
    import json

    path = REPO / "docs" / "benchmarks" / "sequence_transverse_smoke.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema"] == "sequence-transverse-v1"
    assert payload["g5_earned"] is True
    assert payload["sequence_submodule_shipped"] is True
    assert payload["gates"]["all_passed"] is True
    assert str(payload["baseline"].get("name") or "").startswith("S4D")
    assert payload["seeds"] == [0, 1, 2, 3, 4]
    assert len(payload["per_seed"]) == 5
    assert payload["honesty"]["kernel_order"] == 0
    assert payload["honesty"]["width_equals_horizon"] is True
    assert max(float(row["r2_gap"]) for row in payload["per_seed"]) <= 0.02
