# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Fields front end for theory 03-10 (requires omnibias-difference)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("omnibias.difference.singularity")


def test_fields_reexport_disclaimer() -> None:
    from omnibias.fields.singularity import DISCLAIMER, honesty_payload, track_singularity

    rows = ((1.0, 2.0, 4.0, 8.0),)
    track = track_singularity(rows, (0.0,), method="pade")
    assert track.disclaimer == DISCLAIMER
    assert honesty_payload()["blowup_proof"] is False


def test_terminology() -> None:
    path = Path(__file__).resolve().parents[1] / "src" / "omnibias" / "fields" / "singularity.py"
    text = path.read_text(encoding="utf-8")
    assert "founding bias collapse" in text
    assert "delta -> 0" in text
    assert "beta -> inf" in text
    assert "feasibility" in text
    assert "do not conflate" in text.lower()
