# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Theory-tree structural guards (theory 06-02 G3/G5)."""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
FRONTIER = REPO / "theory" / "07-frontier"
THEORY = REPO / "theory"
HONESTY_SRC = (
    REPO
    / "packages"
    / "omnibias-pinn"
    / "src"
    / "omnibias"
    / "pinn"
    / "solver"
    / "_core"
    / "honesty.py"
)
METHOD_LABELS = ("CLOSED_FORM", "AUTODIFF", "NUMERICAL", "SPECTRAL", "HIGH_ORDER")


def test_group_07_specs_have_section_13() -> None:
    files = sorted(p for p in FRONTIER.glob("*.md") if p.name != "README.md")
    assert files, "theory/07-frontier/ has no specs"
    for path in files:
        text = path.read_text(encoding="utf-8")
        assert re.search(r"^## 13\.", text, re.MULTILINE), f"{path.name} missing ## 13."
        assert re.search(r"(?i)\bparent\b", text), f"{path.name} does not name a parent"
        assert re.search(
            r"(?i)(stays an external|external obligation|does not claim|not claimed)",
            text,
        ), f"{path.name} does not state the non-claim"


def test_method_labels_exist_in_honesty_module() -> None:
    text = HONESTY_SRC.read_text(encoding="utf-8")
    missing = [label for label in METHOD_LABELS if label not in text]
    assert missing == [], f"honesty.py missing labels {missing}"


def test_claim_docs_name_the_five_method_labels() -> None:
    paths = (
        THEORY / "06-program" / "02-honesty-and-claim-boundaries.md",
        REPO / "docs" / "honesty.md",
    )
    for path in paths:
        text = path.read_text(encoding="utf-8")
        missing = [label for label in METHOD_LABELS if label not in text]
        assert missing == [], f"{path.name} missing method labels {missing}"
