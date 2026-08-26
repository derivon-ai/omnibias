# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Training-idea ledger guards (theory 08-01 G1–G5).

The ledger is a document. These tests read markdown and shipped symbol
names; they do not train a network or mint a package.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
TRAINING = REPO / "theory" / "08-training"
LEDGER = TRAINING / "01-training-idea-ledger.md"
THEORY_README = REPO / "theory" / "README.md"
PACKAGES = REPO / "packages"

LEDGER_TABLE_HEADER = "| Spec | Taxonomy | Home | Floor |"
LEDGER_OPTIMIZERS = ("03-12", "08-04", "08-06", "08-07", "08-08", "08-10", "08-11", "08-12")
LEDGER_LEARNING_RULES = ("08-02", "08-03", "08-05")
LEDGER_FILTERS = ("08-09",)
REJECTED = (
    "skip_chain_rule_global_min",
    "full_parameter_jacobian_flow",
)
STACK_SYMBOLS = (
    (
        "GaussNewton",
        "packages/omnibias-torch/src/omnibias/torch/optim.py",
        r"^class GaussNewton\b",
    ),
    (
        "CubicNewton",
        "packages/omnibias-torch/src/omnibias/torch/optim.py",
        r"^class CubicNewton\b",
    ),
    (
        "gauss_newton_minimize",
        "packages/omnibias-jax/src/omnibias/jax/optim.py",
        r"^def gauss_newton_minimize\b",
    ),
    (
        "cubic_regularized_gauss_newton_minimize",
        "packages/omnibias-jax/src/omnibias/jax/optim.py",
        r"^def cubic_regularized_gauss_newton_minimize\b",
    ),
    (
        "recommended_stack_step",
        "packages/omnibias-core/src/omnibias/core/train_stack.py",
        r"^def recommended_stack_step\b",
    ),
)
_SECTION = re.compile(
    r"^## (?P<n>\d+)\. [^\n]+\n(?P<body>.*?)(?=^## \d|\Z)",
    re.MULTILINE | re.DOTALL,
)


def _section(text: str, number: int) -> str:
    for match in _SECTION.finditer(text):
        if int(match.group("n")) == number:
            return match.group("body")
    raise AssertionError(f"missing ## {number}.")


def _entry_specs() -> list[Path]:
    return sorted(
        path
        for path in TRAINING.glob("*.md")
        if path.name != "README.md" and path != LEDGER
    )


def _ledger_rows(text: str) -> list[list[str]]:
    rows: list[list[str]] = []
    in_table = False
    for line in text.splitlines():
        if line.startswith(LEDGER_TABLE_HEADER):
            in_table = True
            continue
        if in_table and line.startswith("|---"):
            continue
        if in_table and line.startswith("| "):
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if len(cells) == 4:
                rows.append(cells)
            continue
        if in_table:
            break
    return rows


def test_g1_every_group_08_entry_is_in_the_table() -> None:
    rows = _ledger_rows(LEDGER.read_text(encoding="utf-8"))
    assert rows, "section-4 ledger table is empty"
    listed = {row[0] for row in rows}
    ids = set()
    for path in _entry_specs():
        match = re.match(r"^(\d+)-", path.name)
        assert match, path.name
        ids.add(f"08-{int(match.group(1)):02d}")
    assert ids == listed, f"ledger table {sorted(listed)} != Group 08 entries {sorted(ids)}"
    for spec, taxonomy, home, floor in rows:
        assert taxonomy in {"optimizer", "learning rule", "filter"}, spec
        assert home, f"{spec} missing home"
        assert floor, f"{spec} missing floor"


def test_g2_rejects_are_named_and_unimplemented() -> None:
    section4 = _section(LEDGER.read_text(encoding="utf-8"), 4)
    for token in REJECTED:
        assert token in section4, f"section 4 dropped {token}"
    assert "Skip the chain rule and solve for a global min" in section4
    assert "Full `d h / d theta`" in section4
    banned = (
        "skip_chain_rule",
        "full_parameter_jacobian",
        "parameter_jacobian_flow",
    )
    hits = [
        path.relative_to(REPO).as_posix()
        for path in (REPO / "packages").rglob("*")
        if path.is_file() and any(token in path.name for token in banned)
    ]
    assert not hits, f"rejected ideas grew implementation files: {hits}"
    assert not (PACKAGES / "omnibias-training").exists()


def test_g3_no_stretch_inference() -> None:
    for path in _entry_specs():
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"CCF_STRETCH_RESIDUAL_GATE\s*=", text), path.name
        honesty = _section(text, 10)
        assert "Hilbert" in honesty, f"{path.name} honesty dropped the Hilbert floor"


def test_g4_no_global_min_or_skipped_chain_rule() -> None:
    for path in _entry_specs():
        honesty = _section(path.read_text(encoding="utf-8"), 10)
        assert re.search(r"global min(?:imum)?", honesty, re.I), (
            f"{path.name} honesty does not forbid a global min"
        )
        assert re.search(r"chain rule", honesty, re.I), (
            f"{path.name} honesty does not name the chain rule"
        )


def test_g5_recommended_stack_is_implementable() -> None:
    text = LEDGER.read_text(encoding="utf-8")
    section4 = _section(text, 4)
    for name in LEDGER_OPTIMIZERS + LEDGER_LEARNING_RULES + LEDGER_FILTERS:
        assert name in section4, f"stack / taxonomy dropped {name}"
    assert (REPO / "theory" / "03-algorithms" / "12-exact-jet-line-search.md").is_file()
    for symbol, rel, pattern in STACK_SYMBOLS:
        src = (REPO / rel).read_text(encoding="utf-8")
        assert re.search(pattern, src, re.MULTILINE), f"{rel} is missing {symbol}"
    assert "CCF_STRETCH_RESIDUAL_GATE = 1e-13" in text


def test_index_row_is_shipped() -> None:
    row = re.search(
        r"\| \[08-01 training-idea ledger\]\([^)]+\) \| (\w+) \|",
        THEORY_README.read_text(encoding="utf-8"),
    )
    assert row is not None
    assert row.group(1) == "shipped"


def test_training_ledger_self_checks() -> None:
    assert set(LEDGER_OPTIMIZERS) & set(LEDGER_LEARNING_RULES) == set()
    assert "08-09" in LEDGER_FILTERS
    assert "09-02" not in LEDGER_OPTIMIZERS + LEDGER_LEARNING_RULES + LEDGER_FILTERS
