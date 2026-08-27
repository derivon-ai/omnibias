# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Invention-ledger guards (theory 09-01 G1–G5).

The ledger is a document. These tests read markdown; they do not ship a
layer or mint a package.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
INVENTIONS = REPO / "theory" / "09-inventions"
LEDGER = INVENTIONS / "01-invention-ledger.md"
THEORY = REPO / "theory"
THEORY_README = THEORY / "README.md"
PACKAGES = REPO / "packages"

COMPARISON_HEADER = "| Entry | Taxonomy | What is new | Typical floor |"
POINTER_HEADER = "| Idea | Spec | Why not a new file |"
LEDGER_ARCHITECTURES = (
    "09-02",
    "09-03",
    "09-04",
    "09-05",
    "09-06",
    "09-07",
    "09-08",
    "09-09",
    "09-10",
    "09-11",
    "09-12",
    "09-13",
    "09-14",
    "09-15",
    "09-27",
    "09-28",
    "09-29",
)
LEDGER_LEARNING_RULES = (
    "09-16",
    "09-17",
    "09-18",
    "09-19",
    "09-20",
    "09-21",
    "09-22",
    "09-23",
)
LEDGER_EXPORTS = ("09-24", "09-25", "09-26", "09-30")
ALREADY_SPECIFIED = (
    "08-03",
    "08-05",
    "03-06",
    "03-10",
    "03-13",
    "02-06",
    "08-04",
    "08-06",
    "08-09",
    "03-02",
    "05-02",
    "04-02",
    "03-05",
)
REJECTED = (
    "skip_chain_rule_global_min",
    "full_parameter_jacobian_flow",
    "ccf_stretch_by_architecture",
    "generic_imagenet_reimplementation",
)
EXPECTED_TAXONOMY = {
    **dict.fromkeys(LEDGER_ARCHITECTURES, "architecture"),
    **dict.fromkeys(LEDGER_LEARNING_RULES, "learning rule"),
    **dict.fromkeys(LEDGER_EXPORTS, "export"),
}
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
        for path in INVENTIONS.glob("*.md")
        if path.name != "README.md" and path != LEDGER
    )


def _table_rows(text: str, header: str, n_cells: int) -> list[list[str]]:
    rows: list[list[str]] = []
    in_table = False
    for line in text.splitlines():
        if line.startswith(header):
            in_table = True
            continue
        if in_table and line.startswith("|---"):
            continue
        if in_table and line.startswith("| "):
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if len(cells) == n_cells:
                rows.append(cells)
            continue
        if in_table:
            break
    return rows


def _spec_id_from_filename(path: Path) -> str:
    match = re.match(r"^(\d+)-", path.name)
    assert match, path.name
    return f"09-{int(match.group(1)):02d}"


def _theory_file_for(spec_id: str) -> Path | None:
    group, number = spec_id.split("-")
    folders = {
        "01": "01-geometry",
        "02": "02-architectures",
        "03": "03-algorithms",
        "04": "04-bridges",
        "05": "05-applications",
        "06": "06-program",
        "07": "07-frontier",
        "08": "08-training",
        "09": "09-inventions",
    }
    folder = THEORY / folders[group]
    hits = list(folder.glob(f"{int(number):02d}-*.md"))
    return hits[0] if hits else None


def test_g1_every_group_09_entry_is_in_the_table() -> None:
    rows = _table_rows(
        LEDGER.read_text(encoding="utf-8"), COMPARISON_HEADER, 4
    )
    assert rows, "section-4 comparison table is empty"
    listed = {row[0].split()[0] for row in rows}
    ids = {_spec_id_from_filename(path) for path in _entry_specs()}
    assert ids == listed, (
        f"comparison table {sorted(listed)} != Group 09 entries {sorted(ids)}"
    )
    for entry, taxonomy, _new, floor in rows:
        spec = entry.split()[0]
        assert taxonomy == EXPECTED_TAXONOMY[spec], spec
        assert floor, f"{spec} missing floor"


def test_g2_already_specified_rows_point_at_existing_specs() -> None:
    rows = _table_rows(LEDGER.read_text(encoding="utf-8"), POINTER_HEADER, 3)
    assert rows, "already-specified pointer table is empty"
    pointed = {row[1].split()[0] for row in rows}
    assert set(ALREADY_SPECIFIED) <= pointed, (
        f"pointer table missing {sorted(set(ALREADY_SPECIFIED) - pointed)}"
    )
    missing = [spec for spec in ALREADY_SPECIFIED if _theory_file_for(spec) is None]
    assert not missing, f"already-specified ids have no theory file: {missing}"
    invention_ids = {_spec_id_from_filename(path) for path in _entry_specs()}
    overlap = invention_ids & set(ALREADY_SPECIFIED)
    assert not overlap, f"duplicate Group 09 files for pointer rows: {overlap}"


def test_g3_rejects_are_named_and_unimplemented() -> None:
    section4 = _section(LEDGER.read_text(encoding="utf-8"), 4)
    for token in REJECTED:
        assert token in section4, f"section 4 dropped {token}"
    assert "Skip the chain rule and solve for a global min" in section4
    assert "Full `d h / d theta`" in section4
    assert "CCF `1e-13`" in section4
    assert "Generic ImageNet" in section4
    banned = (
        "skip_chain_rule",
        "full_parameter_jacobian",
        "ccf_stretch_by_architecture",
        "generic_imagenet",
    )
    hits = [
        path.relative_to(REPO).as_posix()
        for path in PACKAGES.rglob("*")
        if path.is_file() and any(token in path.name for token in banned)
    ]
    assert not hits, f"rejected ideas grew implementation files: {hits}"


def test_g4_no_stretch_inference() -> None:
    for path in _entry_specs():
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"CCF_STRETCH_RESIDUAL_GATE\s*=", text), path.name


def test_g5_no_new_package() -> None:
    assert not (PACKAGES / "omnibias-invention").exists()
    assert not (PACKAGES / "omnibias-inventions").exists()
    text = LEDGER.read_text(encoding="utf-8")
    assert "Zero new packages" in text
    assert "earn independent existence" in text


def test_index_row_is_shipped() -> None:
    row = re.search(
        r"\| \[09-01 invention ledger\]\([^)]+\) \| (\w+) \|",
        THEORY_README.read_text(encoding="utf-8"),
    )
    assert row is not None
    assert row.group(1) == "shipped"


def test_invention_ledger_self_checks() -> None:
    arch = set(LEDGER_ARCHITECTURES)
    rules = set(LEDGER_LEARNING_RULES)
    exports = set(LEDGER_EXPORTS)
    assert not (arch & rules or arch & exports or rules & exports)
    assert len(arch | rules | exports) == 29
    assert "08-03" in ALREADY_SPECIFIED
    assert "09-02" not in ALREADY_SPECIFIED
