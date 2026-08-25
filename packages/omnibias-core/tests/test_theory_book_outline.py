# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Book-outline guard (theory 06-04).

The monograph stays an outline. A ``book/`` tree, a mkdocs nav entry, or a
status other than ``concept`` would mean drafting started before G1 allows it.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUTLINE = REPO / "theory" / "06-program" / "04-book-outline.md"
MKDOCS = REPO / "mkdocs.yml"
THEORY_README = REPO / "theory" / "README.md"


def test_outline_stays_concept_and_unwritten() -> None:
    text = OUTLINE.read_text(encoding="utf-8")
    assert re.search(r"^\- \*\*Status\*\*: concept$", text, re.MULTILINE), (
        "06-04 must stay concept; do not mark gated and do not draft the book"
    )
    assert "The book must not be written yet" in text
    assert "Not a package under any reading of the rule" in text


def test_book_tree_is_not_started() -> None:
    """A dedicated monograph tree is a later decision, not this spec."""
    assert not (REPO / "book").exists(), (
        "book/ must not exist until 06-04 G1 allows drafting"
    )
    mkdocs = MKDOCS.read_text(encoding="utf-8")
    nav_paths = re.findall(r":\s*([A-Za-z0-9_./-]+\.md)", mkdocs)
    booked = [path for path in nav_paths if path.startswith("book/")]
    assert not booked, f"mkdocs ships a book/ nav path: {booked}"
    assert re.search(r"(?m)^docs_dir:\s*docs$", mkdocs)


def test_index_row_keeps_the_outline_as_concept() -> None:
    text = THEORY_README.read_text(encoding="utf-8")
    row = re.search(
        r"\| \[06-04 book outline\]\([^)]+\) \| (\w+) \|",
        text,
    )
    assert row is not None, "theory/README.md lost the 06-04 index row"
    assert row.group(1) == "concept", (
        f"06-04 index status drifted to {row.group(1)!r}; it must stay concept"
    )


def test_outline_parser_self_checks() -> None:
    """Synthetic bait so a rewrite that drops the status line fails loudly."""
    assert re.search(r"^\- \*\*Status\*\*: concept$", "- **Status**: concept", re.M)
    assert not re.search(
        r"^\- \*\*Status\*\*: concept$",
        "- **Status**: gated",
        re.M,
    )
