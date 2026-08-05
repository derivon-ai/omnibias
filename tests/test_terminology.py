# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""The two collapse limits must be named apart, everywhere.

omnibias has two distinct limits that used to share the word "collapse":

* **bias collapse** -- the founding ``delta -> 0`` limit in which ``K`` biases
  coalesce and the unit becomes ``sigma^(K-1)``, a smooth *derivative*;
* **temperature collapse** -- the ``beta -> inf`` limit in which one gate
  sharpens into a 0/1 *indicator*, a feasibility step.

Naming the second one "collapsed-bias" or "the bias-collapse penalty" is what
made them confusable in the first place, so those wordings are retired and this
guard keeps them retired. See ``docs/theory.md`` sec "Two senses of collapse".
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

#: Wordings that name the ``beta -> inf`` axis after the founding one.
RETIRED = re.compile(r"collapsed[- ]bias|bias[- ]collapse penalt(y|ies)", re.IGNORECASE)

#: The name every one of those sites must use instead.
CANONICAL = "temperature collapse"

SCANNED_SUFFIXES = frozenset({".py", ".md", ".mdc", ".txt", ".toml", ".yml", ".yaml"})

SCANNED_ROOTS = (
    "packages",
    "docs",
    "tests",
    "scripts",
    ".cursor/rules",
    ".cursor/skills",
    ".claude/skills",
)

SCANNED_FILES = ("AGENTS.md", "CLAUDE.md", "README.md", "llms.txt", "mkdocs.yml")

#: Files allowed to quote the retired wording, because they are what retires it.
ALLOWED = frozenset(
    {
        "tests/test_terminology.py",
        # The per-package lineage guard quotes the retired wording as the
        # synthetic violation its own self-test must detect.
        "packages/omnibias-core/tests/test_lineage_declared.py",
        ".cursor/rules/omnibias.md",
        ".cursor/skills/omnibias-dev-core-concepts/SKILL.md",
        ".claude/skills/omnibias-dev-core-concepts/SKILL.md",
        # A changelog is a historical record; past entries keep the wording they
        # shipped with. The rename is itself recorded under [Unreleased].
        "CHANGELOG.md",
    }
)

EXCLUDED_PARTS = frozenset({"__pycache__", ".venv", "node_modules", "site", ".git"})


def _tracked_files() -> list[Path]:
    seen: set[Path] = set()
    for root in SCANNED_ROOTS:
        base = REPO / root
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if path.suffix not in SCANNED_SUFFIXES or not path.is_file():
                continue
            if EXCLUDED_PARTS & set(path.parts) or ".egg-info" in str(path):
                continue
            seen.add(path)
    for name in SCANNED_FILES:
        path = REPO / name
        if path.is_file():
            seen.add(path)
    return sorted(seen)


def _offenders() -> list[tuple[str, int, str]]:
    hits: list[tuple[str, int, str]] = []
    for path in _tracked_files():
        rel = path.relative_to(REPO).as_posix()
        if rel in ALLOWED:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:  # pragma: no cover - binary stragglers
            continue
        if not RETIRED.search(text):
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if RETIRED.search(line):
                hits.append((rel, lineno, line.strip()))
    return hits


def test_the_retired_collapse_wordings_are_gone() -> None:
    offenders = _offenders()
    assert not offenders, (
        "The `beta -> inf` axis is called **temperature collapse**; naming it after the "
        "founding `delta -> 0` bias collapse is what made the two confusable.\n"
        + "\n".join(f"  {rel}:{no}: {line}" for rel, no, line in offenders)
    )


@pytest.mark.parametrize(
    "rel",
    [
        "docs/theory.md",
        ".cursor/rules/omnibias.md",
        ".cursor/skills/omnibias-dev-core-concepts/SKILL.md",
        ".claude/skills/omnibias-dev-core-concepts/SKILL.md",
        "AGENTS.md",
    ],
)
def test_the_canonical_sources_teach_the_canonical_name(rel: str) -> None:
    text = (REPO / rel).read_text(encoding="utf-8").lower()
    assert CANONICAL in text, f"{rel} must name the `beta -> inf` axis {CANONICAL!r}"


def test_the_founding_sense_keeps_its_own_name() -> None:
    """The rename must not have swallowed the `delta -> 0` limit."""
    for rel in ("docs/theory.md", "packages/omnibias-torch/src/omnibias/torch/unit.py"):
        text = (REPO / rel).read_text(encoding="utf-8").lower()
        assert "bias collapse" in text, f"{rel} lost the founding term"
        assert "delta -> 0" in text, f"{rel} lost the founding limit"


# --------------------------------------------------------------------------
# The geometric statement, and the boundary it is easy to trip over.
# --------------------------------------------------------------------------

#: "One hyperplane" alone does not identify bias collapse: a sharpened gate
#: `sigma(beta (w.x - t))` also has a single hyperplane as its decision
#: boundary, and that is *temperature* collapse. Bias collapse is specifically
#: `K` **parallel** hyperplanes coalescing. A source that offers the
#: single-hyperplane picture as the definition of bias collapse, without the
#: word "parallel" nearby, is reproducing exactly that confusion.
_SINGLE_HYPERPLANE_AS_BIAS_COLLAPSE = re.compile(
    r"(?:bias[- ]collapse[^.\n]{0,60}single[- ]hyperplane"
    r"|single[- ]hyperplane[^.\n]{0,60}bias[- ]collapse)",
    re.IGNORECASE,
)

#: Canonical sources that must carry the geometric statement in full.
GEOMETRY_SOURCES = (
    "docs/theory.md",
    "docs/operator-surface.md",
    ".cursor/skills/omnibias-dev-core-concepts/SKILL.md",
    ".claude/skills/omnibias-dev-core-concepts/SKILL.md",
)


@pytest.mark.parametrize("rel", GEOMETRY_SOURCES)
def test_the_geometric_statement_is_documented(rel: str) -> None:
    """Bias collapse must be stated geometrically, with 'parallel' spelled out."""
    text = (REPO / rel).read_text(encoding="utf-8").lower()
    assert "parallel hyperplane" in text, (
        f"{rel} must say bias collapse coalesces K *parallel* hyperplanes; "
        "'one hyperplane' alone also describes temperature collapse"
    )
    assert "antiderivative" in text, (
        f"{rel} must record that the `integral` role is the same geometry run in "
        "the antiderivative direction"
    )


def test_no_source_equates_bias_collapse_with_a_lone_hyperplane() -> None:
    """The single-hyperplane picture must never stand in for bias collapse."""
    offenders: list[tuple[str, int, str]] = []
    for path in _tracked_files():
        rel = path.relative_to(REPO).as_posix()
        if rel in ALLOWED or rel in GEOMETRY_SOURCES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:  # pragma: no cover - binary stragglers
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if not _SINGLE_HYPERPLANE_AS_BIAS_COLLAPSE.search(line):
                continue
            if "parallel" in line.lower():
                continue
            offenders.append((rel, lineno, line.strip()))
    assert not offenders, (
        "bias collapse is `K` *parallel* hyperplanes coalescing into one -- a lone "
        "hyperplane sharpened (`beta -> inf`) is temperature collapse. Say which.\n"
        + "\n".join(f"  {rel}:{no}: {line}" for rel, no, line in offenders)
    )


def test_the_hyperplane_guard_is_not_vacuous() -> None:
    """The pattern must catch the mislabel it exists to prevent."""
    mislabel = "The bias-collapse / single-hyperplane view: the hard sign is the beta -> inf limit"
    assert _SINGLE_HYPERPLANE_AS_BIAS_COLLAPSE.search(mislabel)
    assert _SINGLE_HYPERPLANE_AS_BIAS_COLLAPSE.search(
        "the single hyperplane picture of bias collapse"
    )
    assert not _SINGLE_HYPERPLANE_AS_BIAS_COLLAPSE.search(
        "temperature collapse sharpens a single hyperplane"
    )
