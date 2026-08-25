# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Theory-homes guard (theory 06-03 G1/G2/G5).

Every spec's section 2 must name an existing package, a submodule of one,
or an explicit docs-only / formal / benchmark home. New distributions are
checked against ``NEW_PACKAGES_ALLOWED``, which starts empty on purpose:
adding a package requires editing this file with a justification.

Stdlib + pytest only: this runs in the numpy-free core CI job.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
THEORY = REPO / "theory"
PACKAGES = REPO / "packages"

# Empty by design. A new distribution is a deliberate edit here, not a
# silent side-effect of implementing a spec.
NEW_PACKAGES_ALLOWED: frozenset[str] = frozenset()

# Formal Lean projects, not Python distributions.
FORMAL_PROJECTS: frozenset[str] = frozenset(
    {
        "omnibias-verified-kernel",
        "omnibias-analytic",
    }
)

# Folded names must stay gone (same set as test_package_registry.FOLDED_AWAY).
FOLDED_AWAY: frozenset[str] = frozenset(
    {
        "omnibias-pde",
        "omnibias-gauge",
        "omnibias-flow",
    }
)

# Spec 06-03 assignment table: these stay documents, not packages.
DOCS_ONLY_SPECS: frozenset[str] = frozenset(
    {
        "theory/01-geometry/13-operator-family.md",
        "theory/06-program/01-acceptance-gates-and-benchmarks.md",
        "theory/06-program/02-honesty-and-claim-boundaries.md",
        "theory/06-program/03-packaging-and-rollout.md",
        "theory/06-program/04-book-outline.md",
        "theory/06-program/05-public-primitive-and-citation-path.md",
        "theory/07-frontier/01-sub-obligation-ledger.md",
        "theory/08-training/01-training-idea-ledger.md",
        "theory/09-inventions/01-invention-ledger.md",
    }
)

_SKIP_NAMES = frozenset({"README.md", "_TEMPLATE.md"})
_SECTION_TWO = re.compile(
    r"^## 2\. Where it lands\n(.*?)(?=^## |\Z)",
    re.MULTILINE | re.DOTALL,
)
_MODULE_HOME = re.compile(r"omnibias(?:\.\{torch,jax\}|\.([A-Za-z_][\w]*))")
_DIST_TOKEN = re.compile(r"(?:packages/)?(omnibias-[a-z0-9-]+)")
_DOCS_ONLY = re.compile(
    r"(?is)(?:this (?:file|document)|this file only|docs-only|"
    r"a document|no module|nothing to build|no package\.?|"
    r"not a package|manuscript|`docs/`|docs/|benchmarks/|formal/|"
    r"a separate manuscript|documentation plus)"
)
_REJECT_CONTEXT = re.compile(
    r"(?i)(?:\bno\b|\bnot\b|resist|re-creat|do not|don't|never|"
    r"fails|would repeat|must not|must stay|not mint|without )"
)


def _package_dirs() -> set[str]:
    return {
        path.name
        for path in PACKAGES.glob("omnibias-*")
        if (path / "pyproject.toml").is_file()
    }


def _package_roots(existing: set[str]) -> set[str]:
    return {name.removeprefix("omnibias-") for name in existing}


def _theory_specs() -> list[Path]:
    return sorted(
        path
        for path in THEORY.rglob("*.md")
        if path.name not in _SKIP_NAMES
    )


def _section_two(text: str) -> str | None:
    match = _SECTION_TWO.search(text)
    return None if match is None else match.group(1)


def _has_home(body: str, existing: set[str]) -> bool:
    if _DOCS_ONLY.search(body):
        return True
    if "omnibias.{torch,jax}" in body:
        return True
    roots = _package_roots(existing)
    for match in _MODULE_HOME.finditer(body):
        root = match.group(1)
        if root is None or root in roots:
            return True
    for match in _DIST_TOKEN.finditer(body):
        name = match.group(1)
        if name in existing or name in FORMAL_PROJECTS:
            return True
    return False


def _proposed_new_packages(body: str, existing: set[str]) -> set[str]:
    proposed: set[str] = set()
    for match in _DIST_TOKEN.finditer(body):
        name = match.group(1)
        if name in existing or name in FORMAL_PROJECTS:
            continue
        start = max(0, match.start() - 120)
        window = body[start : match.end() + 40]
        if _REJECT_CONTEXT.search(window):
            continue
        proposed.add(name)
    return proposed


def test_new_packages_allowlist_starts_empty() -> None:
    """Adding a package is an edit of this constant, with a justification."""
    assert NEW_PACKAGES_ALLOWED == frozenset()


def test_every_spec_declares_a_home() -> None:
    """Each theory spec's section 2 names an existing package or an explicitly
    justified new one; the set of new packages proposed across the tree is
    checked against the allowlist in this file."""
    specs = _theory_specs()
    assert specs, "theory/ has no specs; the homes guard would go vacuous"
    existing = _package_dirs()
    missing_section: list[str] = []
    missing_home: list[str] = []
    proposed: dict[str, list[str]] = {}
    for path in specs:
        rel = path.relative_to(REPO).as_posix()
        body = _section_two(path.read_text(encoding="utf-8"))
        if body is None:
            missing_section.append(rel)
            continue
        if not _has_home(body, existing):
            missing_home.append(rel)
        extra = _proposed_new_packages(body, existing)
        if extra:
            proposed[rel] = sorted(extra)
    assert not missing_section, (
        "specs missing '## 2. Where it lands': " + ", ".join(missing_section)
    )
    assert not missing_home, (
        "section 2 does not name an existing package, submodule, or "
        "docs-only home: " + ", ".join(missing_home)
    )
    unexpected = {
        rel: names
        for rel, names in proposed.items()
        if not set(names) <= NEW_PACKAGES_ALLOWED
    }
    assert not unexpected, (
        "section 2 proposes a new distribution that is not on "
        f"NEW_PACKAGES_ALLOWED: {unexpected}"
    )


def test_docs_only_assignment_table_stays_documents() -> None:
    """06-03's docs-only row must keep a document home, not a new package."""
    existing = _package_dirs()
    missing = [rel for rel in sorted(DOCS_ONLY_SPECS) if not (REPO / rel).is_file()]
    assert not missing, f"assignment-table docs-only specs vanished: {missing}"
    for rel in sorted(DOCS_ONLY_SPECS):
        body = _section_two((REPO / rel).read_text(encoding="utf-8"))
        assert body is not None, f"{rel} has no section 2"
        assert _DOCS_ONLY.search(body), f"{rel} lost its docs-only home language"
        extra = _proposed_new_packages(body, existing)
        assert not extra, f"{rel} proposed new packages {sorted(extra)}"


def test_arrangement_and_folded_names_are_not_minted() -> None:
    """G5 / G1: arrangement stays a submodule; folded packages stay gone."""
    existing = _package_dirs()
    assert "omnibias-arrangement" not in existing
    assert not (PACKAGES / "omnibias-arrangement").exists()
    resurrected = sorted(existing & FOLDED_AWAY)
    assert not resurrected, f"folded packages reappeared: {resurrected}"
    for name in FOLDED_AWAY:
        assert not (PACKAGES / name).exists(), f"{name} directory must stay gone"


def test_wave0_a5_is_recorded_failed() -> None:
    """G4: A5 ran and failed; the sequence submodule is retired, not invented."""
    text = (THEORY / "README.md").read_text(encoding="utf-8")
    assert re.search(r"\| A5 \|", text), "Wave-0 table lost the A5 row"
    assert re.search(r"A5.*\*\*failed\*\*", text), "A5 must be recorded as failed"
    assert not re.search(r"A5.*not run", text), "A5 already ran; drop 'not run'"
    seq = (
        PACKAGES
        / "omnibias-torch"
        / "src"
        / "omnibias"
        / "torch"
        / "sequence.py"
    )
    assert not seq.exists(), "G5 failed: do not ship omnibias.torch.sequence"


def test_home_parser_self_checks() -> None:
    """Synthetic bait so the parser cannot go vacuous."""
    existing = {"omnibias-core", "omnibias-torch", "omnibias-jax"}
    assert _section_two("# no section\n") is None
    assert not _has_home("Some prose with no home.", existing)
    assert _has_home("Lives in `omnibias.core.scan`.", existing)
    assert _has_home("New submodule `omnibias.{torch,jax}.implicit`.", existing)
    assert _has_home("This document (docs-only). No new package.", existing)
    assert _has_home("`benchmarks/_gates.py` already exists.", existing)
    assert _proposed_new_packages("Ship a `omnibias-scan` package.", existing) == {
        "omnibias-scan"
    }
    assert (
        _proposed_new_packages("No `omnibias-deq` distribution.", existing) == set()
    )
    assert (
        _proposed_new_packages(
            "Resisting the urge to ship a `omnibias-jetbundle` package.",
            existing,
        )
        == set()
    )
    assert (
        _proposed_new_packages(
            "plus a Lean lemma file in `formal/omnibias-verified-kernel/`.",
            existing,
        )
        == set()
    )
