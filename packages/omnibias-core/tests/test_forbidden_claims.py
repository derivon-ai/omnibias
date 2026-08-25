# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Forbidden-claims register (theory 06-02 G2/G6).

Regex over text is a floor, not a proof of honesty. Each pattern carries
synthetic bait so the guard cannot go vacuous. A match on a line that is
already a *denial* (never write / does not claim / external) is not a hit.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from omnibias.core.proof.certificate import (
    CERTIFICATE_SCHEMA_VERSION,
    make_certificate,
    schema_errors_v1,
)

REPO = Path(__file__).resolve().parents[3]

# (name, pattern, bait the pattern must catch)
FORBIDDEN: tuple[tuple[str, str, str], ...] = (
    (
        "ns_global",
        r"prove[sd]?\s+global\s+regularity",
        "we proved global regularity for the continuum NSE",
    ),
    (
        "ym_gap",
        r"prove[sd]?\s+the\s+(Yang-Mills\s+)?mass\s+gap",
        "we proved the Yang-Mills mass gap",
    ),
    (
        "rh",
        r"prove[sd]?\s+(the\s+)?Riemann\s+Hypothesis",
        "we proved the Riemann Hypothesis",
    ),
    (
        "p_eq_np",
        r"\bP\s*=\s*NP\b",
        "therefore P = NP and the decoder is exact",
    ),
    (
        "dirichlet_continuation",
        r"analytic\s+continuation\s+of\s+.*(zeta|Dirichlet)",
        "this is analytic continuation of the zeta function past Re(s)=1",
    ),
)

DENIAL = re.compile(
    r"never write|never a |does not claim|must never|not licensed|"
    r"write instead|external obligation|forbidden|not a `?P\s*=\s*NP|"
    r"tao \(2009\)|historical citation|quote the forbidden|"
    r"imply|yes-if|not an exact|would be|no P=NP|no `P = NP|"
    r"not analytic|P=NP claim|P = NP claim|P=NP limit",
    re.IGNORECASE,
)

SCANNED_SUFFIXES = frozenset({".py", ".md", ".mdc", ".txt"})
SCANNED_ROOTS = (
    "packages",
    "docs",
    "theory",
    "tests",
    "scripts",
    ".cursor/rules",
    ".cursor/skills",
    ".claude/skills",
)
SCANNED_FILES = ("AGENTS.md", "CLAUDE.md", "README.md", "llms.txt")
EXCLUDED_PARTS = frozenset({"__pycache__", ".venv", "node_modules", "site", ".git"})

# Files allowed to quote an affirmative forbidden sentence. Each entry
# needs a reason comment on the preceding source line (G6 / ALLOWED).
ALLOWED = frozenset(
    {
        # This file quotes bait strings so the self-test can fire.
        "packages/omnibias-core/tests/test_forbidden_claims.py",
        # The honesty spec's register table quotes the never-write sentences.
        "theory/06-program/02-honesty-and-claim-boundaries.md",
        # Ledger quotes the never-write column as the thing to refuse.
        "theory/07-frontier/01-sub-obligation-ledger.md",
        # Public docs page repeats the register so a reader sees the sentences.
        "docs/honesty.md",
        # Public ledger quotes the never-write column as the thing to refuse.
        "docs/frontier-ledger.md",
    }
)


def _tracked_files() -> list[Path]:
    seen: set[Path] = set()
    for root in SCANNED_ROOTS:
        base = REPO / root
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if path.suffix not in SCANNED_SUFFIXES or not path.is_file():
                continue
            if EXCLUDED_PARTS & set(path.parts):
                continue
            seen.add(path)
    for name in SCANNED_FILES:
        path = REPO / name
        if path.is_file():
            seen.add(path)
    return sorted(seen)


def _rel(path: Path) -> str:
    return path.relative_to(REPO).as_posix()


def _hits(text: str, pattern: re.Pattern[str], *, name: str = "") -> list[str]:
    lines: list[str] = []
    prev = ""
    for line in text.splitlines():
        window = f"{prev} {line}"
        if not pattern.search(line):
            prev = line
            continue
        if DENIAL.search(window):
            prev = line
            continue
        if name == "p_eq_np" and re.search(
            r"claim|imply|never|not |without|disclaim|or P|exact-|exact argmin|"
            r"P=NP;|no P|subtext|solved;",
            window,
            re.IGNORECASE,
        ):
            prev = line
            continue
        lines.append(line.strip())
        prev = line
    return lines


def _compile(name: str, raw: str) -> re.Pattern[str]:
    flags = 0 if name == "p_eq_np" else re.IGNORECASE
    return re.compile(raw, flags)


def test_every_pattern_catches_its_bait() -> None:
    for name, raw, bait in FORBIDDEN:
        compiled = _compile(name, raw)
        assert compiled.search(bait), f"{name} does not match its bait"
        assert not DENIAL.search(bait), f"{name} bait is accidentally a denial"


def test_forbidden_set_is_not_empty() -> None:
    assert len(FORBIDDEN) >= 5


def test_tree_has_no_affirmative_forbidden_claim() -> None:
    compiled = [(name, _compile(name, raw)) for name, raw, _bait in FORBIDDEN]
    violations: list[str] = []
    for path in _tracked_files():
        rel = _rel(path)
        if rel in ALLOWED:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for name, pattern in compiled:
            for line in _hits(text, pattern, name=name):
                violations.append(f"{rel} [{name}]: {line}")
    assert violations == [], "affirmative forbidden claims:\n" + "\n".join(violations)


def test_allowed_entries_have_reason_comments() -> None:
    src = Path(__file__).read_text(encoding="utf-8")
    lines = src.splitlines()
    for rel in ALLOWED:
        needle = f'"{rel}"'
        idx = next((i for i, line in enumerate(lines) if needle in line), None)
        assert idx is not None, f"{rel} missing from ALLOWED literal"
        prev = lines[idx - 1].strip() if idx else ""
        assert prev.startswith("#"), f"{rel} ALLOWED entry needs a reason comment"


def test_pade_does_not_claim_dirichlet_continuation() -> None:
    """03-10 Padé / Borel must not claim continuation of a Dirichlet series."""
    pattern = re.compile(
        r"analytic\s+continuation\s+of\s+.*(zeta|Dirichlet|L-function)",
        re.IGNORECASE,
    )
    roots = (
        REPO / "packages" / "omnibias-symbolic",
        REPO / "packages" / "omnibias-core" / "src" / "omnibias" / "core" / "verified",
        REPO / "theory" / "03-algorithms",
        REPO / "docs" / "api",
    )
    hits: list[str] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.suffix not in {".py", ".md"} or not path.is_file():
                continue
            text = path.read_text(encoding="utf-8")
            for line in _hits(text, pattern, name="dirichlet_continuation"):
                hits.append(f"{_rel(path)}: {line}")
    assert hits == [], "Padé/Dirichlet continuation claim:\n" + "\n".join(hits)


def test_reserved_honesty_key_still_rejected() -> None:
    """06-02 G4: theorem_prover_verified cannot be sealed by a producer."""
    with pytest.raises(ValueError, match="theorem_prover_verified"):
        make_certificate(
            claim="forged",
            payload={"type": "interval"},
            honesty={"theorem_prover_verified": True},
        )
    forged = {
        "schema_version": CERTIFICATE_SCHEMA_VERSION,
        "claim": "hand-built",
        "payload": {"type": "interval"},
        "honesty": {"theorem_prover_verified": True},
    }
    assert any("theorem_prover_verified" in e for e in schema_errors_v1(forged))
