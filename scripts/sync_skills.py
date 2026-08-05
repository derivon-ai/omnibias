#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Mirror the repo-only maintainer skills into the Claude Code skills directory.

The maintainer library (``omnibias-dev-*``) is hand-authored **canonically** in
``.cursor/skills/``. Both Cursor and Claude Code read the same Agent-Skill
``SKILL.md`` format, so this script keeps a byte-identical copy under
``.claude/skills/``. The consumer ``omnibias-*`` skills are *not* touched here --
those are owned by the ``omnibias-skills`` package installer and its own
``--check`` drift gate.

Usage::

    python scripts/sync_skills.py            # write / refresh the .claude mirror
    python scripts/sync_skills.py --check    # exit non-zero if the mirror drifts
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PREFIX = "omnibias-dev-"
_SKILL_FILE = "SKILL.md"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _dev_skill_dirs(cursor_skills: Path) -> list[Path]:
    if not cursor_skills.is_dir():
        return []
    return sorted(
        p for p in cursor_skills.iterdir() if p.is_dir() and p.name.startswith(_PREFIX)
    )


def sync(repo_root: Path, *, check: bool) -> int:
    cursor_skills = repo_root / ".cursor" / "skills"
    claude_skills = repo_root / ".claude" / "skills"

    drifted: list[str] = []
    written: list[str] = []
    for src_dir in _dev_skill_dirs(cursor_skills):
        src = src_dir / _SKILL_FILE
        if not src.is_file():
            continue
        content = src.read_text(encoding="utf-8")
        dst = claude_skills / src_dir.name / _SKILL_FILE
        current = dst.read_text(encoding="utf-8") if dst.is_file() else None
        if current == content:
            continue
        if check:
            drifted.append(str(dst.relative_to(repo_root)))
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(content, encoding="utf-8")
        written.append(str(dst.relative_to(repo_root)))

    if check:
        if drifted:
            print("Claude maintainer-skill mirror is stale; run: python scripts/sync_skills.py")
            for path in drifted:
                print(f"  DRIFT: {path}")
            return 1
        print("maintainer-skill mirror: up to date")
        return 0

    for path in written:
        print(f"  synced: {path}")
    print(f"maintainer-skill mirror: {len(written)} written")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Mirror omnibias-dev-* skills to .claude/skills.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero if the .claude mirror drifts from the .cursor canonical",
    )
    args = parser.parse_args(argv)
    return sync(_repo_root(), check=args.check)


if __name__ == "__main__":
    sys.exit(main())
