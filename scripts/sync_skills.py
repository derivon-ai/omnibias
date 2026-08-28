#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Mirror ``.cursor/skills/omnibias-*`` into ``.claude/skills/``.

Canonical Agent Skills live under ``.cursor/skills/``. Both Cursor and Claude
Code read the same ``SKILL.md`` format, so this script keeps a byte-identical
copy under ``.claude/skills/``. Capability skills that ``omnibias-skills``
ships are the same files; the installer drift gate compares the bundle to
these copies.

Usage::

    python scripts/sync_skills.py            # write / refresh the .claude mirror
    python scripts/sync_skills.py --check    # exit non-zero if the mirror drifts
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PREFIX = "omnibias-"
_SKILL_FILE = "SKILL.md"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _skill_dirs(cursor_skills: Path) -> list[Path]:
    if not cursor_skills.is_dir():
        return []
    return sorted(
        p
        for p in cursor_skills.iterdir()
        if p.is_dir() and p.name.startswith(_PREFIX) and (p / _SKILL_FILE).is_file()
    )


def sync(repo_root: Path, *, check: bool) -> int:
    cursor_skills = repo_root / ".cursor" / "skills"
    claude_skills = repo_root / ".claude" / "skills"

    drifted: list[str] = []
    written: list[str] = []
    stale: list[str] = []
    expected_names = {p.name for p in _skill_dirs(cursor_skills)}
    if claude_skills.is_dir():
        for dst_dir in claude_skills.iterdir():
            if (
                dst_dir.is_dir()
                and dst_dir.name.startswith(_PREFIX)
                and dst_dir.name not in expected_names
            ):
                rel = str((dst_dir / _SKILL_FILE).relative_to(repo_root))
                if check:
                    stale.append(rel)
                else:
                    skill = dst_dir / _SKILL_FILE
                    if skill.is_file():
                        skill.unlink()
                    try:
                        dst_dir.rmdir()
                    except OSError:
                        pass
                    written.append(f"removed {rel}")

    for src_dir in _skill_dirs(cursor_skills):
        src = src_dir / _SKILL_FILE
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
        problems = drifted + [f"STALE: {p}" for p in stale]
        if problems:
            print("Claude skill mirror is stale; run: python scripts/sync_skills.py")
            for path in problems:
                print(f"  DRIFT: {path}")
            return 1
        print("skill mirror: up to date")
        return 0

    for path in written:
        print(f"  synced: {path}")
    print(f"skill mirror: {len(written)} written")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Mirror omnibias-* skills to .claude/skills.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero if the .claude mirror drifts from the .cursor canonical",
    )
    args = parser.parse_args(argv)
    return sync(_repo_root(), check=args.check)


if __name__ == "__main__":
    sys.exit(main())
