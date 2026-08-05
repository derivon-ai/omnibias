#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Stamp every Python file with the SPDX header its licence tier requires.

omnibias ships an open-core split: the tier of each distribution is recorded in
``[tool.omnibias.license_tiers]`` in the repository root ``pyproject.toml``, and
that table is the single source of truth. This script projects it onto the file
headers.

    permissive -> # SPDX-License-Identifier: Apache-2.0
    copyleft   -> # SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial

Files outside ``packages/`` (repo-root ``scripts/``, ``tests/``, ``examples/``,
``docs/``) are repository infrastructure rather than a shipped distribution;
they are stamped permissive so that copying a snippet out of an example never
drags the copyleft obligation along with it.

Usage::

    python scripts/license_headers.py            # rewrite in place
    python scripts/license_headers.py --check    # report only, non-zero on drift

``packages/omnibias-core/tests/test_license_consistency.py`` is the enforcing
guard; this script is the fixer. Keep them agreeing.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import tomllib

REPO_ROOT = Path(__file__).resolve().parent.parent

SPDX_LINE = re.compile(r"^#\s*SPDX-License-Identifier:\s*(?P<expr>.+?)\s*$")
COPYRIGHT_LINE = re.compile(r"^#\s*Copyright\b.*$")
SHEBANG = re.compile(r"^#!")
CODING = re.compile(r"^#.*coding[:=]")

COPYRIGHT = "# Copyright (C) 2026 Derivon"

#: Directories that are never part of the shipped surface.
SKIP_PARTS = frozenset(
    {
        ".git",
        ".venv",
        "__pycache__",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        "site",
        ".lake",
        "artifacts",
        "build",
        "dist",
        "node_modules",
    }
)

#: Repo-root trees that are stamped, in addition to ``packages/``.
ROOT_TREES = ("scripts", "tests", "examples", "docs", "notebooks", "benchmarks")


def load_tiers() -> tuple[dict[str, str], dict[str, str]]:
    """Return ``(dist -> tier, tier -> SPDX expression)`` from the root pyproject."""
    cfg = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    omnibias = cfg["tool"]["omnibias"]
    return dict(omnibias["license_tiers"]), dict(omnibias["license_expressions"])


def package_dist_name(pkg_dir: Path) -> str:
    data = tomllib.loads((pkg_dir / "pyproject.toml").read_text(encoding="utf-8"))
    name: str = data["project"]["name"]
    return name


def owning_package(path: Path) -> str | None:
    """The distribution directory name owning ``path``, or ``None`` for repo infra."""
    rel = path.relative_to(REPO_ROOT).parts
    if len(rel) >= 2 and rel[0] == "packages":
        return rel[1]
    return None


def target_files() -> list[Path]:
    out: list[Path] = []
    roots = [REPO_ROOT / "packages"] + [REPO_ROOT / t for t in ROOT_TREES]
    for root in roots:
        if not root.is_dir():
            continue
        for f in sorted(root.rglob("*.py")):
            if SKIP_PARTS & set(f.relative_to(REPO_ROOT).parts):
                continue
            out.append(f)
    for f in sorted(REPO_ROOT.glob("*.py")):
        out.append(f)
    return out


def expected_expression(
    path: Path, tiers: dict[str, str], expressions: dict[str, str]
) -> str:
    pkg = owning_package(path)
    if pkg is None:
        # Repository infrastructure: permissive, so snippets stay copyable.
        return expressions["permissive"]
    dist = package_dist_name(REPO_ROOT / "packages" / pkg)
    tier = tiers.get(dist)
    if tier is None:
        raise SystemExit(
            f"{dist} is missing from [tool.omnibias.license_tiers] in pyproject.toml"
        )
    return expressions[tier]


def rewrite(text: str, expression: str) -> str:
    """Return ``text`` with exactly one correct SPDX + copyright header on top."""
    lines = text.splitlines(keepends=True)

    # Preserve a shebang and/or coding declaration at the very top.
    prefix: list[str] = []
    i = 0
    if i < len(lines) and SHEBANG.match(lines[i]):
        prefix.append(lines[i])
        i += 1
    if i < len(lines) and CODING.match(lines[i]):
        prefix.append(lines[i])
        i += 1

    # Drop any existing SPDX / copyright lines in the leading comment block.
    while i < len(lines):
        line = lines[i]
        if SPDX_LINE.match(line) or COPYRIGHT_LINE.match(line):
            i += 1
            continue
        break

    header = f"# SPDX-License-Identifier: {expression}\n{COPYRIGHT}\n"
    return "".join(prefix) + header + "".join(lines[i:])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="report drift without writing"
    )
    args = parser.parse_args()

    tiers, expressions = load_tiers()
    drifted: list[tuple[str, str]] = []

    for path in target_files():
        expression = expected_expression(path, tiers, expressions)
        original = path.read_text(encoding="utf-8")
        updated = rewrite(original, expression)
        if updated == original:
            continue
        rel = str(path.relative_to(REPO_ROOT))
        found = next(
            (
                m.group("expr")
                for line in original.splitlines()[:5]
                if (m := SPDX_LINE.match(line))
            ),
            "<none>",
        )
        drifted.append((rel, f"{found} -> {expression}"))
        if not args.check:
            path.write_text(updated, encoding="utf-8")

    for rel, change in drifted:
        print(f"  {rel}: {change}")
    verb = "would restamp" if args.check else "restamped"
    print(f"{verb} {len(drifted)} file(s)")
    if args.check and drifted:
        print("run `python scripts/license_headers.py` to fix", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
