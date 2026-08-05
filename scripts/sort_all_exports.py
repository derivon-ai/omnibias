#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Re-sort every module-level ``__all__`` list into Python's default sorted order.

``packages/omnibias-core/tests/test_all_sorted.py`` is the enforcing guard; this
script is the fixer. Only literal ``__all__ = [...]`` assignments of plain string
constants are rewritten -- anything computed is left untouched.

    python scripts/sort_all_exports.py [--check]
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

SKIP_PARTS = {
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
}

REPO_ROOT = Path(__file__).resolve().parent.parent


def _all_node(tree: ast.Module) -> ast.Assign | None:
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets
        ):
            return node
    return None


def sort_file(path: Path) -> bool:
    """Rewrite ``path``'s ``__all__`` if unsorted. Returns True when it changed."""
    src = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return False
    node = _all_node(tree)
    if node is None or not isinstance(node.value, ast.List):
        return False
    if not all(isinstance(e, ast.Constant) and isinstance(e.value, str) for e in node.value.elts):
        return False

    current = [e.value for e in node.value.elts]
    wanted = sorted(current)
    if current == wanted:
        return False

    lines = src.splitlines(keepends=True)
    body = "".join(f'    "{n}",\n' for n in wanted)
    new = "".join(lines[: node.lineno - 1]) + "__all__ = [\n" + body + "]\n" + "".join(lines[node.end_lineno :])
    path.write_text(new, encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="report without writing")
    args = parser.parse_args()

    offenders: list[str] = []
    for path in sorted(REPO_ROOT.rglob("*.py")):
        if any(part in SKIP_PARTS for part in path.parts):
            continue
        if args.check:
            src = path.read_text(encoding="utf-8")
            try:
                node = _all_node(ast.parse(src))
            except SyntaxError:
                continue
            if node is None or not isinstance(node.value, ast.List):
                continue
            if not all(
                isinstance(e, ast.Constant) and isinstance(e.value, str) for e in node.value.elts
            ):
                continue
            names = [e.value for e in node.value.elts]
            if names != sorted(names):
                offenders.append(str(path.relative_to(REPO_ROOT)))
        elif sort_file(path):
            offenders.append(str(path.relative_to(REPO_ROOT)))

    verb = "unsorted" if args.check else "sorted"
    for rel in offenders:
        print(f"  {verb}: {rel}")
    print(f"{len(offenders)} file(s) {verb}")
    return 1 if (args.check and offenders) else 0


if __name__ == "__main__":
    sys.exit(main())
