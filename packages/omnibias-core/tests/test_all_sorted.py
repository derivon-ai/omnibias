# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Release guard: every ``__all__`` export list is sorted.

The repository convention (see AGENTS.md, "Regenerate the ``__all__`` block ...
when you add or remove a public symbol") is that every module's ``__all__`` is
kept in canonical, deterministic order so diffs stay minimal and the public
surface is easy to scan. "Canonical" here is Python's default (case-sensitive /
ASCII) ``sorted()`` order, which the overwhelming majority of the tree already
follows.

This backend-free test runs in the core CI job. It walks every shipped source
module (``packages/*/src/**/*.py``), finds each module-level ``__all__`` that is
a plain literal list/tuple of string names, and fails if any is not sorted. It
deliberately ignores dynamically-built ``__all__`` (anything that is not a
literal of string constants) because those cannot be ordered statically.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def _literal_all_names(node: ast.stmt) -> list[str] | None:
    """Return the string names of a module-level ``__all__`` literal, or None.

    Returns ``None`` when *node* is not an ``__all__`` assignment, or when its
    value is not a list/tuple of string constants (dynamic ``__all__`` is out of
    scope for a static sortedness guard).
    """
    if isinstance(node, ast.Assign):
        targets, value = node.targets, node.value
    elif isinstance(node, ast.AnnAssign) and node.value is not None:
        targets, value = [node.target], node.value
    else:
        return None
    if not any(isinstance(t, ast.Name) and t.id == "__all__" for t in targets):
        return None
    if not isinstance(value, ast.List | ast.Tuple):
        return None
    names: list[str] = []
    for elt in value.elts:
        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
            names.append(elt.value)
        else:
            return None  # non-string element -> not a static literal
    return names


def _iter_source_modules() -> list[Path]:
    return [p for p in REPO_ROOT.glob("packages/*/src/**/*.py") if p.is_file()]


def test_all_export_lists_are_sorted() -> None:
    offenders: dict[str, tuple[list[str], list[str]]] = {}
    for path in _iter_source_modules():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover - shipped source must parse
            continue
        for node in tree.body:  # module-level only
            names = _literal_all_names(node)
            if names is None:
                continue
            if names != sorted(names):
                offenders[str(path.relative_to(REPO_ROOT))] = (names, sorted(names))

    assert not offenders, (
        "unsorted __all__ export lists found (sort them with Python's default "
        "sorted(), i.e. case-sensitive / ASCII order): "
        + "; ".join(
            f"{path}: {is_} -> {want}" for path, (is_, want) in sorted(offenders.items())
        )
    )
