# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Release guard: every distribution ships a wired-up ``py.typed`` marker.

omnibias is a fully type-annotated library, so every distribution must be
PEP 561 compliant: its import package must contain a ``py.typed`` marker *and*
that marker must be declared in the wheel via
``[tool.setuptools.package-data]``. A marker on disk that is not declared in
``package-data`` is silently dropped from the built wheel (setuptools does not
auto-include it), so downstream ``mypy`` would treat the package as untyped --
this guard catches both halves of that failure mode.

The test is backend-free and runs in the core CI job. It is intentionally
dependency-free (no ``tomllib``, matching ``test_package_registry.py``) so it
runs identically on every supported interpreter.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def _top_import_package(dist_dir: Path) -> Path | None:
    """Return the single ``src/omnibias/<pkg>`` import package of *dist_dir*."""
    src = dist_dir / "src" / "omnibias"
    if not src.is_dir():
        return None
    tops = [d for d in src.iterdir() if d.is_dir() and (d / "__init__.py").is_file()]
    return tops[0] if len(tops) == 1 else None


def _declares_py_typed(pyproject_text: str, modname: str) -> bool:
    """True if ``pyproject`` package-data declares ``py.typed`` for *modname*."""
    table = re.search(
        r"^\[tool\.setuptools\.package-data\]\s*$(.*?)(?=^\[|\Z)",
        pyproject_text,
        re.MULTILINE | re.DOTALL,
    )
    if table is None:
        return False
    entry = re.search(
        rf'^\s*"{re.escape(modname)}"\s*=\s*(\[.*?\])',
        table.group(1),
        re.MULTILINE | re.DOTALL,
    )
    return entry is not None and '"py.typed"' in entry.group(1)


def _distributions() -> list[Path]:
    return sorted(
        p
        for p in (REPO_ROOT / "packages").glob("omnibias-*")
        if (p / "pyproject.toml").is_file()
    )


def test_every_distribution_has_py_typed_marker() -> None:
    missing: list[str] = []
    for dist in _distributions():
        top = _top_import_package(dist)
        assert top is not None, f"{dist.name}: expected exactly one src/omnibias/<pkg>"
        if not (top / "py.typed").is_file():
            missing.append(f"{dist.name}: omnibias/{top.name}/py.typed")
    assert not missing, (
        "distributions missing a PEP 561 py.typed marker file "
        "(add an empty py.typed under the import package): " + "; ".join(missing)
    )


def test_every_py_typed_marker_is_declared_in_package_data() -> None:
    undeclared: list[str] = []
    for dist in _distributions():
        top = _top_import_package(dist)
        assert top is not None, f"{dist.name}: expected exactly one src/omnibias/<pkg>"
        text = (dist / "pyproject.toml").read_text(encoding="utf-8")
        if not _declares_py_typed(text, f"omnibias.{top.name}"):
            undeclared.append(f"{dist.name}: omnibias.{top.name}")
    assert not undeclared, (
        "py.typed marker not declared in [tool.setuptools.package-data] (it will "
        'be dropped from the wheel; add `"omnibias.<pkg>" = ["py.typed"]`): '
        + "; ".join(undeclared)
    )
