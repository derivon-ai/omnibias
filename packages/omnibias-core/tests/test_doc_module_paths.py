# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Docs accuracy guard: dotted ``omnibias.*`` paths in the docs must resolve.

The docs are the primary context an AI assistant reads before answering a
question about omnibias, so a stale module path there becomes a confident wrong
answer downstream. This guard scans the published prose for dotted
``omnibias.<something>`` paths and checks them two ways:

1. **Package level (environment-independent).** The second component must be a
   package this repo actually ships (derived from ``packages/omnibias-*``), so a
   reference to a folded-away distribution (``omnibias.pde``, ``omnibias.gauge``,
   ``omnibias.flow``) or a typo fails everywhere, including the core CI job.
2. **Module / attribute level (environment-dependent).** When the owning package
   *is* importable, the rest of the path must resolve -- as a submodule, or as an
   attribute chain hanging off the longest real submodule prefix. Paths whose
   package is not installed in the current environment are skipped rather than
   guessed at, which is what keeps the guard low-noise: it is toothless in the
   core job and fully armed in the ``docs`` job, where every distribution is
   installed.

The walk uses :func:`importlib.util.find_spec` to decide *existence* before
importing anything, so a missing third-party backend (no JAX installed, say)
degrades to a skip instead of a false failure.
"""

from __future__ import annotations

import importlib
import importlib.util
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

#: Any dotted path rooted at ``omnibias`` with at least one component after it.
_DOTTED = re.compile(r"omnibias(?:\.[A-Za-z_][A-Za-z0-9_]*)+")

#: Prose that *looks* like a dotted module path but is not one. Each entry
#: carries the reason it can never resolve.
ALLOWLIST: frozenset[str] = frozenset(
    {
        "omnibias.md",  # the always-apply rule file .cursor/rules/omnibias.md
        "omnibias.ipynb",  # a notebook filename
        "omnibias.ai",  # the documentation domain, https://omnibias.ai/
        # TOML table paths in the root pyproject, not importable modules.
        "omnibias.license_tiers",
        "omnibias.license_expressions",
    }
)

#: Prose surfaces that must stay accurate: the published site plus the two
#: machine-facing summaries downstream agents read.
_SCAN_GLOBS = ("docs/**/*.md",)
_SCAN_FILES = ("llms.txt", "README.md")

_EXEMPT_FILES: frozenset[str] = frozenset()


def _scanned_files() -> list[Path]:
    files = [p for pattern in _SCAN_GLOBS for p in REPO_ROOT.glob(pattern)]
    files += [REPO_ROOT / name for name in _SCAN_FILES]
    return sorted(
        p
        for p in files
        if p.is_file() and str(p.relative_to(REPO_ROOT)) not in _EXEMPT_FILES
    )


def _shipped_packages() -> frozenset[str]:
    """Import-module names shipped by this repo (``omnibias.<name>``)."""
    prefix = "omnibias-"
    return frozenset(
        p.name[len(prefix) :]
        for p in (REPO_ROOT / "packages").glob(f"{prefix}*")
        if (p / "pyproject.toml").is_file()
    )


def _mentions() -> dict[str, list[str]]:
    """Map every dotted path found in the docs to the files that mention it."""
    found: dict[str, list[str]] = {}
    for path in _scanned_files():
        rel = str(path.relative_to(REPO_ROOT))
        for match in _DOTTED.findall(path.read_text(encoding="utf-8")):
            if match in ALLOWLIST:
                continue
            where = found.setdefault(match, [])
            if rel not in where:
                where.append(rel)
    return found


def _longest_real_module(parts: list[str]) -> int:
    """Length of the longest prefix of *parts* that names an existing module."""
    for depth in range(len(parts), 1, -1):
        try:
            spec = importlib.util.find_spec(".".join(parts[:depth]))
        except (ImportError, AttributeError, ValueError):
            # A parent could not be imported here (missing backend) or is a
            # plain module, not a package -- keep shortening.
            continue
        if spec is not None:
            return depth
    return 0


def _unresolved_reason(dotted: str) -> str | None:
    """Why *dotted* fails to resolve, or ``None`` when it is fine / undecidable."""
    parts = dotted.split(".")
    root = ".".join(parts[:2])
    try:
        if importlib.util.find_spec(root) is None:
            return None  # package not installed in this environment
    except (ImportError, ValueError):
        return None
    if len(parts) == 2:
        return None

    depth = _longest_real_module(parts)
    if depth == 0:
        return None  # the package itself will not import here
    if depth == len(parts):
        return None  # the whole path is a real module

    module_path = ".".join(parts[:depth])
    try:
        obj: object = importlib.import_module(module_path)
    except Exception:  # noqa: BLE001 - cannot introspect here, so do not guess
        return None
    for name in parts[depth:]:
        if not hasattr(obj, name):
            return f"{module_path!r} has no attribute {name!r}"
        obj = getattr(obj, name)
    return None


def test_docs_reference_only_shipped_packages() -> None:
    shipped = _shipped_packages()
    assert "core" in shipped, "expected packages/omnibias-core to be discoverable"
    bad: list[str] = []
    for dotted, files in sorted(_mentions().items()):
        package = dotted.split(".")[1]
        if package not in shipped:
            bad.append(f"{dotted} ({', '.join(files)})")
    assert not bad, (
        "docs reference an omnibias package this repo does not ship (fix the path, "
        "or add it to ALLOWLIST with a reason if it is prose): " + "; ".join(bad)
    )


def test_docs_dotted_paths_resolve() -> None:
    mentions = _mentions()
    assert mentions, "expected the docs to mention at least one omnibias module"
    bad: list[str] = []
    for dotted, files in sorted(mentions.items()):
        reason = _unresolved_reason(dotted)
        if reason is not None:
            bad.append(f"{dotted}: {reason} ({', '.join(files)})")
    assert not bad, (
        "docs reference omnibias paths that do not resolve in an environment where "
        "their package is installed: " + "; ".join(bad)
    )
