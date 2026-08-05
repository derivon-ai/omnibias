# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""omnibias-skills: the consumer agent-skill library for building on omnibias.

This package bundles Cursor / Claude Code *Agent Skills* that teach an AI
coding assistant how to **use** omnibias correctly, and an idempotent installer
that places them into a project's ``.cursor/skills`` and ``.claude/skills``
directories. It has no runtime dependencies and no import-time side effects:
skills are written only through an explicit :func:`install_skills` call (or the
``omnibias-skills`` console script).

Skills for *developing omnibias itself* are a separate, repo-only library
(``.cursor/skills/omnibias-dev-*``) and are intentionally not shipped here.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError as _PkgNotFound
from importlib.metadata import version as _pkg_version

from omnibias.skills._installer import (
    TOOLS,
    InstallResult,
    SkillInfo,
    bundled_skills,
    check_skills,
    install_skills,
    uninstall_skills,
)

try:
    __version__ = _pkg_version("omnibias-skills")
except _PkgNotFound:  # pragma: no cover - bare source checkout
    __version__ = "0.0.0+unknown"

# Founding-idea lineage (see docs/theory.md "Two senses of collapse").
__lineage__ = "exempt: infrastructure"

__all__ = [
    "InstallResult",
    "SkillInfo",
    "TOOLS",
    "__lineage__",
    "__version__",
    "bundled_skills",
    "check_skills",
    "install_skills",
    "uninstall_skills",
]
