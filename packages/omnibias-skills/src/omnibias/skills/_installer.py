# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Idempotent installer for the bundled omnibias consumer skills.

The bundled skills live in ``_bundled/skills/<name>/SKILL.md`` (one canonical
copy). Both Cursor and Claude Code consume the same Agent-Skill ``SKILL.md``
format, so a single source is written to each tool's directory rather than
kept twice. Everything here is pure standard library and free of import-time
side effects.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

TOOLS: tuple[str, ...] = ("cursor", "claude")
Tool = Literal["cursor", "claude"]

_MANIFEST_NAME = ".omnibias-skills.manifest.json"
_SKILL_FILE = "SKILL.md"


def _bundle_dir() -> Path:
    return Path(__file__).resolve().parent / "_bundled"


def _skills_source() -> Path:
    return _bundle_dir() / "skills"


@dataclass(frozen=True)
class SkillInfo:
    """Metadata parsed from a bundled skill's ``SKILL.md`` frontmatter."""

    name: str
    description: str


@dataclass
class InstallResult:
    """Outcome of an install / check / uninstall pass over the target tools."""

    written: list[Path] = field(default_factory=list)
    skipped: list[Path] = field(default_factory=list)
    drifted: list[Path] = field(default_factory=list)
    removed: list[Path] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when nothing has drifted from the bundled source of truth."""
        return not self.drifted


def _parse_frontmatter(text: str) -> dict[str, str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    meta: dict[str, str] = {}
    for line in lines[1:]:
        if line.strip() == "---":
            break
        if ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip().strip('"').strip("'")
    return meta


def _skill_dirs() -> list[Path]:
    root = _skills_source()
    return sorted(p for p in root.iterdir() if p.is_dir())


def bundled_skills() -> list[SkillInfo]:
    """Return the ``(name, description)`` of every bundled consumer skill."""
    infos: list[SkillInfo] = []
    for skill_dir in _skill_dirs():
        meta = _parse_frontmatter((skill_dir / _SKILL_FILE).read_text(encoding="utf-8"))
        infos.append(
            SkillInfo(
                name=meta.get("name", skill_dir.name),
                description=meta.get("description", ""),
            )
        )
    return infos


def _planned_files() -> dict[str, str]:
    """Map ``<skill-name>/SKILL.md`` (relative to a skills root) to its content."""
    plan: dict[str, str] = {}
    for skill_dir in _skill_dirs():
        plan[f"{skill_dir.name}/{_SKILL_FILE}"] = (
            (skill_dir / _SKILL_FILE).read_text(encoding="utf-8")
        )
    return plan


def _base_dir(dest: Path | str, *, global_: bool) -> Path:
    return Path.home() if global_ else Path(dest)


def _skills_root(base: Path, tool: str) -> Path:
    return base / f".{tool}" / "skills"


def _validate_tools(tools: Sequence[str]) -> None:
    for tool in tools:
        if tool not in TOOLS:
            raise ValueError(f"unknown tool {tool!r}; expected one of {TOOLS}")


def _write_manifest(base: Path, tools: Sequence[str], plan: dict[str, str]) -> None:
    from omnibias.skills import __version__

    manifest = {
        "omnibias_skills_version": __version__,
        "tools": list(tools),
        "files": sorted(
            f".{tool}/skills/{rel}" for tool in tools for rel in plan
        ),
    }
    (base / _MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def _apply(
    dest: Path | str,
    tools: Sequence[str],
    *,
    global_: bool,
    force: bool,
    dry_run: bool,
    check: bool,
) -> InstallResult:
    _validate_tools(tools)
    base = _base_dir(dest, global_=global_)
    plan = _planned_files()
    result = InstallResult()
    for tool in tools:
        root = _skills_root(base, tool)
        for rel, content in plan.items():
            target = root / rel
            current = target.read_text(encoding="utf-8") if target.is_file() else None
            if current == content and not force:
                result.skipped.append(target)
                continue
            if check:
                result.drifted.append(target)
                continue
            if dry_run:
                result.written.append(target)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            result.written.append(target)
    if not check and not dry_run:
        _write_manifest(base, tools, plan)
    return result


def install_skills(
    dest: Path | str = ".",
    tools: Sequence[str] = TOOLS,
    *,
    global_: bool = False,
    force: bool = False,
    dry_run: bool = False,
) -> InstallResult:
    """Write the bundled skills into each tool's ``.<tool>/skills`` directory.

    Idempotent: a skill whose content already matches is skipped. Only
    ``omnibias-*`` skill directories are ever created or updated; existing user
    files are never touched.
    """
    return _apply(dest, tools, global_=global_, force=force, dry_run=dry_run, check=False)


def check_skills(
    dest: Path | str = ".",
    tools: Sequence[str] = TOOLS,
    *,
    global_: bool = False,
) -> InstallResult:
    """Compare on-disk skills against the bundle without writing anything.

    The returned result's :attr:`InstallResult.drifted` is non-empty (and
    :attr:`InstallResult.ok` is ``False``) when any target is missing or stale.
    """
    return _apply(dest, tools, global_=global_, force=False, dry_run=True, check=True)


def uninstall_skills(
    dest: Path | str = ".",
    tools: Sequence[str] = TOOLS,
    *,
    global_: bool = False,
    dry_run: bool = False,
) -> InstallResult:
    """Remove the installed ``omnibias-*`` skills and the manifest."""
    _validate_tools(tools)
    base = _base_dir(dest, global_=global_)
    plan = _planned_files()
    result = InstallResult()
    for tool in tools:
        root = _skills_root(base, tool)
        for rel in plan:
            target = root / rel
            if not target.is_file():
                continue
            if not dry_run:
                target.unlink()
                try:
                    target.parent.rmdir()
                except OSError:
                    pass
            result.removed.append(target)
    manifest = base / _MANIFEST_NAME
    if manifest.is_file() and not dry_run:
        manifest.unlink()
    return result
