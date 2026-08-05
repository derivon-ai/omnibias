# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Tests for the idempotent, namespaced skill installer."""

from __future__ import annotations

from pathlib import Path

import pytest
from omnibias.skills import (
    bundled_skills,
    check_skills,
    install_skills,
    uninstall_skills,
)

_N_SKILLS = len(bundled_skills())


def _skill_files(base: Path, tool: str) -> list[Path]:
    root = base / f".{tool}" / "skills"
    return sorted(root.glob("*/SKILL.md"))


def test_install_creates_all_skills(tmp_path: Path) -> None:
    result = install_skills(tmp_path, ("cursor",))
    assert len(result.written) == _N_SKILLS
    assert not result.drifted
    files = _skill_files(tmp_path, "cursor")
    assert len(files) == _N_SKILLS
    # content is byte-identical to the bundled source
    for info in bundled_skills():
        assert (tmp_path / ".cursor" / "skills" / info.name / "SKILL.md").is_file()


def test_install_all_tools(tmp_path: Path) -> None:
    install_skills(tmp_path, ("cursor", "claude"))
    assert len(_skill_files(tmp_path, "cursor")) == _N_SKILLS
    assert len(_skill_files(tmp_path, "claude")) == _N_SKILLS


def test_install_is_idempotent(tmp_path: Path) -> None:
    install_skills(tmp_path, ("cursor",))
    second = install_skills(tmp_path, ("cursor",))
    assert not second.written
    assert len(second.skipped) == _N_SKILLS


def test_check_passes_after_install(tmp_path: Path) -> None:
    install_skills(tmp_path, ("cursor", "claude"))
    report = check_skills(tmp_path, ("cursor", "claude"))
    assert report.ok
    assert not report.drifted


def test_check_flags_missing_as_drift(tmp_path: Path) -> None:
    report = check_skills(tmp_path, ("cursor",))
    assert not report.ok
    assert len(report.drifted) == _N_SKILLS


def test_check_flags_edited_file(tmp_path: Path) -> None:
    install_skills(tmp_path, ("cursor",))
    victim = tmp_path / ".cursor" / "skills" / "omnibias-verify" / "SKILL.md"
    victim.write_text(victim.read_text(encoding="utf-8") + "\n<!-- tampered -->\n", encoding="utf-8")
    report = check_skills(tmp_path, ("cursor",))
    assert not report.ok
    assert victim in report.drifted


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    result = install_skills(tmp_path, ("cursor",), dry_run=True)
    assert len(result.written) == _N_SKILLS
    assert not (tmp_path / ".cursor").exists()


def test_manifest_written(tmp_path: Path) -> None:
    import json

    install_skills(tmp_path, ("cursor", "claude"))
    manifest = tmp_path / ".omnibias-skills.manifest.json"
    assert manifest.is_file()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["omnibias_skills_version"] == "0.1.0a1"
    assert payload["tools"] == ["cursor", "claude"]
    assert len(payload["files"]) == 2 * _N_SKILLS


def test_uninstall_removes_only_managed(tmp_path: Path) -> None:
    # a user's own skill must survive install + uninstall untouched
    mine = tmp_path / ".cursor" / "skills" / "my-own" / "SKILL.md"
    mine.parent.mkdir(parents=True)
    mine.write_text("mine\n", encoding="utf-8")

    install_skills(tmp_path, ("cursor",))
    assert mine.read_text(encoding="utf-8") == "mine\n"

    removed = uninstall_skills(tmp_path, ("cursor",))
    assert len(removed.removed) == _N_SKILLS
    assert mine.is_file()  # not clobbered
    assert not (tmp_path / ".omnibias-skills.manifest.json").exists()
    # only the user's own skill survives; every omnibias-* skill is gone
    assert _skill_files(tmp_path, "cursor") == [mine]


def test_unknown_tool_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown tool"):
        install_skills(tmp_path, ("bogus",))
