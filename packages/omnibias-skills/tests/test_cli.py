# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Tests for the ``omnibias-skills`` command-line entry point."""

from __future__ import annotations

from pathlib import Path

from omnibias.skills.cli import main


def test_list_runs(capsys) -> None:  # type: ignore[no-untyped-def]
    assert main(["list"]) == 0
    out = capsys.readouterr().out
    assert "omnibias-backends" in out


def test_install_then_check_roundtrip(tmp_path: Path) -> None:
    assert main(["install", "--tool", "cursor", "--dest", str(tmp_path)]) == 0
    # check passes on a fresh, matching install
    assert main(["install", "--check", "--tool", "cursor", "--dest", str(tmp_path)]) == 0


def test_check_fails_on_missing(tmp_path: Path) -> None:
    assert main(["install", "--check", "--tool", "cursor", "--dest", str(tmp_path)]) == 1


def test_uninstall_runs(tmp_path: Path) -> None:
    assert main(["install", "--tool", "all", "--dest", str(tmp_path)]) == 0
    assert main(["uninstall", "--tool", "all", "--dest", str(tmp_path)]) == 0
    assert list((tmp_path / ".cursor" / "skills").glob("*/SKILL.md")) == []
    assert list((tmp_path / ".claude" / "skills").glob("*/SKILL.md")) == []
