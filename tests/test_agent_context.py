# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""The active maintenance catalog has one owner per responsibility and valid routes."""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('agent_context', ROOT / 'scripts/check_agent_context.py')
assert SPEC and SPEC.loader
context = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(context)


def fixture_repository(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / 'pyproject.toml').write_text(
        '[tool.uv.workspace]\nmembers = ["packages/omnibias-example"]\n'
    )
    package = root / 'packages/omnibias-example'
    package.mkdir(parents=True)
    (package / 'pyproject.toml').write_text('[project]\nname = "omnibias-example"\n')
    (root / 'AGENTS.md').write_text(
        'Read [.cursor/rules/omnibias.mdc](.cursor/rules/omnibias.mdc). '
        'Select a skill in .agents/skills/.\n'
    )
    (root / 'CLAUDE.md').write_text('Read [AGENTS.md](AGENTS.md).\n')
    rule = root / context.RULE
    rule.parent.mkdir(parents=True)
    rule.write_text('---\nalwaysApply: true\n---\n' + context.EVIDENCE + '\n')
    skill = root / context.SKILLS / 'omnibias-example/SKILL.md'
    skill.parent.mkdir(parents=True)
    skill.write_text(
        '---\nname: omnibias-example\ndescription: Maintain the example representation.\n---\n'
        '\nInspect the owning source before changing its representation.\n'
    )
    return skill


def test_real_main_context_is_consistent() -> None:
    errors, report = context.check(ROOT)
    assert errors == []
    assert report['skill_count'] == len(list((ROOT / 'packages').glob('*/pyproject.toml')))


def test_archived_guidance_is_not_discovered_as_active(tmp_path: Path) -> None:
    fixture_repository(tmp_path / 'main')
    archived = tmp_path / 'main/legacy/.cursor/skills/old/SKILL.md'
    archived.parent.mkdir(parents=True)
    archived.write_text('Historical instructions with retired paths.\n')
    assert context.check(tmp_path / 'main')[0] == []


def test_duplicate_discovery_root_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / 'main'
    skill = fixture_repository(root)
    mirror = root / '.claude/skills/omnibias-example/SKILL.md'
    mirror.parent.mkdir(parents=True)
    mirror.write_text(skill.read_text())
    assert any('duplicate discovery root' in error for error in context.check(root)[0])


def test_duplicate_substantive_context_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / 'main'
    skill = fixture_repository(root)
    paragraph = (
        'The exact parameter ordering is part of this operator contract. A change '
        'must preserve its relation to the returned matrix, serialized state and '
        'independently evaluated directional product.\n'
    )
    skill.write_text(skill.read_text() + '\n' + paragraph)
    agents = root / 'AGENTS.md'
    agents.write_text(agents.read_text() + '\n' + paragraph)
    assert any('repeated substantive paragraph' in error for error in context.check(root)[0])


def test_copied_benchmark_context_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / 'main'
    skill = fixture_repository(root)
    skill.write_text(skill.read_text() + context.EVIDENCE + '\n')
    assert any('one owner' in error for error in context.check(root)[0])


def test_missing_reference_and_test_target_are_rejected(tmp_path: Path) -> None:
    root = tmp_path / 'main'
    skill = fixture_repository(root)
    skill.write_text(skill.read_text() + '\n[Reference](missing.md)\n\n'
                     '```bash\nuv run pytest tests/not_present.py -q\n```\n')
    errors = context.check(root)[0]
    assert any('unresolved local link' in error for error in errors)
    assert any('nonexistent test target' in error for error in errors)


def test_consumer_catalog_coverage_follows_metadata(tmp_path: Path) -> None:
    root = tmp_path / 'main'
    fixture_repository(root)
    projects = tmp_path / 'projects'
    consumer = projects / 'omnibias-consumer'
    consumer.mkdir(parents=True)
    (consumer / 'pyproject.toml').write_text('[project]\nname = "omnibias-consumer"\n')
    assert any('skill coverage missing' in error for error in context.check(root, projects)[0])
    skill = consumer / context.SKILLS / 'omnibias-consumer/SKILL.md'
    skill.parent.mkdir(parents=True)
    skill.write_text('---\nname: omnibias-consumer\ndescription: Maintain the consumer protocol.\n---\n')
    assert context.check(root, projects)[0] == []
