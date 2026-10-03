# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Validate single-owner agent context without loading historical catalogs."""
from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[import-not-found]

ROOT = Path(__file__).resolve().parents[1]
RULE = Path('.cursor/rules/omnibias.mdc')
SKILLS = Path('.agents/skills')
EVIDENCE = '<!-- BEGIN GENERATED CAPABILITY EVIDENCE -->'
ANCILLARY = ('omnibias_experiments', 'omnibias_papers', 'omnibias_web')


def project_name(path: Path) -> str:
    return str(tomllib.loads(path.read_text())['project']['name'])


def context_files(root: Path) -> list[Path]:
    """Only active entrypoints, rules and skills; never recurse into archives."""
    files = [root / name for name in ('AGENTS.md', 'CLAUDE.md') if (root / name).is_file()]
    rules = root / '.cursor/rules'
    if rules.is_dir():
        files.extend(p for p in rules.iterdir() if p.is_file() and p.suffix in {'.md', '.mdc'})
    files.extend(sorted((root / SKILLS).glob('*/SKILL.md')))
    return files


def expected_repositories(root: Path, projects: Path | None) -> dict[Path, set[str]]:
    config = tomllib.loads((root / 'pyproject.toml').read_text())
    result = {
        root: {
            project_name(root / member / 'pyproject.toml')
            for member in config['tool']['uv']['workspace']['members']
        }
    }
    if projects is not None:
        for metadata in sorted(projects.glob('omnibias-*/pyproject.toml')):
            name = project_name(metadata)
            if name.startswith('omnibias-'):
                result[metadata.parent] = {name}
        for name in ANCILLARY:
            directory = projects / name
            if directory.is_dir():
                result[directory] = {name.replace('_', '-')}
    return result


def local_links(path: Path) -> list[Path]:
    result = []
    for target in re.findall(r'\]\(([^)]+)\)', path.read_text()):
        parsed = urlsplit(target)
        if parsed.scheme or parsed.netloc or not parsed.path:
            continue
        result.append((path.parent / unquote(parsed.path)).resolve())
    return result


def substantive_paragraphs(text: str) -> list[str]:
    text = re.sub(r'^---\n.*?\n---\n', '', text, count=1, flags=re.S)
    text = re.sub(r'```.*?```', '', text, flags=re.S)
    return [
        normalized for paragraph in re.split(r'\n\s*\n', text)
        if len(re.sub(r'\[([^]]+)\]\([^)]+\)', r'\1',
                      normalized := ' '.join(paragraph.split()))) >= 160
        and not normalized.startswith(('#', '<!--'))
    ]


def check(root: Path = ROOT, projects: Path | None = None) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    repositories = expected_repositories(root, projects)
    evidence_owners: list[Path] = []
    paragraphs: dict[str, Path] = {}
    descriptions: dict[str, Path] = {}
    repository_reports: list[dict[str, Any]] = []
    for repository, expected in repositories.items():
        skills = sorted((repository / SKILLS).glob('*/SKILL.md'))
        found = {path.parent.name for path in skills}
        if found != expected:
            errors.append(f'{repository.name}: skill coverage missing={sorted(expected-found)}, extra={sorted(found-expected)}')
        for old_root in ('.cursor/skills', '.claude/skills', '.codex/skills'):
            if list((repository / old_root).rglob('SKILL.md')):
                errors.append(f'{repository.name}: duplicate discovery root {old_root}')
        files = context_files(repository)
        for path in files:
            text = path.read_text()
            if EVIDENCE in text:
                evidence_owners.extend([path] * text.count(EVIDENCE))
            for target in local_links(path):
                if not target.exists():
                    errors.append(f'{path}: unresolved local link {target}')
            for paragraph in substantive_paragraphs(text):
                if paragraph in paragraphs and paragraphs[paragraph] != path:
                    errors.append(f'{path}: repeated substantive paragraph from {paragraphs[paragraph]}')
                paragraphs[paragraph] = path
        for path in skills:
            text = path.read_text()
            frontmatter = re.match(r'^---\nname: ([a-z0-9-]+)\ndescription: (.+)\n---\n', text)
            if not frontmatter or frontmatter[1] != path.parent.name:
                errors.append(f'{path}: invalid name/description discovery metadata')
                continue
            description = frontmatter[2]
            if description in descriptions:
                errors.append(f'{path}: duplicate skill description from {descriptions[description]}')
            descriptions[description] = path
            for line in text.splitlines():
                if line.startswith('uv run pytest '):
                    for item in shlex.split(line)[3:]:
                        if item.startswith(('tests/', 'packages/')) and not (repository / item).exists():
                            errors.append(f'{path}: nonexistent test target {item}')
        repository_reports.append({
            'repository': repository.name,
            'skills': len(skills),
            'guidance_bytes': sum(path.stat().st_size for path in files),
            'entrypoint_bytes': sum(path.stat().st_size for path in files if path.name in {'AGENTS.md', 'CLAUDE.md'}),
            'skill_bytes': sum(path.stat().st_size for path in skills),
        })
    rule = root / RULE
    if evidence_owners != [rule]:
        errors.append(f'Capability evidence must have one owner: {RULE}')
    if not rule.is_file() or 'alwaysApply: true' not in rule.read_text():
        errors.append('The canonical capability rule must be a discoverable always-on .mdc file')
    agents = root / 'AGENTS.md'
    if not agents.is_file() or str(RULE) not in agents.read_text() or '.agents/skills/' not in agents.read_text():
        errors.append('AGENTS must route to the canonical rule and on-demand skill catalog')
    claude = root / 'CLAUDE.md'
    if not claude.is_file() or 'AGENTS.md' not in claude.read_text():
        errors.append('CLAUDE must route to the portable entrypoint')
    if (root / '.cursor/rules/omnibias.md').exists():
        errors.append('Retired plain Markdown rule is still present')
    shared = root / 'docs/development/numerical-contracts.md'
    if shared.is_file():
        for target in local_links(shared):
            if not target.exists():
                errors.append(f'{shared}: unresolved shared-reference link {target}')
    report = {
        'repositories': repository_reports,
        'skill_count': sum(item['skills'] for item in repository_reports),
        'always_loaded_main_bytes': sum(
            path.stat().st_size for path in (agents, claude, rule) if path.is_file()
        ),
        'shared_numerical_reference_bytes': shared.stat().st_size if shared.is_file() else 0,
        'errors': errors,
    }
    return errors, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--projects-root', type=Path, help='Also check active sibling repositories')
    parser.add_argument('--report', type=Path, help='Write optional context measurements as JSON')
    args = parser.parse_args()
    errors, report = check(ROOT, args.projects_root.resolve() if args.projects_root else None)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + '\n')
    for error in errors:
        print(error)
    print(f"Agent context: {report['skill_count']} skills, {len(report['repositories'])} repositories, {len(errors)} errors")
    return int(bool(errors))


if __name__ == '__main__':
    raise SystemExit(main())
