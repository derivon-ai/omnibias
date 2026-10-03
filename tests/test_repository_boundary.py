# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Keep primitive dependencies closed and repository tooling self-contained."""
from __future__ import annotations

import ast
import re
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[1]


def _projects() -> dict[str, dict]:
    return {
        path.parent.name: tomllib.loads(path.read_text())
        for path in (ROOT / "packages").glob("*/pyproject.toml")
    }


def test_workspace_contains_every_shipped_distribution() -> None:
    projects = _projects()
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]
    assert set(config["uv"]["workspace"]["members"]) == {
        f"packages/{name}" for name in projects
    }
    assert set(config["omnibias"]["license_tiers"]) == set(projects)
    assert set(config["uv"]["sources"]) == set(projects)


def test_distribution_dependencies_stay_inside_primitive_boundary() -> None:
    projects = _projects()
    for name, data in projects.items():
        project = data["project"]
        dependencies = list(project.get("dependencies", []))
        for extra in project.get("optional-dependencies", {}).values():
            dependencies.extend(extra)
        for requirement in dependencies:
            dependency = re.split(r"[\[<>=!~; ]", requirement)[0]
            if dependency.startswith("omnibias-"):
                assert dependency in projects, (name, requirement)


def test_primitive_source_does_not_import_external_consumers() -> None:
    modules = {name.removeprefix("omnibias-") for name in _projects()}
    failures = []
    for path in (ROOT / "packages").glob("*/src/**/*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            imports = []
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imports.append(node.module)
            elif isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            for module in imports:
                if module.startswith("omnibias.") and module.split(".")[1] not in modules:
                    failures.append((str(path.relative_to(ROOT)), node.lineno, module))
    assert not failures, failures


def test_context_has_one_canonical_guide() -> None:
    assert "AGENTS.md" in (ROOT / "CLAUDE.md").read_text()
    assert "AGENTS.md" in (ROOT / ".cursor/rules/omnibias.md").read_text()
    assert not list((ROOT / ".claude/skills").glob("*/SKILL.md"))
