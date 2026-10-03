# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Keep uv source overrides only for declared dependencies.

Pass project directories explicitly. The default is a read-only report; use
``--write`` to prune or ``--check`` to fail when redundant entries remain.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


def declared_dependencies(config: dict[str, Any]) -> set[str]:
    project = config["project"]
    requirements = list(project.get("dependencies", []))
    for extra in project.get("optional-dependencies", {}).values():
        requirements.extend(extra)
    groups = config.get("dependency-groups", {})

    def visit_group(name: str, visiting: set[str]) -> None:
        if name in visiting:
            raise ValueError(f"Dependency group cycle: {name}")
        if name not in groups:
            raise ValueError(f"Unknown dependency group: {name}")
        for entry in groups[name]:
            if isinstance(entry, str):
                requirements.append(entry)
            elif isinstance(entry, dict) and set(entry) == {"include-group"}:
                visit_group(entry["include-group"], visiting | {name})
            else:
                raise ValueError(f"Unsupported dependency-group entry: {entry!r}")

    for group in groups:
        visit_group(group, set())
    # uv's old development dependency spelling can occur in existing consumers.
    requirements.extend(config.get("tool", {}).get("uv", {}).get("dev-dependencies", []))
    names: set[str] = {canonicalize_name(Requirement(requirement).name)
                       for requirement in requirements}
    names.discard(canonicalize_name(project["name"]))
    return names


def source_span(text: str) -> tuple[int, int]:
    header = re.search(r"(?m)^\[tool\.uv\.sources\][ \t]*(?:#.*)?$", text)
    if not header:
        raise ValueError("Expected [tool.uv.sources] table")
    next_header = re.search(r"(?m)^\[", text[header.end():])
    stop = header.end() + next_header.start() if next_header else len(text)
    return header.end(), stop


def prune(path: Path) -> tuple[str, int, int]:
    """Return pruned text and counts; validate before returning any modification."""
    original = path.read_text()
    config = tomllib.loads(original)
    dependencies = declared_dependencies(config)
    sources = config.get("tool", {}).get("uv", {}).get("sources", {})
    by_name = {canonicalize_name(name): name for name in sources}
    if len(by_name) != len(sources):
        raise ValueError(f"{path}: duplicate normalized source names")
    missing = {name for name in dependencies if name.startswith("omnibias-")} - set(by_name)
    if missing:
        raise ValueError(f"{path}: missing local overrides for {sorted(missing)}")
    retained = {name: value for name, value in sources.items()
                if canonicalize_name(name) in dependencies}
    for name, source in retained.items():
        if canonicalize_name(name) not in dependencies:
            continue
        for variant in source if isinstance(source, list) else [source]:
            if not isinstance(variant, dict) or "path" not in variant:
                continue  # Remote sources retain their original resolution semantics.
            target = (path.parent / variant["path"] / "pyproject.toml").resolve()
            if not target.is_file():
                raise ValueError(f"{path}: {name} points to a missing project")
            actual = tomllib.loads(target.read_text())["project"]["name"]
            if canonicalize_name(actual) != canonicalize_name(name):
                raise ValueError(f"{path}: {name} points to {actual}")
    if retained == sources:
        return original, len(sources), len(retained)
    start, stop = source_span(original)
    body = original[start:stop]
    # Remove a complete assignment, including a multiline inline table or array.
    # The parser validates the result against the original semantic mapping below.
    entries = [match for match in re.finditer(
        r'(?m)^([A-Za-z0-9_.-]+|"[^"\n]+"|\'[^\'\n]+\')[ \t]*=', body
    ) if match[1].strip("\"'") in sources]
    removals = []
    for index, match in enumerate(entries):
        key = match[1].strip("\"'")
        if key in sources and key not in retained:
            end = entries[index + 1].start() if index + 1 < len(entries) else len(body)
            removals.append((match.start(), end))
    for begin, end in reversed(removals):
        body = body[:begin] + body[end:]
    updated = original[:start] + body + original[stop:]
    parsed = tomllib.loads(updated)
    expected = dict(config)
    expected["tool"] = dict(config["tool"])
    expected["tool"]["uv"] = dict(config["tool"]["uv"])
    expected["tool"]["uv"]["sources"] = retained
    if parsed != expected:
        raise ValueError(f"{path}: unsupported source formatting; refusing to change other metadata")
    return updated, len(sources), len(retained)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("projects", nargs="+", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    changes = []
    # Validate the complete selection before writing any file.
    for project in args.projects:
        path = project / "pyproject.toml" if project.is_dir() else project
        updated, before, after = prune(path)
        changes.append((path, updated, before, after))
    for path, updated, before, after in changes:
        print(f"{path.parent.name}: {before} -> {after} source overrides")
        if args.write and before != after:
            path.write_text(updated)
    return int(args.check and any(before != after for _, _, before, after in changes))


if __name__ == "__main__":
    raise SystemExit(main())
