# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Generate package facts and API indexes without importing numerical backends.

Run with ``--check`` in CI. Authored guidance outside marked blocks is preserved.
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_MARKER = "PACKAGE INVENTORY"
LICENSE_MARKER = "LICENSE INVENTORY"
API_MARKER = "API INVENTORY"


def bound_names(statements: list[ast.stmt]) -> set[str]:
    """Collect module bindings, including conditional imports, without executing them."""
    names: set[str] = set()
    for node in statements:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update(alias.asname or alias.name.split(".")[0] for alias in node.names)
        else:
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    names.update(part.id for part in ast.walk(target) if isinstance(part, ast.Name))
            for _field, value in ast.iter_fields(node):
                children = value if isinstance(value, list) else [value]
                for child in children:
                    if isinstance(child, ast.stmt):
                        names.update(bound_names([child]))
                    elif isinstance(child, ast.ExceptHandler):
                        names.update(bound_names(child.body))
    return names


def exported_names(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    assignments = [
        node.value
        for node in tree.body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        )
    ]
    if len(assignments) != 1 or assignments[0] is None:
        raise ValueError(f"{path}: expected one explicit __all__ assignment")
    try:
        exports = ast.literal_eval(assignments[0])
    except (ValueError, TypeError) as error:
        raise ValueError(f"{path}: __all__ must be a literal list or tuple") from error
    if not isinstance(exports, (list, tuple)) or not all(isinstance(name, str) for name in exports):
        raise ValueError(f"{path}: __all__ must contain names")
    if len(exports) != len(set(exports)):
        raise ValueError(f"{path}: duplicate __all__ names")
    missing = set(exports) - bound_names(tree.body)
    if missing:
        raise ValueError(f"{path}: __all__ names have no module binding: {sorted(missing)}")
    return sorted(name for name in exports if not name.startswith("_"))


def packages(root: Path) -> list[dict[str, Any]]:
    config = tomllib.loads((root / "pyproject.toml").read_text())
    result: list[dict[str, Any]] = []
    for member in sorted(config["tool"]["uv"]["workspace"]["members"]):
        directory = root / member
        project = tomllib.loads((directory / "pyproject.toml").read_text())["project"]
        name = project["name"]
        if name != directory.name or not name.startswith("omnibias-"):
            raise ValueError(f"{member}: distribution name does not match directory")
        stem = name.removeprefix("omnibias-")
        source = directory / "src" / "omnibias" / stem
        status = [entry.split(" :: ")[-1] for entry in project.get("classifiers", [])
                  if entry.startswith("Development Status :: ")]
        if len(status) != 1:
            raise ValueError(f"{name}: expected one development-status classifier")
        modules = []
        for path in sorted(source.rglob("*.py")):
            relative = path.relative_to(source)
            if any(part.startswith("_") for part in relative.parts[:-1]):
                continue
            if path.stem.startswith("_") and path.stem != "__init__":
                continue
            parts = list(relative.with_suffix("").parts)
            if parts[-1] == "__init__":
                parts.pop()
            modules.append((".".join(["omnibias", stem, *parts]), path.relative_to(root)))
        result.append({**project, "stem": stem, "maturity": status[0],
                       "exports": exported_names(source / "__init__.py"), "modules": modules})
    return result


def escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def license_label(expression: str) -> str:
    if expression == "AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial":
        return "AGPL-3.0-or-later **or commercial**"
    return expression


def package_table(items: list[dict[str, Any]], *, compact: bool = False) -> str:
    columns = ["Distribution", "Version", "License"] if compact else [
        "Distribution", "Version", "Python", "Maturity", "License", "Responsibility"
    ]
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for item in items:
        target = f"packages/{item['name']}/" if compact else f"api/{item['stem']}.md"
        row = [f"[{item['name']}]({target})", item["version"]]
        if not compact:
            row.extend([item["requires-python"], item["maturity"]])
        row.append(license_label(item["license"]))
        if not compact:
            row.append(item["description"])
        rows.append("| " + " | ".join(escape(cell) for cell in row) + " |")
    return "\n".join(rows)


def license_table(items: list[dict[str, Any]]) -> str:
    groups: dict[str, list[str]] = {}
    for item in items:
        groups.setdefault(item["license"], []).append(
            f"[{item['stem']}](packages/{item['name']}/LICENSE)"
        )
    return "\n".join([
        "| License choice | Distributions |", "| --- | --- |",
        *(f"| {license_label(expression)} | {', '.join(names)} |"
          for expression, names in sorted(groups.items())),
    ])


def api_index(item: dict[str, Any]) -> str:
    rows = [
        f"Version **{item['version']}** · Python **{item['requires-python']}** · "
        f"**{item['maturity']}** · {license_label(item['license'])}",
        "", '<details markdown="1">', "<summary>Public modules and top-level exports</summary>", "",
    ]
    repository = item["urls"]["Source"].rstrip("/")
    namespace = f"omnibias.{item['stem']}"
    source = f"packages/{item['name']}/src/omnibias/{item['stem']}"
    rows.extend([
        f"[Source]({repository}/tree/main/{source}). Modules below are relative to "
        f"`{namespace}`; underscored modules are internal.", "",
        ", ".join(f"`{name.removeprefix(namespace + '.')}`"
                  for name, _path in item["modules"] if name != namespace) + ".",
    ])
    if item["exports"]:
        rows.extend(["", f"Exports from `omnibias.{item['stem']}`:", "",
                     ", ".join(f"`{name}`" for name in item["exports"]) + "."])
    rows.extend(["", "</details>"])
    return "\n".join(rows)


def replace_block(text: str, marker: str, content: str) -> str:
    begin = f"<!-- BEGIN GENERATED {marker} -->"
    end = f"<!-- END GENERATED {marker} -->"
    if text.count(begin) != 1 or text.count(end) != 1:
        raise ValueError(f"Expected exactly one ordered pair of {marker} markers")
    start, stop = text.index(begin), text.index(end)
    if stop < start:
        raise ValueError(f"Reversed {marker} markers")
    generated = begin + "\n\n" + content + "\n\n" + end
    return text[:start] + generated + text[stop + len(end):]


def render(root: Path, *, readme: bool = True, licenses: bool = True) -> dict[Path, str]:
    items = packages(root)
    targets = [(root / "docs/packages.md", PACKAGE_MARKER, package_table(items))]
    targets.extend((root / f"docs/api/{item['stem']}.md", API_MARKER, api_index(item))
                   for item in items)
    if readme:
        targets.append((root / "README.md", PACKAGE_MARKER, package_table(items, compact=True)))
    if licenses:
        targets.append((root / "LICENSING.md", LICENSE_MARKER, license_table(items)))
    return {path: replace_block(path.read_text(), marker, content)
            for path, marker, content in targets}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail on stale generated content")
    parser.add_argument("--docs-only", action="store_true", help="only regenerate documentation")
    args = parser.parse_args()
    try:
        rendered = render(ROOT, readme=not args.docs_only, licenses=not args.docs_only)
    except (ValueError, OSError) as error:
        print(error, file=sys.stderr)
        return 1
    stale = [path for path, content in rendered.items() if content != path.read_text()]
    for path in stale:
        if args.check:
            print(f"stale: {path.relative_to(ROOT)}")
        else:
            path.write_text(rendered[path])
            print(f"generated: {path.relative_to(ROOT)}")
    return int(args.check and bool(stale))


if __name__ == "__main__":
    raise SystemExit(main())
