# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Generated facts follow metadata and public exports without backend imports."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


inventory = load_script("generate_inventory")
sources = load_script("prune_sources")


def test_generated_docs_match_source_metadata_and_exports() -> None:
    for path, content in inventory.render(ROOT, readme=False, licenses=False).items():
        assert path.read_text() == content, f"Regenerate {path.relative_to(ROOT)}"


def test_package_table_uses_metadata_without_description_truncation() -> None:
    item = inventory.packages(ROOT)[0]
    changed = {**item, "version": "9.1.2", "requires-python": ">=3.14",
               "maturity": "Production/Stable", "description": "a | b\n" + "detail " * 100}
    table = inventory.package_table([changed])
    assert "9.1.2" in table and ">=3.14" in table and "Production/Stable" in table
    assert inventory.escape(changed["description"]) in table


def test_readme_links_open_package_front_pages() -> None:
    for item in inventory.packages(ROOT):
        table = inventory.package_table([item], compact=True)
        assert f"](packages/{item['name']}/)" in table
        assert (ROOT / "packages" / item["name"] / "README.md").is_file()
        assert f"](api/{item['stem']}.md)" in inventory.package_table([item])


def test_marker_replacement_preserves_authored_text() -> None:
    text = "intro\n<!-- BEGIN GENERATED X -->\nold\n<!-- END GENERATED X -->\noutro\n"
    assert inventory.replace_block(text, "X", "new") == (
        "intro\n<!-- BEGIN GENERATED X -->\n\nnew\n\n<!-- END GENERATED X -->\noutro\n"
    )


@pytest.mark.parametrize("text", ["missing", "<!-- END GENERATED X --><!-- BEGIN GENERATED X -->",
                                 "<!-- BEGIN GENERATED X --><!-- BEGIN GENERATED X --><!-- END GENERATED X -->"])
def test_ambiguous_markers_fail(text: str) -> None:
    with pytest.raises(ValueError):
        inventory.replace_block(text, "X", "new")


def test_exports_are_read_without_executing_imports(tmp_path: Path) -> None:
    path = tmp_path / "__init__.py"
    path.write_text("from deliberately_unavailable import public\n"
                    "try:\n    __version__ = '1'\nexcept Exception:\n    __version__ = '0'\n"
                    "__all__ = ['__version__', 'public']\n")
    assert inventory.exported_names(path) == ["public"]


@pytest.mark.parametrize("source", [
    "__all__ = ['missing']",
    "public = 1\n__all__ = ['public', 'public']",
    "public = 1\n__all__ = list(['public'])",
    "def local():\n    hidden = 1\n__all__ = ['hidden']",
])
def test_invalid_exports_fail(tmp_path: Path, source: str) -> None:
    path = tmp_path / "__init__.py"
    path.write_text(source)
    with pytest.raises(ValueError):
        inventory.exported_names(path)


def test_module_inventory_omits_private_modules() -> None:
    for package in inventory.packages(ROOT):
        names = [name for name, _path in package["modules"]]
        assert names == sorted(set(names))
        assert all(not part.startswith("_") for name in names for part in name.split("."))


def make_project(tmp_path: Path) -> Path:
    for name in ("core", "torch", "jax", "unused"):
        project = tmp_path / f"omnibias-{name}"
        project.mkdir()
        (project / "pyproject.toml").write_text(f'[project]\nname = "omnibias-{name}"\n')
    path = tmp_path / "pyproject.toml"
    path.write_text('''[project]
name = "omnibias-consumer"
dependencies = ["omnibias-core>=1", "omnibias-consumer[torch]", "some-other-package>=1", "numpy>=1"]
[project.optional-dependencies]
torch = ["omnibias-torch>=1; python_version >= '3.10'"]
[dependency-groups]
test = ["omnibias-jax>=1"]
dev = [{include-group = "test"}]
[tool.uv.sources]
omnibias-core = {path = "omnibias-core", editable = true}
omnibias-torch = {path = "omnibias-torch", editable = true}
omnibias-jax = {path = "omnibias-jax", editable = true}
omnibias-unused = {path = "omnibias-unused", editable = true}
some-other-package = {git = "https://example.com/other"}
unused-foreign = {git = "https://example.com/unused"}

[tool.pytest.ini_options]
testpaths = ["tests"]
''')
    return path


def test_sources_retain_extras_groups_and_unrelated_configuration(tmp_path: Path) -> None:
    path = make_project(tmp_path)
    result, before, after = sources.prune(path)
    assert (before, after) == (6, 4)
    assert "omnibias-unused =" not in result
    assert "unused-foreign =" not in result
    assert "omnibias-torch =" in result and "omnibias-jax =" in result
    assert 'some-other-package = {git = "https://example.com/other"}' in result
    assert '[tool.pytest.ini_options]\ntestpaths = ["tests"]' in result
    assert "omnibias-unused =" in path.read_text()  # read-only until CLI --write
    path.write_text(result)
    assert sources.prune(path) == (result, 4, 4)


def test_sources_validate_target_project_name(tmp_path: Path) -> None:
    path = make_project(tmp_path)
    target = tmp_path / "omnibias-core/pyproject.toml"
    target.write_text('[project]\nname = "unrelated-project"\n')
    with pytest.raises(ValueError, match="points to unrelated-project"):
        sources.prune(path)


def test_sources_remove_complete_multiline_values(tmp_path: Path) -> None:
    path = make_project(tmp_path)
    path.write_text(path.read_text().replace(
        'omnibias-unused = {path = "omnibias-unused", editable = true}',
        'omnibias-unused = [\n{path = "omnibias-unused", editable = true},\n]'))
    result, before, after = sources.prune(path)
    assert (before, after) == (6, 4)
    assert "omnibias-unused =" not in result


def test_sources_drop_non_omnibias_self_override(tmp_path: Path) -> None:
    path = make_project(tmp_path)
    path.write_text(path.read_text().replace("omnibias-consumer", "consumer-app").replace(
        "[tool.uv.sources]\n", '[tool.uv.sources]\nconsumer-app = {path = ".", editable = true}\n'))
    result, before, after = sources.prune(path)
    assert (before, after) == (7, 4)
    assert "consumer-app =" not in result
    assert 'some-other-package = {git = "https://example.com/other"}' in result
    assert '"consumer-app[torch]"' in result


def test_sources_missing_override_fails(tmp_path: Path) -> None:
    path = make_project(tmp_path)
    path.write_text(path.read_text().replace(
        'omnibias-core = {path = "omnibias-core", editable = true}\n', ""))
    with pytest.raises(ValueError, match="missing local overrides"):
        sources.prune(path)


def test_dependency_group_cycles_fail() -> None:
    with pytest.raises(ValueError, match="cycle"):
        sources.declared_dependencies({"project": {"name": "example"},
                                       "dependency-groups": {"dev": [{"include-group": "dev"}]}})
