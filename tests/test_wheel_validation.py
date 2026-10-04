# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Installed-wheel checks must fail when an editable/source path leaks in."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

SPEC = importlib.util.spec_from_file_location(
    "import_smoke", Path(__file__).resolve().parents[1] / "scripts/import_smoke.py"
)
assert SPEC and SPEC.loader
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)


def test_rejects_source_tree_module(tmp_path: Path) -> None:
    module = ModuleType("omnibias.example")
    module.__file__ = str(tmp_path / "source/example.py")
    with pytest.raises(ValueError, match="Source leakage"):
        smoke.check_module(module, (tmp_path / "site-packages",))


def test_namespace_must_have_only_installed_paths(tmp_path: Path) -> None:
    module = ModuleType("omnibias")
    module.__path__ = [str(tmp_path / "site-packages/omnibias"), str(tmp_path / "src/omnibias")]
    with pytest.raises(ValueError, match="Source leakage"):
        smoke.check_module(module, (tmp_path / "site-packages",))


def test_editable_distribution_rejected() -> None:
    class Editable:
        metadata = {"Name": "omnibias-example"}

        def read_text(self, name: str) -> str:
            return '{"dir_info": {"editable": true}}'

    with pytest.raises(ValueError, match="Editable installation"):
        smoke.check_distribution(Editable(), permissive=False)


def test_rejects_copyleft_in_permissive_install() -> None:
    class Copyleft:
        metadata = {"Name": "omnibias-example", "License-Expression": "AGPL-3.0-or-later"}
        files = ["example.dist-info/METADATA"]

        def read_text(self, name: str) -> None:
            return None

    with pytest.raises(ValueError, match="Non-permissive dependency"):
        smoke.check_distribution(Copyleft(), permissive=True)


def test_overlapping_wheels_fail_even_if_content_matches(tmp_path: Path) -> None:
    import zipfile

    spec = importlib.util.spec_from_file_location(
        "validate_wheels", Path(__file__).resolve().parents[1] / "scripts/validate_wheels.py"
    )
    assert spec and spec.loader
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    for name in ("first", "second"):
        with zipfile.ZipFile(tmp_path / f"{name}.whl", "w") as archive:
            archive.writestr(f"{name}.dist-info/METADATA", f"Name: {name}\nVersion: 0.1\n")
            archive.writestr("omnibias/shared.py", "")
    with pytest.raises(ValueError, match="Overlapping wheel file"):
        validator.wheel_index(tmp_path)


def test_archive_build_excludes_stale_setuptools_modules(tmp_path: Path) -> None:
    """A removed integration bridge must not survive through an old build cache."""
    import os
    import shutil
    import zipfile

    if shutil.which("uv") is None:
        pytest.skip("uv is required for the source-archive build regression")
    spec = importlib.util.spec_from_file_location(
        "validate_wheels", Path(__file__).resolve().parents[1] / "scripts/validate_wheels.py"
    )
    assert spec and spec.loader
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    project = tmp_path / "project"
    package = project / "src/wheel_build_probe"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("VALUE = 7\n")
    (project / "pyproject.toml").write_text(
        '[build-system]\nrequires = ["setuptools>=77"]\n'
        'build-backend = "setuptools.build_meta"\n'
        '[project]\nname = "omnibias-wheel-build-probe"\nversion = "0.0.0"\n'
        'requires-python = ">=3.10"\n'
        '[tool.setuptools.packages.find]\nwhere = ["src"]\n'
    )
    stale = project / "build/lib/wheel_build_probe/deleted_bridge.py"
    stale.parent.mkdir(parents=True)
    stale.write_text("REMOVED = True\n")
    old_manifest = project / "src/omnibias_wheel_build_probe.egg-info/SOURCES.txt"
    old_manifest.parent.mkdir(parents=True)
    old_manifest.write_text("src/wheel_build_probe/deleted_bridge.py\n")
    wheelhouse = tmp_path / "wheels"
    wheelhouse.mkdir()
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"PYTHONPATH", "VIRTUAL_ENV", "PYTHONHOME"}
    }
    validator.build_wheel(project, wheelhouse, log=tmp_path / "build.log", env=env)
    wheels = list(wheelhouse.glob("*.whl"))
    assert len(wheels) == 1
    with zipfile.ZipFile(wheels[0]) as archive:
        assert archive.read("wheel_build_probe/__init__.py") == b"VALUE = 7\n"
        assert not any("deleted_bridge" in name for name in archive.namelist())
    # Validation did not need to delete the user's existing build directory.
    assert stale.read_text() == "REMOVED = True\n"


def test_artifact_metadata_cannot_masquerade_as_current_source(tmp_path: Path) -> None:
    import zipfile

    spec = importlib.util.spec_from_file_location(
        "validate_wheels", Path(__file__).resolve().parents[1] / "scripts/validate_wheels.py"
    )
    assert spec and spec.loader
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    wheel = tmp_path / "old.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            "omnibias_probe.dist-info/METADATA",
            "Name: omnibias-probe\nVersion: 0.1\nLicense-Expression: Apache-2.0\n\nWheel example",
        )
    expected = {"name": "omnibias-probe", "version": "0.2", "license": "Apache-2.0"}
    with pytest.raises(ValueError, match="Version differs"):
        validator.checked_wheel_metadata(wheel, expected)
    expected["version"] = "0.1"
    expected["license"] = "AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial"
    with pytest.raises(ValueError, match="License-Expression differs"):
        validator.checked_wheel_metadata(wheel, expected)
    expected["license"] = "Apache-2.0"
    assert validator.checked_wheel_metadata(wheel, expected).get_payload() == "Wheel example"


def test_wheel_readme_preserves_unicode_without_email_charset() -> None:
    from email.parser import BytesParser

    spec = importlib.util.spec_from_file_location(
        'validate_wheels', Path(__file__).resolve().parents[1] / 'scripts/validate_wheels.py'
    )
    assert spec and spec.loader
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    source = '# A derivative σ⁽ⁿ⁾\n\n```python\nα = 2\nassert α ** 2 == 4\n```\n'
    metadata = BytesParser().parsebytes(
        ('Name: omnibias-probe\nDescription-Content-Type: text/markdown\n\n' + source).encode('utf-8')
    )
    assert validator.wheel_readme(metadata) == source


def test_ecosystem_readme_command_runs_primitives_and_consumers(tmp_path, monkeypatch) -> None:
    import sys

    spec = importlib.util.spec_from_file_location(
        'validate_wheels', Path(__file__).resolve().parents[1] / 'scripts/validate_wheels.py'
    )
    assert spec and spec.loader
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    primitive, consumer = tmp_path / 'primitive', tmp_path / 'consumer'
    wheels = {p.name: (tmp_path / (p.name + '.whl')) for p in [primitive, consumer]}
    seen = []
    monkeypatch.setattr(validator, 'discover', lambda _: ([primitive], [consumer]))
    monkeypatch.setattr(validator, 'project_data', lambda p: {'name': p.name})
    monkeypatch.setattr(validator, 'wheel_index', lambda _: wheels)

    def validate(project, *args, **kwargs):
        assert kwargs['readme']
        seen.append(project.name)
        return {'name': project.name, 'status': 'passed'}

    monkeypatch.setattr(validator, 'validate', validate)
    monkeypatch.setattr(sys, 'argv', ['validate_wheels.py', '--readme', '--artifact-only', '--projects-root', str(tmp_path), '--output', str(tmp_path / 'output')])
    assert validator.main() == 0
    assert seen == ['primitive', 'consumer']
