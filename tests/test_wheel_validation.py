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
