# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Guards that keep the whole test tree collectable in a single pytest run.

The suite is spread over 40+ package test directories. Two layout mistakes can
make it impossible to collect them together, and both have bitten this repo:

1. Two test modules sharing a basename. Under the default ``prepend`` import
   mode pytest derives a module name from the basename alone, so the second one
   collected dies with ``import file mismatch``. The root config sets
   ``--import-mode=importlib`` (which keys on the full path) so this is no
   longer fatal -- these tests simply keep the escape hatch honest.
2. Two shared ``_``-prefixed helpers sharing a basename. Those *are* imported by
   plain module name via the ``pythonpath`` entries, so a duplicate would let
   one package's helper silently shadow another's -- a far nastier failure than
   a collection error, because the tests still run, just against the wrong
   helper.
3. A ``pythonpath`` entry containing a subdirectory named after a real
   third-party package. Because the entry goes on ``sys.path``, such a
   subdirectory becomes an implicit *namespace package* that answers
   ``import torch`` whenever the genuine package is absent. The symptom is
   nasty: ``pytest.importorskip("torch")`` succeeds against an empty module and
   the test fails with ``AttributeError`` instead of skipping, so a machine
   without the optional backend reports broken tests rather than skipped ones.
   (This happened: ``packages/omnibias-fields/tests`` is registered and once held
   empty ``torch/`` and ``jax/`` directories.)

Guarding (2) and (3) is the load-bearing half.
"""

from __future__ import annotations

import collections
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[1]

#: Top-level import names that must never be shadowed by a test subdirectory on
#: ``pythonpath``. These are the optional backends whose absence must produce a
#: clean skip, plus the scientific stack the suite compares against.
SHADOWABLE_IMPORTS = frozenset(
    {
        "flax",
        "jax",
        "jaxlib",
        "keras",
        "mpmath",
        "numpy",
        "optax",
        "scipy",
        "sympy",
        "tensorflow",
        "torch",
    }
)


def _pytest_config() -> dict[str, object]:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        data = tomllib.load(handle)
    config = data["tool"]["pytest"]["ini_options"]
    assert isinstance(config, dict)
    return config


def _helper_modules() -> list[Path]:
    """Shared ``_``-prefixed helpers living in any registered test directory."""
    found: list[Path] = []
    for entry in _pytest_config()["pythonpath"]:  # type: ignore[union-attr]
        directory = ROOT / str(entry)
        if not directory.is_dir():
            continue
        found.extend(
            path
            for path in directory.glob("_*.py")
            if not path.name.startswith("__")
        )
    return found


def test_import_mode_is_importlib() -> None:
    """Losing this makes the 39 duplicate test basenames fatal again."""
    addopts = str(_pytest_config()["addopts"])
    assert "--import-mode=importlib" in addopts


def test_markers_are_strict() -> None:
    """A typo'd marker must fail collection, not silently skip the filter."""
    assert "--strict-markers" in str(_pytest_config()["addopts"])


def test_registered_helper_directories_exist() -> None:
    for entry in _pytest_config()["pythonpath"]:  # type: ignore[union-attr]
        assert (ROOT / str(entry)).is_dir(), f"stale pythonpath entry: {entry}"


def test_registered_directories_do_not_shadow_real_packages() -> None:
    """A registered directory must not expose a third-party name on sys.path.

    Such a subdirectory answers ``import torch`` as an empty namespace package
    whenever real PyTorch is absent, turning what should be a clean skip into a
    confusing ``AttributeError``. Rename the test directory (collection keys on
    the full path under importlib mode, so the name is free) or move the shared
    helper so the parent no longer needs registering.
    """
    offenders: dict[str, list[str]] = {}
    for entry in _pytest_config()["pythonpath"]:  # type: ignore[union-attr]
        directory = ROOT / str(entry)
        if not directory.is_dir():
            continue
        clashes = sorted(
            child.name
            for child in directory.iterdir()
            if child.is_dir() and child.name in SHADOWABLE_IMPORTS
        )
        if clashes:
            offenders[str(entry)] = clashes
    assert not offenders, (
        "these registered pythonpath directories contain subdirectories that "
        f"shadow real top-level packages: {offenders}"
    )


def test_shared_test_helpers_have_unique_basenames() -> None:
    """A duplicate helper name would shadow silently and run the wrong code."""
    by_name: dict[str, list[Path]] = collections.defaultdict(list)
    for path in _helper_modules():
        by_name[path.name].append(path)
    duplicates = {name: paths for name, paths in by_name.items() if len(paths) > 1}
    assert not duplicates, (
        "shared test helpers must have globally unique basenames because they are "
        f"imported by bare module name: {duplicates}"
    )


def test_every_shared_helper_import_resolves() -> None:
    """Every bare ``_helper`` import must resolve from some registered directory.

    A shared helper is deliberately *not* co-located with all of its importers --
    ``tests/_enclosure.py`` is used from four other packages' test trees -- so what
    matters is that the helper's module name is findable on the ``pythonpath``,
    not that the importing file happens to sit next to it.
    """
    registered = [
        (ROOT / str(entry)).resolve()
        for entry in _pytest_config()["pythonpath"]  # type: ignore[union-attr]
    ]
    importers = list((ROOT / "packages").glob("*/tests/**/test_*.py"))
    importers.extend((ROOT / "tests").glob("test_*.py"))
    unresolved: dict[str, str] = {}
    for path in importers:
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped.startswith(("from _", "import _")):
                continue
            if "__future__" in stripped:
                continue
            module = stripped.split()[1].split(".")[0]
            # Resolvable either from a registered directory or beside the importer
            # (pytest's importlib mode still lets a test's own package resolve).
            searched = [*registered, path.parent.resolve()]
            if not any((directory / f"{module}.py").is_file() for directory in searched):
                unresolved[str(path.relative_to(ROOT))] = module
    assert not unresolved, (
        "these tests import a bare `_helper` module that no registered directory "
        "provides; add its directory to [tool.pytest.ini_options].pythonpath: "
        f"{unresolved}"
    )
