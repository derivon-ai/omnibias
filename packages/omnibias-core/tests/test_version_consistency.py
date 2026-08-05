# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Release guard: version single-source-of-truth.

``pyproject.toml`` metadata is the *only* place a distribution's version is
declared. Each package's top-level ``__init__`` derives ``__version__`` from the
installed metadata (``importlib.metadata.version``) with a bare-source-checkout
fallback, so the code can never drift from the packaging metadata. This
backend-free test pins that contract:

* every ``packages/omnibias-*`` top-level ``__init__`` derives its version and
  hardcodes no version literal at module scope;
* no *sub*-module ``__init__`` re-declares ``__version__`` (a distribution has
  exactly one version, carried by its top-level package);
* every ``pyproject.toml`` version is a valid PEP 440 string; and
* the README citation *and* ``CITATION.cff`` versions match ``omnibias-core``.

It is deliberately *static* (reads the source tree, not installed metadata) so
it is independent of whether the editable install has been re-synced.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - the toolchain is >=3.11
    import tomli as tomllib

REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGES = REPO_ROOT / "packages"

_TOP_LEVEL_LITERAL = re.compile(r'^__version__\s*=\s*["\']', re.MULTILINE)
_ANY_VERSION_ASSIGN = re.compile(r'^__version__\s*=', re.MULTILINE)
_PEP440 = re.compile(
    r"^\d+(\.\d+)*"  # release segment
    r"(a\d+|b\d+|rc\d+)?"  # optional pre-release
    r"(\.post\d+)?(\.dev\d+)?$"  # optional post / dev
)


def _package_pyprojects() -> list[Path]:
    return sorted(PACKAGES.glob("omnibias-*/pyproject.toml"))


def _pyproject_version(pyproject: Path) -> str:
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    return str(data["project"]["version"])


def _all_init_files() -> list[Path]:
    return sorted(PACKAGES.glob("omnibias-*/src/omnibias/**/__init__.py"))


def _is_top_level(init: Path) -> bool:
    # ``.../src/omnibias/<pkg>/__init__.py`` -> parent.parent is the ``omnibias``
    # namespace dir; deeper submodules have a package name there instead.
    return init.parent.parent.name == "omnibias"


def test_top_level_init_derives_version() -> None:
    offenders: list[str] = []
    for init in _all_init_files():
        if not _is_top_level(init):
            continue
        text = init.read_text(encoding="utf-8")
        rel = str(init.relative_to(REPO_ROOT))
        if "_pkg_version(" not in text:
            offenders.append(f"{rel}: does not derive __version__ from metadata")
        if _TOP_LEVEL_LITERAL.search(text):
            offenders.append(f"{rel}: hardcodes a module-scope __version__ literal")
    assert not offenders, "version-derivation contract violated:\n" + "\n".join(offenders)


def test_submodules_declare_no_version() -> None:
    offenders: list[str] = []
    for init in _all_init_files():
        if _is_top_level(init):
            continue
        text = init.read_text(encoding="utf-8")
        if _ANY_VERSION_ASSIGN.search(text):
            offenders.append(str(init.relative_to(REPO_ROOT)))
    assert not offenders, (
        "sub-module __init__ files must not re-declare __version__ (the "
        "distribution version lives only in the top-level package): " + ", ".join(offenders)
    )


def test_pyproject_versions_are_pep440() -> None:
    bad: list[str] = []
    for pyproject in _package_pyprojects():
        version = _pyproject_version(pyproject)
        if not _PEP440.match(version):
            bad.append(f"{pyproject.parent.name}: {version!r}")
    assert not bad, "non-PEP440 pyproject versions: " + ", ".join(bad)


def test_readme_citation_matches_core() -> None:
    core_version = _pyproject_version(PACKAGES / "omnibias-core" / "pyproject.toml")
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    match = re.search(r"version\s*=\s*\{([^}]+)\}", readme)
    assert match is not None, "README citation block has no version field"
    assert match.group(1).strip() == core_version, (
        f"README citation version {match.group(1).strip()!r} != omnibias-core "
        f"pyproject version {core_version!r}"
    )


def test_citation_cff_matches_core() -> None:
    core_version = _pyproject_version(PACKAGES / "omnibias-core" / "pyproject.toml")
    cff = (REPO_ROOT / "CITATION.cff").read_text(encoding="utf-8")
    match = re.search(r'^version:\s*"([^"]+)"', cff, re.MULTILINE)
    assert match is not None, "CITATION.cff has no quoted version field"
    assert match.group(1).strip() == core_version, (
        f"CITATION.cff version {match.group(1).strip()!r} != omnibias-core "
        f"pyproject version {core_version!r}"
    )


def _omnibias_requirements() -> list[tuple[str, str, str]]:
    """``(dependent, requirement string, where)`` for every intra-omnibias edge."""
    out: list[tuple[str, str, str]] = []
    for pyproject in _package_pyprojects():
        project = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]
        specs = [(str(s), "dependencies") for s in project.get("dependencies", [])]
        for extra, extra_specs in (project.get("optional-dependencies") or {}).items():
            specs += [(str(s), f"optional-dependencies.{extra}") for s in extra_specs]
        for spec, where in specs:
            if spec.lstrip().startswith("omnibias-"):
                out.append((str(project["name"]), spec, where))
    return out


def test_intra_package_constraints_admit_the_current_versions() -> None:
    """A sibling pin must be satisfiable by the version actually in the tree.

    This is the failure mode that only shows up at ``pip install`` time, after
    publication: ``omnibias-holonomic`` requiring ``omnibias-qcalculus>=0.1.0``
    while qcalculus is at ``0.1.0a1`` is unsatisfiable under PEP 440 ordering
    (``0.1.0a1 < 0.1.0``), so the dependent becomes uninstallable the moment it
    reaches an index. Pinning against an alpha sibling needs the alpha floor
    (``>=0.1.0a1``), not the release it previews.
    """
    from packaging.requirements import Requirement
    from packaging.version import Version

    versions = {
        str(tomllib.loads(p.read_text(encoding="utf-8"))["project"]["name"]): Version(
            _pyproject_version(p)
        )
        for p in _package_pyprojects()
    }

    edges = _omnibias_requirements()
    assert len(edges) > 40, f"only {len(edges)} intra-omnibias edges found -- parser broke"

    unsatisfiable = [
        f"{dependent} requires {spec!r} ({where}) but {req.name} is {versions[req.name]}"
        for dependent, spec, where in edges
        if (req := Requirement(spec)).name in versions
        and not req.specifier.contains(versions[req.name], prereleases=True)
    ]
    assert not unsatisfiable, (
        "a sibling pin cannot be satisfied by the version in this tree, so the "
        "dependent would be uninstallable once published: " + "; ".join(unsatisfiable)
    )
