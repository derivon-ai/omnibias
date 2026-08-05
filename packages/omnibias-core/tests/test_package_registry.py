# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Filesystem guard for package-registry consistency (the earn-existence rule).

This test is backend-free and runs in the core CI job. It codifies the "package
earns independent existence" hygiene rule (see AGENTS.md) by pinning the one
mechanical invariant that a fold / new-package must not break: **every
``packages/omnibias-*`` distribution is accounted for exactly once** in the root
``[tool.uv.workspace]`` config -- it is a stable ``member``, or an ``exclude``d
extension, or the single documented exception (``omnibias-keras``, which ships
its own pyproject and installs per-backend).

Concretely it catches the two drift modes the consolidation work is prone to:

* a **dangling exclude** -- a ``packages/omnibias-<pkg>`` line left in ``exclude``
  after the directory was folded away (e.g. the ``pde`` / ``gauge`` / ``flow``
  folds), and
* an **unwired package** -- a new ``packages/omnibias-*`` directory added without
  being registered in the workspace at all.

It deliberately does *not* try to police CI-job / docs presence (those vary in
shape per package and are covered by ``mkdocs build --strict`` and the CI matrix);
it guards the single source of truth that every other wiring file keys off.

It additionally pins the **interpreter-support metadata**, which drifted badly
once (most packages declared ``requires-python = ">=3.10"`` while classifying
only ``3.10``, understating support to anyone reading PyPI). Every package now
declares the same floor and the same canonical classifier block, so adding an
interpreter is a single deliberate edit here plus the CI matrix.

Parsing is done with regexes rather than ``tomllib`` on purpose: this guard must
run on the declared floor, Python 3.10, where ``tomllib`` does not exist.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
ROOT_PYPROJECT = REPO_ROOT / "pyproject.toml"

# The one package directory that is intentionally neither a workspace member nor
# an ``exclude`` entry: Keras 3 ships its own pyproject and is installed per
# backend (KERAS_BACKEND=...), so it is not part of the uv workspace at all.
KNOWN_NON_WORKSPACE = {"omnibias-keras"}

# Distributions that were folded into a home package and must stay gone: their
# directory must not reappear and no wiring line may reference them.
FOLDED_AWAY = {
    "omnibias-pde": "omnibias-pinn (omnibias.pinn.solver)",
    "omnibias-gauge": "omnibias-geometry (omnibias.geometry.gauge)",
    "omnibias-flow": "omnibias-score (omnibias.score.flow)",
}


# The single shared interpreter floor, and the classifier block it implies. Bump
# these together with the CI matrix in .github/workflows/ci.yml; nothing else in
# the tree should name a Python version.
REQUIRES_PYTHON = '>=3.10'
SUPPORTED_PYTHONS = ("3.10", "3.11", "3.12", "3.13", "3.14")
CANONICAL_PYTHON_CLASSIFIERS = (
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3 :: Only",
    *(f"Programming Language :: Python :: {v}" for v in SUPPORTED_PYTHONS),
)


def _package_dirs() -> set[str]:
    return {
        p.name
        for p in (REPO_ROOT / "packages").glob("omnibias-*")
        if (p / "pyproject.toml").is_file()
    }


def _package_pyprojects() -> list[Path]:
    return sorted((REPO_ROOT / "packages").glob("omnibias-*/pyproject.toml"))


def _workspace_members_and_excludes() -> tuple[set[str], set[str]]:
    text = ROOT_PYPROJECT.read_text()
    # The workspace table is everything before the next top-level [tool.uv] table.
    workspace = text.split("[tool.uv]", 1)[0]
    before_exclude, _, after_exclude = workspace.partition("exclude")
    members = set(re.findall(r"packages/(omnibias-[\w-]+)", before_exclude))
    excludes = set(re.findall(r"packages/(omnibias-[\w-]+)", after_exclude))
    return members, excludes


def test_no_dangling_workspace_excludes() -> None:
    """Every ``exclude`` entry points at a real package directory."""
    dirs = _package_dirs()
    _, excludes = _workspace_members_and_excludes()
    dangling = sorted(excludes - dirs)
    assert not dangling, (
        "root pyproject [tool.uv.workspace].exclude references packages that do "
        f"not exist (fold left a dangling entry?): {dangling}"
    )


def test_every_package_is_registered_exactly_once() -> None:
    """Every package dir is a member, an exclude, or the documented exception."""
    dirs = _package_dirs()
    members, excludes = _workspace_members_and_excludes()

    overlap = sorted(members & excludes)
    assert not overlap, f"packages listed as BOTH member and exclude: {overlap}"

    unregistered = sorted(dirs - members - excludes - KNOWN_NON_WORKSPACE)
    assert not unregistered, (
        "package directories not registered in [tool.uv.workspace] (add them to "
        f"`exclude`, or promote to `members`): {unregistered}"
    )


def test_folded_packages_stay_folded() -> None:
    """The pde / gauge / flow distributions must not reappear anywhere."""
    dirs = _package_dirs()
    members, excludes = _workspace_members_and_excludes()
    registered = members | excludes
    for pkg, home in sorted(FOLDED_AWAY.items()):
        assert pkg not in dirs, f"{pkg} was folded into {home}; its directory must not return"
        assert pkg not in registered, (
            f"{pkg} was folded into {home}; remove its [tool.uv.workspace] entry"
        )


def test_every_package_declares_the_shared_python_floor() -> None:
    """One floor for the whole tree: a package may not quietly raise or drop it."""
    wrong: dict[str, str] = {}
    for pyproject in _package_pyprojects():
        found = re.findall(r'^requires-python\s*=\s*"([^"]+)"', pyproject.read_text(), re.M)
        if found != [REQUIRES_PYTHON]:
            wrong[pyproject.parent.name] = str(found)
    assert not wrong, (
        f'every package must declare requires-python = "{REQUIRES_PYTHON}"; '
        f"offenders: {wrong}"
    )


def test_python_classifiers_match_the_declared_floor() -> None:
    """Classifiers must advertise exactly the interpreters the floor implies.

    A package declaring ``>=3.10`` but classifying only ``3.10`` understates its
    support on PyPI; one classifying an interpreter the CI matrix does not test
    overstates it. Both are metadata bugs, so pin the block exactly.
    """
    offenders: dict[str, list[str]] = {}
    for pyproject in _package_pyprojects():
        found = re.findall(
            r'^\s*"(Programming Language :: Python[^"]*)",\s*$', pyproject.read_text(), re.M
        )
        if tuple(found) != CANONICAL_PYTHON_CLASSIFIERS:
            offenders[pyproject.parent.name] = found
    assert not offenders, (
        "Python classifiers drifted from the canonical block "
        f"{list(CANONICAL_PYTHON_CLASSIFIERS)}; offenders: {offenders}"
    )


def test_ci_matrix_covers_every_supported_python() -> None:
    """The advertised interpreters and the tested interpreters must not diverge."""
    ci = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text()
    tested = set(re.findall(r"['\"](3\.\d+)['\"]", ci))
    missing = sorted(set(SUPPORTED_PYTHONS) - tested)
    assert not missing, (
        "these interpreters are advertised in the package classifiers but never "
        f"appear in the CI matrix: {missing}"
    )
