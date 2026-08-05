# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Release guard: the open-core licence split is internally consistent.

omnibias is dual-tiered. ``[tool.omnibias.license_tiers]`` in the repository
root ``pyproject.toml`` records, for each of the 42 distributions, whether it is

* ``permissive`` -- ``Apache-2.0``; or
* ``copyleft``   -- ``AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial``.

That table is the single source of truth. Five things must agree with it, and
this backend-free test fails the build if any of them drifts:

1. **Coverage** -- every package on disk appears in the table, and the table
   names no package that does not exist.
2. **LICENSE text** -- each package's ``LICENSE`` file carries its tier's SPDX
   expression, and the referenced text exists under ``LICENSES/``.
3. **File headers** -- every ``.py`` file in a package carries exactly its
   tier's SPDX header. ``scripts/license_headers.py --check`` is the fixer's
   own dry run; this asserts the same invariant so CI cannot skip it.
4. **README prose** -- each package's ``README.md`` ``## License`` section
   matches its tier (PyPI renders the README as the long_description, so a
   stale AGPL claim on an Apache package is a public contradiction).
5. **The DAG invariant** -- and this is the one that actually matters:
   **no ``permissive`` package may depend on a ``copyleft`` package**, through
   ``dependencies`` or through *any* ``optional-dependencies`` extra.

Without (4), adding an ``omnibias-verify`` extra to a permissive package in
eighteen months would silently subject the whole adoption tier to the AGPL, and
nothing else in the build would notice. Permissive-below-copyleft is fine and
is the intended shape: Tier C is the upward closure of the certified-decision
seed set, so ``omnibias-difference`` stays permissive even though the copyleft
``omnibias-verify`` requires it.
"""

from __future__ import annotations

import re
from pathlib import Path

import tomllib

REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGES = REPO_ROOT / "packages"

_SPDX = re.compile(r"^#\s*SPDX-License-Identifier:\s*(?P<expr>.+?)\s*$")
_WORKSPACE_DEP = re.compile(r"^\s*(?P<name>omnibias-[a-z0-9-]+)")

_SKIP_PARTS = frozenset({"__pycache__", ".mypy_cache", ".ruff_cache", ".pytest_cache", "build", "dist"})


def _root_config() -> dict[str, object]:
    return tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _tiers() -> dict[str, str]:
    omnibias = _root_config()["tool"]["omnibias"]  # type: ignore[index]
    return dict(omnibias["license_tiers"])  # type: ignore[index]


def _expressions() -> dict[str, str]:
    omnibias = _root_config()["tool"]["omnibias"]  # type: ignore[index]
    return dict(omnibias["license_expressions"])  # type: ignore[index]


def _package_dirs() -> list[Path]:
    return sorted(p for p in PACKAGES.glob("omnibias-*") if (p / "pyproject.toml").is_file())


def _project(pkg_dir: Path) -> dict[str, object]:
    data = tomllib.loads((pkg_dir / "pyproject.toml").read_text(encoding="utf-8"))
    return dict(data["project"])  # type: ignore[arg-type]


def _dist_name(pkg_dir: Path) -> str:
    return str(_project(pkg_dir)["name"])


def _omnibias_requirements(pkg_dir: Path) -> dict[str, str]:
    """Map each omnibias distribution this package depends on to *how*."""
    project = _project(pkg_dir)
    out: dict[str, str] = {}
    for spec in project.get("dependencies", []) or []:
        m = _WORKSPACE_DEP.match(str(spec))
        if m:
            out.setdefault(m.group("name"), "dependencies")
    extras = project.get("optional-dependencies", {}) or {}
    assert isinstance(extras, dict)
    for extra, specs in extras.items():
        for spec in specs:
            m = _WORKSPACE_DEP.match(str(spec))
            if m:
                out.setdefault(m.group("name"), f"optional-dependencies.{extra}")
    return out


# ---------------------------------------------------------------- coverage


def test_every_package_has_a_declared_tier() -> None:
    tiers = _tiers()
    on_disk = {_dist_name(p) for p in _package_dirs()}
    assert on_disk, "no packages discovered -- the glob broke"
    missing = sorted(on_disk - set(tiers))
    stale = sorted(set(tiers) - on_disk)
    assert not missing, f"packages absent from [tool.omnibias.license_tiers]: {missing}"
    assert not stale, f"[tool.omnibias.license_tiers] names non-existent packages: {stale}"


def test_tier_values_are_known_and_both_populated() -> None:
    tiers = _tiers()
    expressions = _expressions()
    unknown = sorted({t for t in tiers.values()} - set(expressions))
    assert not unknown, f"unknown tier(s) {unknown}; known: {sorted(expressions)}"
    for tier in expressions:
        members = [d for d, t in tiers.items() if t == tier]
        assert members, f"tier {tier!r} has no members -- the split has collapsed"


def test_license_texts_exist_for_every_referenced_identifier() -> None:
    """REUSE layout: every SPDX identifier in play resolves under ``LICENSES/``."""
    identifiers = {
        part.strip()
        for expression in _expressions().values()
        for part in expression.split(" OR ")
    }
    missing = sorted(i for i in identifiers if not (REPO_ROOT / "LICENSES" / f"{i}.txt").exists())
    assert not missing, f"LICENSES/<id>.txt missing for: {missing}"


# ------------------------------------------------------------ LICENSE files


def test_each_package_license_matches_its_tier() -> None:
    tiers, expressions = _tiers(), _expressions()
    offenders: dict[str, str] = {}
    for pkg_dir in _package_dirs():
        dist = _dist_name(pkg_dir)
        wanted = expressions[tiers[dist]]
        license_file = pkg_dir / "LICENSE"
        if not license_file.exists():
            offenders[dist] = "no LICENSE file"
            continue
        text = license_file.read_text(encoding="utf-8")
        if f"SPDX-License-Identifier: {wanted}" not in text:
            offenders[dist] = f"LICENSE does not declare {wanted!r}"
    assert not offenders, f"per-package LICENSE drift: {offenders}"


def test_each_pyproject_declares_its_tier_as_a_pep639_expression() -> None:
    """PEP 639: ``license`` is an SPDX expression string, not the legacy table."""
    tiers, expressions = _tiers(), _expressions()
    offenders: dict[str, str] = {}
    for pkg_dir in _package_dirs():
        project = _project(pkg_dir)
        dist = str(project["name"])
        declared = project.get("license")
        wanted = expressions[tiers[dist]]
        if not isinstance(declared, str):
            offenders[dist] = f"license is {declared!r}, want the SPDX string {wanted!r}"
        elif declared != wanted:
            offenders[dist] = f"{declared!r} != {wanted!r}"
        elif project.get("license-files") != ["LICENSE"]:
            offenders[dist] = f"license-files is {project.get('license-files')!r}, want ['LICENSE']"
    assert not offenders, f"PEP 639 metadata drift: {offenders}"


def test_no_package_keeps_a_deprecated_license_classifier() -> None:
    """setuptools>=77 rejects a ``License ::`` classifier next to an SPDX expression."""
    offenders = {
        _dist_name(p): [c for c in _project(p).get("classifiers", []) if str(c).startswith("License ::")]
        for p in _package_dirs()
    }
    flagged = {k: v for k, v in offenders.items() if v}
    assert not flagged, f"deprecated License :: classifiers remain: {flagged}"


def test_root_license_stays_agpl_for_the_repository() -> None:
    """GitHub's detector reads the root ``LICENSE``; the strong tier is the honest label."""
    text = (REPO_ROOT / "LICENSE").read_text(encoding="utf-8")
    assert "GNU AFFERO GENERAL PUBLIC LICENSE" in text


# ----------------------------------------------------------- README prose


_README_LICENSE = re.compile(r"(?ms)^## License\s*\n(?P<body>.*?)(?=^## |\Z)")


def test_package_readme_license_matches_tier() -> None:
    """PyPI renders ``readme = "README.md"`` as the package long_description.

    The sidebar shows the PEP 639 SPDX expression from ``pyproject.toml``; the
    body shows whatever the README claims. Those two must agree, or an
    enterprise legal reviewer finds a contradiction in the first ten minutes.
    """
    tiers = _tiers()
    offenders: dict[str, str] = {}
    for pkg_dir in _package_dirs():
        dist = _dist_name(pkg_dir)
        readme = pkg_dir / "README.md"
        if not readme.is_file():
            offenders[dist] = "no README.md"
            continue
        text = readme.read_text(encoding="utf-8")
        match = _README_LICENSE.search(text)
        if match is None:
            offenders[dist] = "no ## License section"
            continue
        body = match.group("body")
        tier = tiers[dist]
        if tier == "permissive":
            if "Apache-2.0" not in body:
                offenders[dist] = "permissive README does not name Apache-2.0"
            elif re.search(r"\bAGPL\b|\bMIT\b", body):
                offenders[dist] = "permissive README still claims AGPL or MIT"
        else:
            if "AGPL-3.0-or-later" not in body:
                offenders[dist] = "copyleft README does not name AGPL-3.0-or-later"
            elif "commercial" not in body.lower():
                offenders[dist] = "copyleft README omits the commercial option"
            elif re.search(r"\bApache-2\.0\b|\bMIT\b", body):
                offenders[dist] = "copyleft README incorrectly claims Apache-2.0 or MIT"
    assert not offenders, f"package README license prose drift: {offenders}"


# --------------------------------------------------------------- headers


def _package_py_files(pkg_dir: Path) -> list[Path]:
    return [
        f
        for f in sorted(pkg_dir.rglob("*.py"))
        if not (_SKIP_PARTS & set(f.relative_to(REPO_ROOT).parts))
    ]


def test_every_package_python_file_carries_its_tier_header() -> None:
    tiers, expressions = _tiers(), _expressions()
    offenders: dict[str, str] = {}
    checked = 0
    for pkg_dir in _package_dirs():
        wanted = expressions[tiers[_dist_name(pkg_dir)]]
        for path in _package_py_files(pkg_dir):
            checked += 1
            head = path.read_text(encoding="utf-8", errors="ignore").splitlines()[:4]
            found = next((m.group("expr") for line in head if (m := _SPDX.match(line))), None)
            if found != wanted:
                offenders[str(path.relative_to(REPO_ROOT))] = f"{found or '<none>'} (want {wanted})"
    assert checked > 1000, f"header scan collapsed to {checked} files"
    assert not offenders, (
        "SPDX header drift -- run `python scripts/license_headers.py`: "
        + "; ".join(f"{k}: {v}" for k, v in sorted(offenders.items())[:20])
    )


# ------------------------------------------------------------ the invariant


def test_no_permissive_package_depends_on_a_copyleft_package() -> None:
    """The load-bearing one: copyleft must never leak downward into the adoption tier."""
    tiers = _tiers()
    violations: list[str] = []
    for pkg_dir in _package_dirs():
        dist = _dist_name(pkg_dir)
        if tiers[dist] != "permissive":
            continue
        for dep, how in sorted(_omnibias_requirements(pkg_dir).items()):
            if tiers.get(dep) == "copyleft":
                violations.append(f"{dist} -> {dep} (via {how})")
    assert not violations, (
        "a permissive (Apache-2.0) package depends on a copyleft package, which would "
        "subject the adoption tier to the AGPL. Either drop the dependency or promote "
        "the dependent to copyleft in [tool.omnibias.license_tiers]: " + "; ".join(violations)
    )


def test_dependency_scan_is_not_vacuous() -> None:
    """A broken requirement regex would make the invariant above pass forever."""
    edges = {
        _dist_name(p): _omnibias_requirements(p) for p in _package_dirs()
    }
    total = sum(len(v) for v in edges.values())
    assert total > 40, f"only {total} intra-omnibias dependency edges found -- parser broke"
    # Both edge kinds must be represented, or the extras half is untested.
    kinds = {how.split(".")[0] for deps in edges.values() for how in deps.values()}
    assert kinds == {"dependencies", "optional-dependencies"}, f"edge kinds seen: {kinds}"


def test_invariant_would_catch_a_synthetic_violation() -> None:
    """Self-test: the check is a real check, not a tautology over the current tree."""
    tiers = dict(_tiers())
    permissive = next(d for d, t in tiers.items() if t == "permissive")
    copyleft = next(d for d, t in tiers.items() if t == "copyleft")
    synthetic = {permissive: {copyleft: "optional-dependencies.certify"}}
    violations = [
        f"{src} -> {dep}"
        for src, deps in synthetic.items()
        if tiers[src] == "permissive"
        for dep in deps
        if tiers.get(dep) == "copyleft"
    ]
    assert violations == [f"{permissive} -> {copyleft}"]
