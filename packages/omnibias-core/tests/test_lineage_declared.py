# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Every public package must declare -- and then honour -- its founding-idea lineage.

omnibias has two founding limits (bias collapse / temperature collapse) plus a
small exempt set. A package that ships without saying which one it sits on is
how the conflation crept in in the first place. This guard requires a module-
level ``__lineage__`` string on every ``packages/omnibias-*`` top-level
``__init__.py``, and then cross-checks that the package's own prose -- its
``pyproject.toml`` description and its top-level docstring -- does not describe
its mechanism as the *other* collapse.

Naming the other collapse to *disambiguate* ("distinct from the founding bias
collapse", "this is not temperature collapse") is exactly the discipline we want
and is allowed. Naming it *attributively* ("the annealed bias-collapse penalty"
on a beta -> inf package) is the bug this guard exists to catch.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import tomllib

REPO = Path(__file__).resolve().parents[3]
PACKAGES = REPO / "packages"

BIAS = "bias collapse"
TEMPERATURE = "temperature collapse"

ALLOWED = re.compile(r"^(bias collapse|temperature collapse|both|exempt:.+)$")

# Both spellings, since prose alternates between the noun and the adjective.
_TERM = {
    BIAS: re.compile(r"bias[ \-]collapse", re.IGNORECASE),
    TEMPERATURE: re.compile(r"temperature[ \-]collapse", re.IGNORECASE),
}

#: A mention of the *other* collapse is fine when it is being contrasted away.
#: The marker must appear in the run of text immediately preceding the mention.
#: Matched against a whitespace-normalised window, so a marker may wrap a line.
CONTRAST_MARKERS = re.compile(
    r"\b(?:not|never|nor|unlike|versus|vs\.?|distinct|distinguished|separate|"
    r"opposed|rather than|differs?|generali[sz]ed|two senses|two axes|"
    r"second sense|other sense)\b",
    re.IGNORECASE,
)

#: How far back to look for a contrast marker.
CONTRAST_WINDOW = 200


def _package_inits() -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for dist in sorted(PACKAGES.glob("omnibias-*")):
        name = dist.name.removeprefix("omnibias-")
        init = dist / "src" / "omnibias" / name / "__init__.py"
        out.append((name, init))
    return out


def _lineage_literal(tree: ast.Module) -> str | None:
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "__lineage__" for t in node.targets):
            continue
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return node.value.value
    return None


def attributive_mentions(text: str, term: str) -> list[str]:
    """Return every mention of ``term`` in ``text`` that is *not* a disambiguation.

    Exposed (rather than inlined) so the self-test below can exercise it on
    synthetic text and prove the checker is not vacuous.
    """
    offenders: list[str] = []
    for match in _TERM[term].finditer(text):
        window = text[max(0, match.start() - CONTRAST_WINDOW) : match.start()]
        if CONTRAST_MARKERS.search(" ".join(window.split())):
            continue
        start = max(0, match.start() - 70)
        offenders.append(" ".join(text[start : match.end() + 40].split()))
    return offenders


def test_every_package_declares_lineage() -> None:
    missing: list[str] = []
    bad: list[str] = []
    for name, init in _package_inits():
        if not init.is_file():
            missing.append(f"{name}: no __init__.py")
            continue
        value = _lineage_literal(ast.parse(init.read_text(encoding="utf-8")))
        if value is None:
            missing.append(name)
        elif not ALLOWED.match(value):
            bad.append(f"{name}: {value!r}")
    assert not missing, f"packages missing __lineage__: {missing}"
    assert not bad, (
        "illegal __lineage__ values (expected 'bias collapse' / "
        f"'temperature collapse' / 'both' / 'exempt: <reason>'): {bad}"
    )


def test_package_prose_matches_declared_lineage() -> None:
    """No package may claim the collapse it did not declare."""
    offenders: dict[str, list[str]] = {}
    checked = 0
    for name, init in _package_inits():
        if not init.is_file():
            continue
        tree = ast.parse(init.read_text(encoding="utf-8"))
        lineage = _lineage_literal(tree)
        if lineage == BIAS:
            forbidden = TEMPERATURE
        elif lineage == TEMPERATURE:
            forbidden = BIAS
        else:  # "both" may use either; "exempt:" sits off both axes.
            continue
        checked += 1

        pyproject = tomllib.loads((PACKAGES / f"omnibias-{name}" / "pyproject.toml").read_text())
        description = pyproject.get("project", {}).get("description", "")
        docstring = ast.get_docstring(tree) or ""

        hits = attributive_mentions(description, forbidden)
        hits += attributive_mentions(docstring, forbidden)
        if hits:
            offenders[f"{name} (__lineage__ = {lineage!r})"] = hits

    assert checked >= 20, f"guard went vacuous: only {checked} single-axis packages checked"
    assert not offenders, (
        "packages describe their mechanism as the collapse they did not declare. "
        "Either fix the prose or fix __lineage__; if the mention is a deliberate "
        "contrast, phrase it as one (e.g. 'distinct from the founding bias collapse'). "
        + "; ".join(f"{k}: {v}" for k, v in sorted(offenders.items()))
    )


def test_attributive_mention_checker_is_not_vacuous() -> None:
    """The checker must flag a bare claim and clear a genuine disambiguation."""
    claim = "solved differentiably by the annealed bias-collapse penalty"
    assert attributive_mentions(claim, BIAS), "checker missed a bare attributive claim"

    disambiguation = (
        "The gate's beta -> inf limit is the feasibility sense of collapse, "
        "distinct from the founding bias collapse (the multi-bias delta -> 0 limit)."
    )
    assert not attributive_mentions(disambiguation, BIAS), (
        "checker flagged a legitimate disambiguation"
    )

    assert attributive_mentions("a temperature collapse tower", TEMPERATURE)
    assert not attributive_mentions("a delta -> 0 limit, not temperature collapse", TEMPERATURE)
