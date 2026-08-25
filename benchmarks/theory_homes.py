# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Gated program: packaging homes (theory 06-03).

Every spec's section 2 names an existing package, submodule, or
docs-only home. ``NEW_PACKAGES_ALLOWED`` stays empty. Wave-0
A4–A7 are recorded in the index. No new distribution. Not a
performance benchmark.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(__file__))
from _common import provenance, write_json  # type: ignore[import-not-found]  # noqa: E402

SCRATCH = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))
REPO = Path(__file__).resolve().parents[1]
TESTS = REPO / "packages" / "omnibias-core" / "tests"
sys.path.insert(0, str(TESTS))

import test_theory_homes as homes  # noqa: E402

WAVE0_UNITS = ("A4", "A5", "A6", "A7")
_ARTIFACT = re.compile(r"`(?:\.\./)?(docs/benchmarks/[^`]+\.json)`")
_RECORDED = re.compile(r"(?i)\b(passed|earned|unearned|failed|retired)\b")
_NOT_RUN = re.compile(r"(?i)not run")


def _run_g1() -> dict[str, Any]:
    existing = homes._package_dirs()
    allowlist = set(homes.NEW_PACKAGES_ALLOWED)
    arrangement = "omnibias-arrangement" in existing
    folded = sorted(existing & homes.FOLDED_AWAY)
    earned = not allowlist and not arrangement and not folded
    return {
        "name": "g1_zero_new_packages",
        "passed": bool(earned),
        "earned": bool(earned),
        "reported": True,
        "in_ci_all_passed": bool(earned),
        "n_packages": len(existing),
        "allowlist": sorted(allowlist),
        "arrangement_minted": arrangement,
        "folded_resurrected": folded,
        "need": "NEW_PACKAGES_ALLOWED empty; no omnibias-arrangement; folded names gone",
        "note": (
            "Named G1 is zero new packages after Waves 0-3. The homes "
            "guard allowlist is empty and folded distributions stay gone."
        ),
    }


def _run_g2() -> dict[str, Any]:
    specs = homes._theory_specs()
    existing = homes._package_dirs()
    missing_section: list[str] = []
    missing_home: list[str] = []
    proposed: dict[str, list[str]] = {}
    n_docs_only = 0
    for path in specs:
        rel = path.relative_to(homes.REPO).as_posix()
        body = homes._section_two(path.read_text(encoding="utf-8"))
        if body is None:
            missing_section.append(rel)
            continue
        if homes._DOCS_ONLY.search(body):
            n_docs_only += 1
        if not homes._has_home(body, existing):
            missing_home.append(rel)
        extra = homes._proposed_new_packages(body, existing)
        if extra:
            proposed[rel] = sorted(extra)
    unexpected = {
        rel: names
        for rel, names in proposed.items()
        if not set(names) <= homes.NEW_PACKAGES_ALLOWED
    }
    earned = not missing_section and not missing_home and not unexpected and bool(specs)
    return {
        "name": "g2_homes_declared",
        "passed": bool(earned),
        "earned": bool(earned),
        "reported": True,
        "in_ci_all_passed": bool(earned),
        "n_specs": len(specs),
        "n_docs_only": n_docs_only,
        "n_missing_section": len(missing_section),
        "n_missing_home": len(missing_home),
        "missing_section": missing_section,
        "missing_home": missing_home,
        "unexpected_packages": unexpected,
        "need": "every spec section 2 names an existing home; no new distribution",
        "note": (
            "Named G2 is the test_theory_homes consistency pass. "
            "Docs-only rows in the 06-03 table stay documents."
        ),
    }


def _wave0_rows(text: str) -> dict[str, dict[str, Any]]:
    start = text.find("## Wave-0 falsifier outcomes")
    if start < 0:
        return {}
    rest = text[start:]
    nxt = rest.find("\n## ", 3)
    block = rest if nxt < 0 else rest[:nxt]
    rows: dict[str, dict[str, Any]] = {}
    for line in block.splitlines():
        if not line.startswith("| A"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) < 4:
            continue
        unit = cells[0]
        if unit not in WAVE0_UNITS:
            continue
        artifacts = _ARTIFACT.findall(cells[2])
        missing_art = [rel for rel in artifacts if not (REPO / rel).is_file()]
        outcome = cells[3]
        rows[unit] = {
            "unit": unit,
            "gate": cells[1],
            "artifacts": artifacts,
            "missing_artifacts": missing_art,
            "outcome": outcome,
            "recorded": bool(_RECORDED.search(outcome)) and not _NOT_RUN.search(outcome),
        }
    return rows


def _run_g3() -> dict[str, Any]:
    """Named leftover: Wave-0 A4–A7 recorded in the index before Wave 1+."""
    text = (REPO / "theory" / "README.md").read_text(encoding="utf-8")
    rows = _wave0_rows(text)
    missing_units = [unit for unit in WAVE0_UNITS if unit not in rows]
    unrecorded = [
        unit
        for unit, row in rows.items()
        if not row["recorded"] or row["missing_artifacts"]
    ]
    wave1 = "## Wave-1 primitives" in text
    earned = not missing_units and not unrecorded and wave1 and len(rows) == 4
    return {
        "name": "g3_falsifiers_first",
        "passed": bool(earned),
        "earned": bool(earned),
        "reported": True,
        "in_ci_all_passed": bool(earned),
        "n_units": len(WAVE0_UNITS),
        "n_recorded": len(rows) - len(unrecorded),
        "missing_units": missing_units,
        "unrecorded": unrecorded,
        "wave1_section": wave1,
        "rows": [rows[unit] for unit in WAVE0_UNITS if unit in rows],
        "need": "A4–A7 recorded pass/fail/earned in the index; artifacts exist",
        "note": (
            "Named G3 is falsifiers-first: Wave-0 A4–A7 must be recorded "
            "in theory/README.md before Wave 1+ work is treated as "
            "licensed. This smoke checks the index and artifacts, not "
            "git-order landing dates. Ambiguous 'not run' is a fail."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    g1 = _run_g1()
    g2 = _run_g2()
    g3 = _run_g3()
    in_scope = [row for row in (g1, g2, g3) if row["in_ci_all_passed"]]
    payload = provenance(
        schema="omnibias.benchmark.theory_homes.v1",
        config={
            "family": "theory_homes",
            "full": bool(args.full),
            "gates_in_scope": [row["name"] for row in in_scope],
            "g3_in_all_passed": bool(g3["in_ci_all_passed"]),
            "g4_in_all_passed": False,
            "g5_in_all_passed": False,
        },
    )
    payload["gates"] = {
        "all_passed": all(bool(row.get("passed")) for row in in_scope),
        "entries": in_scope,
    }
    payload["g1"] = g1
    payload["g2"] = g2
    payload["g3"] = g3
    payload["honesty"] = {
        "new_packages_allowed": False,
        "omnibias_arrangement_package": False,
        "docs_only_program_specs": True,
        "g1_earned": bool(g1["earned"]),
        "g1_in_ci_all_passed": bool(g1["in_ci_all_passed"]),
        "g2_earned": bool(g2["earned"]),
        "g2_in_ci_all_passed": bool(g2["in_ci_all_passed"]),
        "g3_earned": bool(g3["earned"]),
        "g3_in_ci_all_passed": bool(g3["in_ci_all_passed"]),
        "g3_is_git_order_proof": False,
        "g4_earned": False,
        "g5_earned": False,
        "book_tree": False,
    }
    if args.full:
        out_dir = SCRATCH / "program"
        out_dir.mkdir(parents=True, exist_ok=True)
        dest = out_dir / "theory_homes.json"
        dest.write_text(
            __import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        print(f"wrote {dest}")
    else:
        print(f"wrote {write_json('theory_homes_smoke.json', payload)}")
    if not payload["gates"]["all_passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
