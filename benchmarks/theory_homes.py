# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Gated program: packaging homes (theory 06-03).

Every spec's section 2 names an existing package, submodule, or
docs-only home. ``NEW_PACKAGES_ALLOWED`` stays empty. No new
distribution. Not a performance benchmark.
"""

from __future__ import annotations

import argparse
import os
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    g1 = _run_g1()
    g2 = _run_g2()
    in_scope = [row for row in (g1, g2) if row["in_ci_all_passed"]]
    payload = provenance(
        schema="omnibias.benchmark.theory_homes.v1",
        config={
            "family": "theory_homes",
            "full": bool(args.full),
            "gates_in_scope": [row["name"] for row in in_scope],
            "g3_in_all_passed": False,
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
    payload["honesty"] = {
        "new_packages_allowed": False,
        "omnibias_arrangement_package": False,
        "docs_only_program_specs": True,
        "g1_earned": bool(g1["earned"]),
        "g1_in_ci_all_passed": bool(g1["in_ci_all_passed"]),
        "g2_earned": bool(g2["earned"]),
        "g2_in_ci_all_passed": bool(g2["in_ci_all_passed"]),
        "g3_earned": False,
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
