#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
"""Adversarial Hilbert-16 campaign verification benchmark."""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any

sys.path.insert(0, os.path.dirname(__file__))
from _common import provenance, write_json  # type: ignore[import-not-found]  # noqa: E402
from _gates import gates_block  # type: ignore[import-not-found]  # noqa: E402
from omnibias.dynamics.drr_audit import audit_report  # noqa: E402
from omnibias.dynamics.hilbert16_ledger import (  # noqa: E402
    default_h16_ledger,
    derived_parent_flags,
    enrich_ledger_with_campaign_evidence,
)
from omnibias.dynamics.open_drr_endpoint import attack_open_endpoint, g1_chart_reaudit  # noqa: E402
from omnibias.geometry.part_a_target import (  # noqa: E402
    PARTA_SELECTED_TARGET,
    sibling_octic_region_tree,
    wide_deep_octic_region_tree,
)


def _parent_flags_gate() -> dict[str, Any]:
    flags = derived_parent_flags(enrich_ledger_with_campaign_evidence(default_h16_ledger()))
    passed = not any(flags.values())
    return {
        "name": "parent_flags_derived_false",
        "passed": passed,
        "flags": flags,
        "detail": "no parent Hilbert-16 claim is earned on the shipped ledger",
    }


def _part_a_trees_gate() -> dict[str, Any]:
    wide = wide_deep_octic_region_tree()
    sibling = sibling_octic_region_tree()
    from omnibias.geometry.part_a_target import count_octic_ovals

    passed = count_octic_ovals(wide) == 22 and count_octic_ovals(sibling) == 22
    return {
        "name": "part_a_target_trees_22_ovals",
        "passed": passed,
        "selected": PARTA_SELECTED_TARGET.scheme,
        "detail": "both corrected (19,3) trees encode exactly 22 ovals",
    }


def _drr_inventory_gate() -> dict[str, Any]:
    report = audit_report()
    return {
        "name": "drr_inventory_complete",
        "passed": report["total"] == 121,
        "open_count": report["by_status"].get("open", 0),
        "detail": "121-graphic inventory is present",
    }


def _g1_reaudit_gate() -> dict[str, Any]:
    report = g1_chart_reaudit()
    return {
        "name": "g1_chart_reaudit_route_specific",
        "passed": report["parent_g1_passed"] is False,
        "detail": report["detail"],
    }


def _open_endpoint_gate() -> dict[str, Any]:
    report = attack_open_endpoint("DF_1b")
    return {
        "name": "open_endpoint_df_1b_blocked",
        "passed": report.status == "open" and report.g1_route_specific_obstruction,
        "status": report.status,
        "detail": report.detail,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    started = time.time()
    entries = [
        _parent_flags_gate(),
        _part_a_trees_gate(),
        _drr_inventory_gate(),
        _g1_reaudit_gate(),
        _open_endpoint_gate(),
    ]
    report = {
        **provenance(
            schema="omnibias.benchmarks.hilbert16_campaign_verify.v1",
            config={"family": "hilbert16_campaign_verify", "full": args.full},
        ),
        "benchmark": "hilbert16_campaign_verify",
        "gates": gates_block(entries),
        "full_hilbert16_solved": False,
        "wall_seconds": time.time() - started,
    }
    artifact = "hilbert16_campaign_verify.json" if args.full else "hilbert16_campaign_verify_smoke.json"
    path = write_json(artifact, report)
    print(f"wrote {path}")
    return 0 if report["gates"]["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
