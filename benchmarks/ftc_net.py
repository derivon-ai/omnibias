# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Frontier 09-03: FTC-Net (integral cell; not a VPINN).

G1 is the worked window identity. G2 is five-seed skill vs an
identity-cell baseline on ``dI/dx = cos x``. G3 keeps
``claimed_weak_form`` false. Jets are founding bias collapse, not
temperature collapse.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(__file__))
from _common import provenance, write_json  # type: ignore[import-not-found]  # noqa: E402
from _gates import gates_block  # type: ignore[import-not-found]  # noqa: E402
from omnibias.core.ftc import DISCLAIMER, honesty_payload, skill_report, worked_example

SCRATCH = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args(argv)
    full = bool(args.full)
    t0 = time.perf_counter()
    ex = worked_example()
    g1 = bool(ex["collapse_err"] < 1e-12 and ex["ftc_err"] < 1e-12)
    skill = skill_report()
    g2 = bool(skill["below_1e4"] and skill["beats_identity"] and float(skill["skill"]) > 0.0)
    g3 = honesty_payload()["claimed_weak_form"] is False
    entries = [
        {"name": "g1_ftc_identity", "passed": g1, "collapse_err": ex["collapse_err"], "ftc_err": ex["ftc_err"]},
        {
            "name": "g2_skill",
            "passed": g2,
            "ftc_median": skill["ftc_median"],
            "identity_median": skill["identity_median"],
            "skill": skill["skill"],
        },
        {"name": "g3_not_vpinn", "passed": g3, "claimed_weak_form": False},
    ]
    for entry in entries:
        print(entry["name"], "ok" if entry["passed"] else "FAIL")
        if not entry["passed"]:
            raise AssertionError(f"{entry['name']} failed: {entry}")
    payload = {
        **provenance(
            schema="omnibias.benchmarks.ftc_net.v1",
            config={"family": "ftc_net", "full": full, "honesty": honesty_payload()},
        ),
        "gates": dict(gates_block(entries)),
        "disclaimer": DISCLAIMER,
        "wall_seconds": time.perf_counter() - t0,
    }
    name = "ftc_net.json" if full else "ftc_net_smoke.json"
    if full:
        dest = SCRATCH / "inventions" / "ftc_net"
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / name
        path.write_text(__import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8")
    else:
        path = write_json(name, payload)
    print(f"wrote {path}")
    return payload


if __name__ == "__main__":
    main()
