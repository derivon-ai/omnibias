# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Frontier 09-08: Characteristic-Net (transport with a time integral).

G1 is frozen ``v=1`` on the Gaussian foot. G2 learns ``v`` on linear
advection. G3 flags Burgers past breaking. Not 02-13. Not NS. Not a
shock-capturing theorem. Time integral is the window knob; ``v`` jets
are founding bias collapse, not temperature collapse.
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
from omnibias.pinn.characteristic import (
    DISCLAIMER,
    honesty_payload,
    learn_v_skill,
    shock_flag_report,
    worked_example,
)

SCRATCH = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args(argv)
    full = bool(args.full)
    t0 = time.perf_counter()
    ex = worked_example()
    g1 = bool(ex["err"] < 1e-12 and ex["crossed"] == 0.0)
    skill = learn_v_skill()
    g2 = bool(skill["g2_earned"] and skill["v_below_1e3"])
    shock = shock_flag_report()
    hon = honesty_payload()
    g3 = bool(shock["crossed_any"] and hon["unique_after_shock_claimed"] is False)
    entries = [
        {"name": "g1_constant_v", "passed": g1, "u": ex["u"], "err": ex["err"]},
        {
            "name": "g2_skill",
            "passed": g2,
            "char_median": skill["char_median"],
            "pinn_median": skill["pinn_median"],
            "v_dev_median": skill["v_dev_median"],
            "g2_earned": skill["g2_earned"],
        },
        {
            "name": "g3_shock_flag",
            "passed": g3,
            "crossed_any": shock["crossed_any"],
            "unique_after_shock_claimed": False,
        },
    ]
    for entry in entries:
        print(entry["name"], "ok" if entry["passed"] else "FAIL")
        if not entry["passed"]:
            raise AssertionError(f"{entry['name']} failed: {entry}")
    payload = {
        **provenance(
            schema="omnibias.benchmarks.characteristic_net.v1",
            config={"family": "characteristic_net", "full": full, "honesty": hon},
        ),
        "gates": dict(gates_block(entries)),
        "disclaimer": DISCLAIMER,
        "wall_seconds": time.perf_counter() - t0,
    }
    name = "characteristic_net.json" if full else "characteristic_net_smoke.json"
    if full:
        dest = SCRATCH / "inventions" / "characteristic"
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / name
        path.write_text(__import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8")
    else:
        path = write_json(name, payload)
    print(f"wrote {path}")
    return payload


if __name__ == "__main__":
    main()
