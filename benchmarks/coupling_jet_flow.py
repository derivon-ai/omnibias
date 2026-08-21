# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Frontier 09-06: coupling jet-flow (not integrate_cnf).

G1 is the tanh(2x) log-det identity. G2 is a 64-point Newton
round-trip. G3 is five-seed NLL skill vs an isotropic Gaussian on
a 2-D two-Gaussian mixture. Jets are founding bias collapse, not
temperature collapse. Not ImageNet. Not CCF stretch.
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
from omnibias.core.coupling_flow import (
    DISCLAIMER,
    honesty_payload,
    mixture_samples,
    mixture_skill,
    roundtrip_grid,
    worked_example,
)

SCRATCH = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))


def _cnf_nll_closed(xs: list[tuple[float, float]]) -> float:
    """Equal-wall OU stand-in of ``integrate_cnf`` (v = -x, T = 0.3).

    Distinct from the coupling stack. Used only as a smoke-budget
    baseline so G3 can say 'not worse than a short CNF'.
    """
    import math

    t_end = 0.3
    scale = math.exp(-t_end)
    log_det = -2.0 * t_end
    log_2pi = math.log(2.0 * math.pi)
    acc = 0.0
    for x0, x1 in xs:
        z0, z1 = scale * x0, scale * x1
        acc += 0.5 * (z0 * z0 + z1 * z1) + log_2pi - log_det
    return acc / float(len(xs))


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args(argv)
    full = bool(args.full)
    t0 = time.perf_counter()
    ex = worked_example()
    g1 = bool(ex["log_det_err"] < 1e-12 and ex["inv_err"] < 1e-12)
    rt = roundtrip_grid()
    g2 = rt < 1e-10
    skill = mixture_skill()
    cnf = _cnf_nll_closed(mixture_samples(256, 0))
    g3 = bool(skill["beats_iso"] and float(skill["flow_mean"]) <= cnf and float(skill["skill"]) > 0.0)
    hon = honesty_payload()
    entries = [
        {"name": "g1_det", "passed": g1, "log_det_err": ex["log_det_err"], "inv_err": ex["inv_err"]},
        {"name": "g2_invert", "passed": g2, "roundtrip": rt},
        {
            "name": "g3_skill",
            "passed": g3,
            "flow_mean": skill["flow_mean"],
            "iso_mean": skill["iso_mean"],
            "cnf_mean": cnf,
            "skill": skill["skill"],
        },
        {
            "name": "g4_honesty",
            "passed": hon["imagenet_claim"] is False and hon["rewrites_integrate_cnf"] is False,
            "imagenet_claim": False,
        },
    ]
    for entry in entries:
        print(entry["name"], "ok" if entry["passed"] else "FAIL")
        if not entry["passed"]:
            raise AssertionError(f"{entry['name']} failed: {entry}")
    payload = {
        **provenance(
            schema="omnibias.benchmarks.coupling_jet_flow.v1",
            config={"family": "coupling_jet_flow", "full": full, "honesty": hon},
        ),
        "gates": dict(gates_block(entries)),
        "disclaimer": DISCLAIMER,
        "wall_seconds": time.perf_counter() - t0,
    }
    name = "coupling_jet_flow.json" if full else "coupling_jet_flow_smoke.json"
    if full:
        dest = SCRATCH / "inventions" / "jet_flow"
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / name
        path.write_text(__import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8")
    else:
        path = write_json(name, payload)
    print(f"wrote {path}")
    return payload


if __name__ == "__main__":
    main()
