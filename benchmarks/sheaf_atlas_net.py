# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Frontier 09-09: sheaf-atlas net (jet cocycle on partition charts).

G1 is the worked ``2x`` then ``y/2`` identity. G2 is two-interval
Poisson with a nonempty overlap. G3 records temperature collapse
separately from founding bias collapse. G4 is torch/jax
bit-identity on G1. Not a sheaf-cohomology theorem. Not P vs NP.
Not CCF stretch.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import torch
from omnibias.geometry.atlas.cocycle import (
    DISCLAIMER,
    AffineChart,
    honesty_payload,
    two_interval_poisson,
    worked_example,
)
from omnibias.geometry.atlas.jax import cocycle_residual as jax_res
from omnibias.geometry.atlas.jax import worked_example as jax_ex
from omnibias.geometry.atlas.torch import cocycle_residual as torch_res
from omnibias.geometry.atlas.torch import worked_example as torch_ex

sys.path.insert(0, os.path.dirname(__file__))
from _common import provenance, write_json  # type: ignore[import-not-found]  # noqa: E402
from _gates import gates_block  # type: ignore[import-not-found]  # noqa: E402

SCRATCH = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args(argv)
    full = bool(args.full)
    t0 = time.perf_counter()
    ex = worked_example()
    g1 = bool(ex["residual"] < 1e-12 and ex["buggy_deriv_gap"] > 1e-3)
    skill = two_interval_poisson()
    g2 = bool(
        skill["nonempty_overlap"]
        and float(skill["skill_vs_zero"]) > 0.0
        and skill["below_1e6"]
        and not skill["ignored_cocycle"]
    )
    hon = honesty_payload()
    g3 = hon["temperature_collapse_used"] is False and hon["sheaf_cohomology_theorem"] is False
    jax.config.update("jax_enable_x64", True)
    torch.set_default_dtype(torch.float64)
    transitions = {
        (2, 1): AffineChart(2.0),
        (3, 2): AffineChart(0.5),
        (3, 1): AffineChart(1.0),
    }
    t = torch_res(transitions, torch.tensor(0.5), ((1, 2, 3),))
    j = jax_res(transitions, jnp.asarray(0.5), ((1, 2, 3),))
    g4 = bool(torch_ex()["residual"] == jax_ex()["residual"] and t["residual"] == j["residual"])
    entries = [
        {
            "name": "g1_algebra",
            "passed": g1,
            "residual": ex["residual"],
            "buggy_deriv_gap": ex["buggy_deriv_gap"],
        },
        {
            "name": "g2_skill",
            "passed": g2,
            "cocycle_mean": skill["cocycle_mean"],
            "overlap_min": skill["overlap_min"],
            "skill_vs_zero": skill["skill_vs_zero"],
        },
        {
            "name": "g3_honesty",
            "passed": g3,
            "temperature_collapse_used": False,
        },
        {
            "name": "g4_parity",
            "passed": g4,
            "torch_residual": t["residual"],
            "jax_residual": j["residual"],
        },
    ]
    for entry in entries:
        print(entry["name"], "ok" if entry["passed"] else "FAIL")
        if not entry["passed"]:
            raise AssertionError(f"{entry['name']} failed: {entry}")
    payload = {
        **provenance(
            schema="omnibias.benchmarks.sheaf_atlas_net.v1",
            config={"family": "sheaf_atlas_net", "full": full, "honesty": hon},
        ),
        "gates": dict(gates_block(entries)),
        "disclaimer": DISCLAIMER,
        "wall_seconds": time.perf_counter() - t0,
    }
    name = "sheaf_atlas_net.json" if full else "sheaf_atlas_net_smoke.json"
    if full:
        dest = SCRATCH / "inventions" / "sheaf_atlas"
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / name
        path.write_text(__import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8")
    else:
        path = write_json(name, payload)
    print(f"wrote {path}")
    return payload


if __name__ == "__main__":
    main()
