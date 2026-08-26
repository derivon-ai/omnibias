# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Frontier 09-10: Riccati flow net (depth is integration time).

G1 matches the closed-form logistic flow. G2 fits ``T`` on logistic
samples and beats a width-8 sigmoid MLP. G3 refuses DEQ / CNF claims.
G4 is torch/jax bit-identity on G1. Not ImageNet. Not CCF stretch.
This forward is not bias collapse.
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
from omnibias.core.riccati_flow import (
    DISCLAIMER,
    RiccatiFlowConfig,
    flow_skill,
    honesty_payload,
    worked_example,
)
from omnibias.jax.architectures.riccati_flow import riccati_flow as jax_flow
from omnibias.jax.architectures.riccati_flow import worked_example as jax_ex
from omnibias.torch.architectures.riccati_flow import riccati_flow as torch_flow
from omnibias.torch.architectures.riccati_flow import worked_example as torch_ex

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
    g1 = bool(ex["err"] < 1e-12 and ex["ds_ds0_err"] < 1e-12)
    skill = flow_skill()
    g2 = bool(skill["below_1e6"] and skill["beats_mlp"])
    hon = honesty_payload()
    g3 = hon["claimed_deq"] is False and hon["claimed_cnf"] is False
    jax.config.update("jax_enable_x64", True)
    torch.set_default_dtype(torch.float64)
    t = torch_flow(torch.tensor(0.25), config=RiccatiFlowConfig(t=1.0))
    j = jax_flow(jnp.asarray(0.25), config=RiccatiFlowConfig(t=1.0))
    g4 = bool(torch_ex()["s"] == jax_ex()["s"] and float(t) == float(j))
    entries = [
        {"name": "g1_closed_form", "passed": g1, "s": ex["s"], "ds_ds0": ex["ds_ds0"]},
        {
            "name": "g2_skill",
            "passed": g2,
            "flow_median": skill["flow_median"],
            "mlp_median": skill["mlp_median"],
        },
        {
            "name": "g3_honesty",
            "passed": g3,
            "claimed_deq": False,
            "claimed_cnf": False,
        },
        {
            "name": "g4_parity",
            "passed": g4,
            "torch_s": float(t),
            "jax_s": float(j),
        },
    ]
    for entry in entries:
        print(entry["name"], "ok" if entry["passed"] else "FAIL")
        if not entry["passed"]:
            raise AssertionError(f"{entry['name']} failed: {entry}")
    payload = {
        **provenance(
            schema="omnibias.benchmarks.riccati_flow_net.v1",
            config={"family": "riccati_flow_net", "full": full, "honesty": hon},
        ),
        "gates": dict(gates_block(entries)),
        "disclaimer": DISCLAIMER,
        "wall_seconds": time.perf_counter() - t0,
    }
    name = "riccati_flow_net.json" if full else "riccati_flow_net_smoke.json"
    if full:
        dest = SCRATCH / "inventions" / "riccati_flow"
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / name
        path.write_text(__import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8")
    else:
        path = write_json(name, payload)
    print(f"wrote {path}")
    return payload


if __name__ == "__main__":
    main()
