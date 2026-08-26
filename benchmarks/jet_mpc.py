# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Wave-3 trainer: receding jet-MPC (theory 08-12).

G1 is Newton recovery plus an input box. G2 is random bowls plus
box / remainder. G3 records no plant MPC. G4 is torch/jax
bit-identity on G1. Not CCF stretch.
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
from omnibias.core.control_mpc import (
    DISCLAIMER,
    JetMPCConfig,
    honesty_payload,
    mpc_skill,
    mpc_step_from_derivatives,
    worked_example,
)
from omnibias.jax.optim_mpc import jet_mpc_step as jax_mpc
from omnibias.jax.optim_mpc import worked_example as jax_ex
from omnibias.torch.optim_mpc import jet_mpc_step as torch_mpc
from omnibias.torch.optim_mpc import worked_example as torch_ex

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
    g1 = bool(ex["newton_recovered"] and ex["boxed"] and ex["receding"])
    skill = mpc_skill()
    g2 = bool(skill["g2_earned"])
    hon = honesty_payload()
    g3 = hon["plant_mpc_claimed"] is False and hon["general_qp_claimed"] is False
    jax.config.update("jax_enable_x64", True)
    torch.set_default_dtype(torch.float64)
    cfg = JetMPCConfig(q=0.0, r=0.0, qf=1.0, horizon=1, u_max=4.0)

    def t_loss(params: torch.Tensor) -> torch.Tensor:
        return (params[0] - 1.0) ** 2

    def j_loss(params: jax.Array) -> jax.Array:
        return (params[0] - 1.0) ** 2

    t_rep = torch_mpc(t_loss, torch.zeros(1), torch.ones(1), config=cfg)
    j_rep = jax_mpc(j_loss, jnp.zeros(1), jnp.ones(1), config=cfg)
    core_rep = mpc_step_from_derivatives((1.0, -2.0, 2.0), config=cfg)
    g4 = bool(
        torch_ex() == jax_ex()
        and t_rep.step == j_rep.step
        and t_rep.step == core_rep.step
        and t_rep.receding is True
    )
    entries = [
        {
            "name": "g1_recede",
            "passed": g1,
            "step": ex["step"],
            "boxed_step": ex["boxed_step"],
        },
        {
            "name": "g2_skill",
            "passed": g2,
            "median_err": skill["median_err"],
            "boxed": skill["boxed"],
            "refused": skill["refused"],
            "g2_earned": skill["g2_earned"],
        },
        {
            "name": "g3_honesty",
            "passed": g3,
            "plant_mpc_claimed": False,
            "general_qp_claimed": False,
        },
        {
            "name": "g4_parity",
            "passed": g4,
            "torch_step": t_rep.step,
            "jax_step": j_rep.step,
        },
    ]
    for entry in entries:
        print(entry["name"], "ok" if entry["passed"] else "FAIL")
        if not entry["passed"]:
            raise AssertionError(f"{entry['name']} failed: {entry}")
    payload = {
        **provenance(
            schema="omnibias.benchmarks.jet_mpc.v1",
            config={"family": "jet_mpc", "full": full, "honesty": hon},
        ),
        "gates": dict(gates_block(entries)),
        "disclaimer": DISCLAIMER,
        "wall_seconds": time.perf_counter() - t0,
    }
    name = "jet_mpc.json" if full else "jet_mpc_smoke.json"
    if full:
        dest = SCRATCH / "training" / "jet_mpc"
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / name
        path.write_text(__import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8")
    else:
        path = write_json(name, payload)
    print(f"wrote {path}")
    return payload


if __name__ == "__main__":
    main()
