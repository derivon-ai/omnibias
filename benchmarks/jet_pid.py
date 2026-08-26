# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Wave-3 trainer: jet-PID on a directional restriction (theory 08-10).

G1 is the section-5 bowl. G2 is random bowls plus a kd-without-phi''
reject. G3 records no global min / plant PID. G4 is torch/jax
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
from omnibias.core.control_pid import (
    DISCLAIMER,
    JetPIDConfig,
    honesty_payload,
    pid_skill,
    pid_step_from_derivatives,
    worked_example,
)
from omnibias.jax.optim_pid import jet_pid_step as jax_pid
from omnibias.jax.optim_pid import worked_example as jax_ex
from omnibias.torch.optim_pid import jet_pid_step as torch_pid
from omnibias.torch.optim_pid import worked_example as torch_ex

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
    g1 = bool(ex["abs_step_err"] < 1e-15 and ex["damped_is_newton"])
    skill = pid_skill()
    g2 = bool(skill["g2_earned"])
    hon = honesty_payload()
    g3 = (
        hon["global_min_claimed"] is False
        and hon["plant_pid_claimed"] is False
        and hon["skips_chain_rule"] is False
    )
    jax.config.update("jax_enable_x64", True)
    torch.set_default_dtype(torch.float64)
    cfg = JetPIDConfig(kp=0.5, ki=0.0, kd=0.0, s_max=2.0)

    def t_loss(params: torch.Tensor) -> torch.Tensor:
        return (params[0] - 1.0) ** 2

    def j_loss(params: jax.Array) -> jax.Array:
        return (params[0] - 1.0) ** 2

    t_rep = torch_pid(t_loss, torch.zeros(1), torch.ones(1), config=cfg, order=2)
    j_rep = jax_pid(j_loss, jnp.zeros(1), jnp.ones(1), config=cfg, order=2)
    core_rep = pid_step_from_derivatives((1.0, -2.0, 2.0), config=cfg)
    g4 = bool(
        torch_ex() == jax_ex()
        and t_rep.step == j_rep.step
        and t_rep.step == core_rep.step
        and t_rep.proportional == j_rep.proportional
    )
    entries = [
        {
            "name": "g1_bowl",
            "passed": g1,
            "step": ex["step"],
            "abs_step_err": ex["abs_step_err"],
            "damped_step": ex["damped_step"],
        },
        {
            "name": "g2_skill",
            "passed": g2,
            "median_err": skill["median_err"],
            "kd_raised": skill["kd_raised"],
            "g2_earned": skill["g2_earned"],
        },
        {
            "name": "g3_honesty",
            "passed": g3,
            "global_min_claimed": False,
            "plant_pid_claimed": False,
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
            schema="omnibias.benchmarks.jet_pid.v1",
            config={"family": "jet_pid", "full": full, "honesty": hon},
        ),
        "gates": dict(gates_block(entries)),
        "disclaimer": DISCLAIMER,
        "wall_seconds": time.perf_counter() - t0,
    }
    name = "jet_pid.json" if full else "jet_pid_smoke.json"
    if full:
        dest = SCRATCH / "training" / "jet_pid"
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / name
        path.write_text(__import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8")
    else:
        path = write_json(name, payload)
    print(f"wrote {path}")
    return payload


if __name__ == "__main__":
    main()
