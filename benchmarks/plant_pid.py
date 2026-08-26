# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Frontier 09-29: plant PID layer with exact I / D.

G1 is FTC vs trapezoid plus exact D. G2 is random windows.
G3 records not-a-trainer. G4 is torch/jax bit-identity on G1.
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
from omnibias.core.pid_layer import (
    DISCLAIMER,
    PlantPIDConfig,
    honesty_payload,
    plant_pid,
    plant_pid_skill,
    worked_example,
)
from omnibias.jax.plant_pid import plant_pid as jax_pid
from omnibias.jax.plant_pid import worked_example as jax_ex
from omnibias.torch.plant_pid import plant_pid as torch_pid
from omnibias.torch.plant_pid import worked_example as torch_ex

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
    g1 = bool(float(ex["ftc_residual"]) < 1e-8 and float(ex["d_residual"]) < 1e-15)
    skill = plant_pid_skill()
    g2 = bool(skill["g2_earned"])
    hon = honesty_payload()
    g3 = hon["trainer_claimed"] is False and hon["discrete_sum_i"] is False
    jax.config.update("jax_enable_x64", True)
    torch.set_default_dtype(torch.float64)
    cfg = PlantPIDConfig(kp=0.0, ki=1.0, kd=1.0, setpoint=0.5, t0=-1.0)
    core = plant_pid(0.0, config=cfg)
    t_rep = torch_pid(torch.tensor(0.0), config=cfg)
    j_rep = jax_pid(jnp.asarray(0.0), config=cfg)
    g4 = bool(
        torch_ex() == jax_ex()
        and t_rep.control == j_rep.control == core.control
        and t_rep.integral == j_rep.integral
    )
    entries = [
        {
            "name": "g1_ftc",
            "passed": g1,
            "ftc_residual": ex["ftc_residual"],
            "d_residual": ex["d_residual"],
        },
        {
            "name": "g2_skill",
            "passed": g2,
            "median_ftc_residual": skill["median_ftc_residual"],
            "unknown_raised": skill["unknown_raised"],
            "g2_earned": skill["g2_earned"],
        },
        {
            "name": "g3_honesty",
            "passed": g3,
            "trainer_claimed": False,
            "discrete_sum_i": False,
        },
        {
            "name": "g4_parity",
            "passed": g4,
            "torch_control": t_rep.control,
            "jax_control": j_rep.control,
        },
    ]
    for entry in entries:
        print(entry["name"], "ok" if entry["passed"] else "FAIL")
        if not entry["passed"]:
            raise AssertionError(f"{entry['name']} failed: {entry}")
    payload = {
        **provenance(
            schema="omnibias.benchmarks.plant_pid.v1",
            config={"family": "plant_pid", "full": full, "honesty": hon},
        ),
        "gates": dict(gates_block(entries)),
        "disclaimer": DISCLAIMER,
        "wall_seconds": time.perf_counter() - t0,
    }
    name = "plant_pid.json" if full else "plant_pid_smoke.json"
    if full:
        dest = SCRATCH / "inventions" / "plant_pid"
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / name
        path.write_text(__import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8")
    else:
        path = write_json(name, payload)
    print(f"wrote {path}")
    return payload


if __name__ == "__main__":
    main()
