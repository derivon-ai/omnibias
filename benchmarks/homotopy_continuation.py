# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Frontier 09-20: Kantorovich homotopy continuation.

G1 accepts the tau=0.1 quadratic step. G2 is honest halt or
semilinear reach. G3 records no stretch claim. G4 is torch/jax
bit-identity on G1.
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
from omnibias.core.homotopy import (
    DISCLAIMER,
    homotopy_skill,
    honesty_payload,
    worked_example,
)
from omnibias.jax.optim_homotopy import homotopy_step as jax_step
from omnibias.jax.optim_homotopy import worked_example as jax_ex
from omnibias.torch.optim_homotopy import homotopy_step as torch_step
from omnibias.torch.optim_homotopy import worked_example as torch_ex

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
    g1 = bool(ex["accepted"] and float(ex["abs_h"]) < 1e-10)
    skill = homotopy_skill()
    g2 = bool(skill["g2_earned"] and skill["honest"])
    hon = honesty_payload()
    g3 = hon["stretch_claim"] is False and hon["theorem_prover_verified"] is False
    jax.config.update("jax_enable_x64", True)
    torch.set_default_dtype(torch.float64)
    t_trial, t_res, t_dec = torch_step(torch.tensor(1.0), 0.1)
    j_trial, j_res, j_dec = jax_step(jnp.asarray(1.0), 0.1)
    g4 = bool(
        torch_ex()["theta"] == jax_ex()["theta"]
        and t_trial == j_trial
        and t_res == j_res
        and t_dec.accepted == j_dec.accepted
    )
    entries = [
        {
            "name": "g1_cell",
            "passed": g1,
            "accepted": ex["accepted"],
            "abs_h": ex["abs_h"],
            "theta": ex["theta"],
        },
        {
            "name": "g2_skill",
            "passed": g2,
            "reached": skill["reached"],
            "quad_halted": skill["quad_halted"],
            "quad_tau": skill["quad_tau"],
            "honest": skill["honest"],
            "g2_earned": skill["g2_earned"],
        },
        {
            "name": "g3_honesty",
            "passed": g3,
            "stretch_claim": False,
            "empty_ball_is_reject": True,
        },
        {
            "name": "g4_parity",
            "passed": g4,
            "torch_accepted": t_dec.accepted,
            "jax_accepted": j_dec.accepted,
        },
    ]
    for entry in entries:
        print(entry["name"], "ok" if entry["passed"] else "FAIL")
        if not entry["passed"]:
            raise AssertionError(f"{entry['name']} failed: {entry}")
    payload = {
        **provenance(
            schema="omnibias.benchmarks.homotopy_continuation.v1",
            config={"family": "homotopy_continuation", "full": full, "honesty": hon},
        ),
        "gates": dict(gates_block(entries)),
        "disclaimer": DISCLAIMER,
        "wall_seconds": time.perf_counter() - t0,
    }
    name = "homotopy_continuation.json" if full else "homotopy_continuation_smoke.json"
    if full:
        dest = SCRATCH / "inventions" / "homotopy"
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / name
        path.write_text(__import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8")
    else:
        path = write_json(name, payload)
    print(f"wrote {path}")
    return payload


if __name__ == "__main__":
    main()
