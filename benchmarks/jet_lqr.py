# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Wave-3 trainer: jet-LQR on a directional restriction (theory 08-11).

G1 is Newton recovery plus R>0 shrinkage. G2 is random bowls plus a
singular refuse. G3 records no DARE / plant LQR. G4 is torch/jax
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
from omnibias.core.control_lqr import (
    DISCLAIMER,
    JetLQRConfig,
    honesty_payload,
    lqr_skill,
    lqr_step_from_derivatives,
    worked_example,
)
from omnibias.jax.optim_lqr import jet_lqr_step as jax_lqr
from omnibias.jax.optim_lqr import worked_example as jax_ex
from omnibias.torch.optim_lqr import jet_lqr_step as torch_lqr
from omnibias.torch.optim_lqr import worked_example as torch_ex

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
    g1 = bool(ex["newton_recovered"] and ex["damped_smaller"])
    skill = lqr_skill()
    g2 = bool(skill["g2_earned"])
    hon = honesty_payload()
    g3 = (
        hon["algebraic_riccati_claimed"] is False
        and hon["activation_riccati_as_gain"] is False
        and hon["plant_lqr_claimed"] is False
    )
    jax.config.update("jax_enable_x64", True)
    torch.set_default_dtype(torch.float64)
    cfg = JetLQRConfig(q=0.0, r=0.0, qf=1.0, horizon=1, s_max=4.0)

    def t_loss(params: torch.Tensor) -> torch.Tensor:
        return (params[0] - 1.0) ** 2

    def j_loss(params: jax.Array) -> jax.Array:
        return (params[0] - 1.0) ** 2

    t_rep = torch_lqr(t_loss, torch.zeros(1), torch.ones(1), config=cfg)
    j_rep = jax_lqr(j_loss, jnp.zeros(1), jnp.ones(1), config=cfg)
    core_rep = lqr_step_from_derivatives((1.0, -2.0, 2.0), config=cfg)
    g4 = bool(
        torch_ex() == jax_ex()
        and t_rep.step == j_rep.step
        and t_rep.step == core_rep.step
        and t_rep.gain == j_rep.gain
    )
    entries = [
        {
            "name": "g1_newton",
            "passed": g1,
            "step": ex["step"],
            "damped_step": ex["damped_step"],
        },
        {
            "name": "g2_skill",
            "passed": g2,
            "median_err": skill["median_err"],
            "singular_raised": skill["singular_raised"],
            "g2_earned": skill["g2_earned"],
        },
        {
            "name": "g3_honesty",
            "passed": g3,
            "algebraic_riccati_claimed": False,
            "activation_riccati_as_gain": False,
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
            schema="omnibias.benchmarks.jet_lqr.v1",
            config={"family": "jet_lqr", "full": full, "honesty": hon},
        ),
        "gates": dict(gates_block(entries)),
        "disclaimer": DISCLAIMER,
        "wall_seconds": time.perf_counter() - t0,
    }
    name = "jet_lqr.json" if full else "jet_lqr_smoke.json"
    if full:
        dest = SCRATCH / "training" / "jet_lqr"
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / name
        path.write_text(__import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8")
    else:
        path = write_json(name, payload)
    print(f"wrote {path}")
    return payload


if __name__ == "__main__":
    main()
