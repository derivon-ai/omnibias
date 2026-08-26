# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Frontier 09-22: Newton-on-input inverse design.

G1 inverts tanh(2x)=0.5. G2 inverts random unsaturated
targets and refuses |y|=0.999. G3 records no global inverse.
G4 is torch/jax bit-identity on G1.
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
from omnibias.core.inverse_design import (
    DISCLAIMER,
    honesty_payload,
    invert_skill,
    worked_example,
)
from omnibias.jax.optim_inverse import invert_input as jax_inv
from omnibias.jax.optim_inverse import worked_example as jax_ex
from omnibias.torch.optim_inverse import invert_input as torch_inv
from omnibias.torch.optim_inverse import worked_example as torch_ex

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
    g1 = ex["residual"] < 1e-12
    skill = invert_skill()
    g2 = bool(skill["g2_earned"])
    hon = honesty_payload()
    g3 = hon["global_inverse_claimed"] is False
    jax.config.update("jax_enable_x64", True)
    torch.set_default_dtype(torch.float64)
    t_rep = torch_inv(None, torch.tensor(0.5), torch.tensor(0.0))
    j_rep = jax_inv(None, jnp.asarray(0.5), jnp.asarray(0.0))
    g4 = bool(
        torch_ex()["x"] == jax_ex()["x"]
        and t_rep.x == j_rep.x
        and t_rep.residual == j_rep.residual
    )
    entries = [
        {
            "name": "g1_cell",
            "passed": g1,
            "x": ex["x"],
            "residual": ex["residual"],
            "f_prime": ex["f_prime"],
        },
        {
            "name": "g2_skill",
            "passed": g2,
            "median_err": skill["median_err"],
            "saturated_raised": skill["saturated_raised"],
            "g2_earned": skill["g2_earned"],
        },
        {
            "name": "g3_honesty",
            "passed": g3,
            "global_inverse_claimed": False,
        },
        {
            "name": "g4_parity",
            "passed": g4,
            "torch_x": t_rep.x,
            "jax_x": j_rep.x,
        },
    ]
    for entry in entries:
        print(entry["name"], "ok" if entry["passed"] else "FAIL")
        if not entry["passed"]:
            raise AssertionError(f"{entry['name']} failed: {entry}")
    payload = {
        **provenance(
            schema="omnibias.benchmarks.inverse_design.v1",
            config={"family": "inverse_design", "full": full, "honesty": hon},
        ),
        "gates": dict(gates_block(entries)),
        "disclaimer": DISCLAIMER,
        "wall_seconds": time.perf_counter() - t0,
    }
    name = "inverse_design.json" if full else "inverse_design_smoke.json"
    if full:
        dest = SCRATCH / "inventions" / "inverse_design"
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / name
        path.write_text(__import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8")
    else:
        path = write_json(name, payload)
    print(f"wrote {path}")
    return payload


if __name__ == "__main__":
    main()
