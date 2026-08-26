# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Frontier 09-14: integral-kernel operator.

G1 matches the OMBU integral cell. G2 is the named Volterra
antiderivative of cos. G3 records not BEM-Net / not FNO SOTA.
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
from omnibias.core.integral_kernel import (
    DISCLAIMER,
    antiderivative_skill,
    honesty_payload,
    worked_example,
)
from omnibias.jax.architectures.integral_kernel import integral_cell as jax_cell
from omnibias.jax.architectures.integral_kernel import worked_example as jax_ex
from omnibias.torch.architectures.integral_kernel import integral_cell as torch_cell
from omnibias.torch.architectures.integral_kernel import worked_example as torch_ex

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
    g1 = bool(ex["g1_err"] < 1e-12)
    skill = antiderivative_skill()
    g2 = bool(skill["below_1e3"] and skill["skill_positive"])
    hon = honesty_payload()
    g3 = hon["claimed_bem_net"] is False and hon["claimed_fno_sota"] is False
    jax.config.update("jax_enable_x64", True)
    torch.set_default_dtype(torch.float64)
    t = float(torch_cell(torch.tensor([0.5]), -0.5, 0.5)[0])
    j = float(jax_cell(jnp.asarray([0.5]), -0.5, 0.5)[0])
    g4 = bool(torch_ex()["g1_cell"] == jax_ex()["g1_cell"] and t == j)
    entries = [
        {
            "name": "g1_cell",
            "passed": g1,
            "g1_err": ex["g1_err"],
            "mass": ex["mass"],
        },
        {
            "name": "g2_skill",
            "passed": g2,
            "max_err": skill["max_err"],
            "skill": skill["skill"],
            "mlp_median": skill["mlp_median"],
            "beats_mlp": skill["beats_mlp"],
            "g2_earned": skill["g2_earned"],
        },
        {
            "name": "g3_honesty",
            "passed": g3,
            "claimed_bem_net": False,
            "claimed_fno_sota": False,
        },
        {
            "name": "g4_parity",
            "passed": g4,
            "torch_cell": t,
            "jax_cell": j,
        },
    ]
    for entry in entries:
        print(entry["name"], "ok" if entry["passed"] else "FAIL")
        if not entry["passed"]:
            raise AssertionError(f"{entry['name']} failed: {entry}")
    payload = {
        **provenance(
            schema="omnibias.benchmarks.integral_kernel_operator.v1",
            config={"family": "integral_kernel_operator", "full": full, "honesty": hon},
        ),
        "gates": dict(gates_block(entries)),
        "disclaimer": DISCLAIMER,
        "wall_seconds": time.perf_counter() - t0,
    }
    name = "integral_kernel_operator.json" if full else "integral_kernel_operator_smoke.json"
    if full:
        dest = SCRATCH / "inventions" / "integral_kernel"
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / name
        path.write_text(__import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8")
    else:
        path = write_json(name, payload)
    print(f"wrote {path}")
    return payload


if __name__ == "__main__":
    main()
