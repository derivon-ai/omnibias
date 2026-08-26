# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Frontier 09-11: Collapse-Net (train stencil, infer by collapse).

G1 matches the spec remainder. G2 collapsed eval of ``d/dx sin x``
beats stencil eval on a denser probe. G3 refuses a continuum PDE
claim. G4 is torch/jax bit-identity on G1. Founding bias collapse
only. Not CCF stretch.
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
from omnibias.core.collapse_net import (
    DISCLAIMER,
    CollapseNetConfig,
    honesty_payload,
    sin_skill,
    worked_example,
)
from omnibias.jax.architectures.collapse_net import collapse_net_forward as jax_fwd
from omnibias.jax.architectures.collapse_net import collapse_remainder as jax_rem
from omnibias.jax.architectures.collapse_net import worked_example as jax_ex
from omnibias.torch.architectures.collapse_net import collapse_net_forward as torch_fwd
from omnibias.torch.architectures.collapse_net import collapse_remainder as torch_rem
from omnibias.torch.architectures.collapse_net import worked_example as torch_ex

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
    g1 = bool(ex["rel_err"] < 1e-6)
    skill = sin_skill()
    g2 = bool(skill["below_1e3"] and skill["skill_positive"] and skill["collapsed_no_worse"])
    hon = honesty_payload()
    g3 = hon["continuum_pde_claimed"] is False
    jax.config.update("jax_enable_x64", True)
    torch.set_default_dtype(torch.float64)
    cfg = CollapseNetConfig(order=1, delta=0.1, mode="stencil")
    t = torch_fwd(torch.tensor(0.0), config=cfg)
    j = jax_fwd(jnp.asarray(0.0), config=cfg)
    tr = torch_rem(torch.tensor(0.0), config=cfg)
    jr = jax_rem(jnp.asarray(0.0), config=cfg)
    g4 = bool(
        torch_ex()["remainder"] == jax_ex()["remainder"]
        and float(t) == float(j)
        and float(tr) == float(jr)
    )
    entries = [
        {
            "name": "g1_remainder",
            "passed": g1,
            "remainder": ex["remainder"],
            "rel_err": ex["rel_err"],
        },
        {
            "name": "g2_skill",
            "passed": g2,
            "max_err": skill["max_err"],
            "skill": skill["skill"],
            "collapsed_probe_max": skill["collapsed_probe_max"],
            "stencil_probe_max": skill["stencil_probe_max"],
        },
        {
            "name": "g3_honesty",
            "passed": g3,
            "continuum_pde_claimed": False,
        },
        {
            "name": "g4_parity",
            "passed": g4,
            "torch_y": float(t),
            "jax_y": float(j),
        },
    ]
    for entry in entries:
        print(entry["name"], "ok" if entry["passed"] else "FAIL")
        if not entry["passed"]:
            raise AssertionError(f"{entry['name']} failed: {entry}")
    payload = {
        **provenance(
            schema="omnibias.benchmarks.collapse_net.v1",
            config={"family": "collapse_net", "full": full, "honesty": hon},
        ),
        "gates": dict(gates_block(entries)),
        "disclaimer": DISCLAIMER,
        "wall_seconds": time.perf_counter() - t0,
    }
    name = "collapse_net.json" if full else "collapse_net_smoke.json"
    if full:
        dest = SCRATCH / "inventions" / "collapse_net"
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / name
        path.write_text(__import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8")
    else:
        path = write_json(name, payload)
    print(f"wrote {path}")
    return payload


if __name__ == "__main__":
    main()
