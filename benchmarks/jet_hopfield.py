# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Frontier 09-13: Jet-Hopfield (germs, not vectors).

G1 retrieves a near-J1 query. G2 contact vs value-only split.
G3 records temperature collapse only for infinite beta. G4 is
torch/jax bit-identity on G1. Not ImageNet. Not CCF stretch.
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
from omnibias.core.jet_hopfield import (
    DISCLAIMER,
    JetHopfieldConfig,
    contact_split,
    honesty_payload,
    worked_example,
)
from omnibias.jax.architectures.jet_hopfield import jet_hopfield_retrieve as jax_ret
from omnibias.jax.architectures.jet_hopfield import worked_example as jax_ex
from omnibias.torch.architectures.jet_hopfield import jet_hopfield_retrieve as torch_ret
from omnibias.torch.architectures.jet_hopfield import worked_example as torch_ex

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
    g1 = bool(ex["value_err"] < 1e-6)
    skill = contact_split()
    g2 = bool(skill["split"])
    hon = honesty_payload()
    g3 = hon["temperature_collapse_used"] is False
    jax.config.update("jax_enable_x64", True)
    torch.set_default_dtype(torch.float64)
    cfg = JetHopfieldConfig(lam=1.0, beta=10.0)
    t = torch_ret(torch.tensor([1.0, 0.01]), torch.tensor([[1.0, 0.0], [0.0, 1.0]]), config=cfg)
    j = jax_ret(jnp.asarray([1.0, 0.01]), jnp.asarray([[1.0, 0.0], [0.0, 1.0]]), config=cfg)
    g4 = bool(
        torch_ex()["retrieved_value"] == jax_ex()["retrieved_value"]
        and float(t[0]) == float(j[0])
        and float(t[1]) == float(j[1])
    )
    entries = [
        {
            "name": "g1_near_j1",
            "passed": g1,
            "retrieved_value": ex["retrieved_value"],
            "value_err": ex["value_err"],
        },
        {
            "name": "g2_skill",
            "passed": g2,
            "lam": skill["lam"],
            "jet_indices": skill["jet_indices"],
            "value_indices": skill["value_indices"],
        },
        {
            "name": "g3_honesty",
            "passed": g3,
            "temperature_collapse_used": False,
        },
        {
            "name": "g4_parity",
            "passed": g4,
            "torch_value": float(t[0]),
            "jax_value": float(j[0]),
        },
    ]
    for entry in entries:
        print(entry["name"], "ok" if entry["passed"] else "FAIL")
        if not entry["passed"]:
            raise AssertionError(f"{entry['name']} failed: {entry}")
    payload = {
        **provenance(
            schema="omnibias.benchmarks.jet_hopfield.v1",
            config={"family": "jet_hopfield", "full": full, "honesty": hon, "lam": 1.0},
        ),
        "gates": dict(gates_block(entries)),
        "disclaimer": DISCLAIMER,
        "wall_seconds": time.perf_counter() - t0,
    }
    name = "jet_hopfield.json" if full else "jet_hopfield_smoke.json"
    if full:
        dest = SCRATCH / "inventions" / "jet_hopfield"
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / name
        path.write_text(__import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8")
    else:
        path = write_json(name, payload)
    print(f"wrote {path}")
    return payload


if __name__ == "__main__":
    main()
