# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Frontier 09-16: exact-MAML (IFT meta-grad, not ImageNet).

G1 is the quadratic IFT identity. G2 is one inner Newton vs five
inner Adam on a 1-D Poisson family. G3 keeps ImageNet / stretch
claims false. G4 is torch/jax bit-identity on G1. Jets are
founding bias collapse, not temperature collapse. IFT is the
chain rule.
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
from omnibias.core.exact_maml import (
    DISCLAIMER,
    honesty_payload,
    poisson_skill,
    quadratic_worked_example,
)
from omnibias.jax.optim_maml import inner_newton_quadratic as jax_step
from omnibias.torch.optim_maml import inner_newton_quadratic as torch_step

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
    ex = quadratic_worked_example()
    g1 = bool(ex["inner_err"] < 1e-12 and ex["ift_err"] < 1e-10)
    skill = poisson_skill()
    g2 = bool(skill["below_1e4"] and skill["beats_adam"] and float(skill["skill"]) > 0.0)
    hon = honesty_payload()
    g3 = hon["imagenet_claim"] is False and hon["stretch_claim"] is False
    jax.config.update("jax_enable_x64", True)
    torch.set_default_dtype(torch.float64)
    t = torch_step(torch.tensor(2.5), torch.tensor(-0.3))
    j = jax_step(jnp.asarray(2.5), jnp.asarray(-0.3))
    g4 = bool(float(t) == float(j))
    entries = [
        {"name": "g1_ift", "passed": g1, "inner_err": ex["inner_err"], "ift_err": ex["ift_err"]},
        {
            "name": "g2_skill",
            "passed": g2,
            "newton_median": skill["newton_median"],
            "adam_median": skill["adam_median"],
            "skill": skill["skill"],
        },
        {"name": "g3_honesty", "passed": g3, "imagenet_claim": False, "stretch_claim": False},
        {
            "name": "g4_parity",
            "passed": g4,
            "torch_step": float(t),
            "jax_step": float(j),
        },
    ]
    for entry in entries:
        print(entry["name"], "ok" if entry["passed"] else "FAIL")
        if not entry["passed"]:
            raise AssertionError(f"{entry['name']} failed: {entry}")
    payload = {
        **provenance(
            schema="omnibias.benchmarks.exact_maml.v1",
            config={"family": "exact_maml", "full": full, "honesty": hon},
        ),
        "gates": dict(gates_block(entries)),
        "disclaimer": DISCLAIMER,
        "wall_seconds": time.perf_counter() - t0,
    }
    name = "exact_maml.json" if full else "exact_maml_smoke.json"
    if full:
        dest = SCRATCH / "inventions" / "exact_maml"
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / name
        path.write_text(__import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8")
    else:
        path = write_json(name, payload)
    print(f"wrote {path}")
    return payload


if __name__ == "__main__":
    main()
