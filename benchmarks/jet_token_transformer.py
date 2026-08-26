# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Frontier 09-02: jet-token transformer (model jet, not ImageNet).

G1 matches compose_jet on the worked mix. G2 is five-seed skill vs a
value-only mix on ``u = sin x``. G3 keeps ``imagenet_claim`` false.
G4 is torch/jax bit-identity on the worked mix. Jets are founding
bias collapse, not temperature collapse.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import jax
import torch
from omnibias.core.jet_token import DISCLAIMER, honesty_payload, jet_token_skill, worked_example
from omnibias.jax.architectures.jet_token import worked_compose_jet as jax_worked
from omnibias.torch.architectures.jet_token import worked_compose_jet as torch_worked

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
    g1 = bool(ex["value_err"] < 1e-12 and ex["deriv_err"] < 1e-12)
    skill = jet_token_skill()
    g2 = bool(skill["below_1e3"] and skill["beats_value"] and float(skill["skill"]) > 0.0)
    g3 = honesty_payload()["imagenet_claim"] is False
    jax.config.update("jax_enable_x64", True)
    torch.set_default_dtype(torch.float64)
    t = torch_worked()
    j = jax_worked()
    g4 = bool(float(t[0]) == float(j[0]) and float(t[1]) == float(j[1]))
    entries = [
        {"name": "g1_algebra", "passed": g1, "value_err": ex["value_err"], "deriv_err": ex["deriv_err"]},
        {
            "name": "g2_skill",
            "passed": g2,
            "jet_median": skill["jet_median"],
            "value_median": skill["value_median"],
            "skill": skill["skill"],
        },
        {"name": "g3_honesty", "passed": g3, "imagenet_claim": False},
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
            schema="omnibias.benchmarks.jet_token_transformer.v1",
            config={"family": "jet_token_transformer", "full": full, "honesty": honesty_payload()},
        ),
        "gates": dict(gates_block(entries)),
        "disclaimer": DISCLAIMER,
        "wall_seconds": time.perf_counter() - t0,
    }
    name = "jet_token_transformer.json" if full else "jet_token_transformer_smoke.json"
    if full:
        dest = SCRATCH / "inventions" / "jet_token"
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / name
        path.write_text(__import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8")
    else:
        path = write_json(name, payload)
    print(f"wrote {path}")
    return payload


if __name__ == "__main__":
    main()
