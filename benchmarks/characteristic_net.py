# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Frontier 09-08: Characteristic-Net (transport with a time integral).

G1 is frozen ``v=1`` on the Gaussian foot. G2 learns ``v`` on linear
advection. G3 flags Burgers past breaking. G4 is torch/jax
bit-identity on G1. Not 02-13. Not NS. Not a shock-capturing
theorem. Time integral is the window knob; ``v`` jets are founding
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
import jax.numpy as jnp
import torch
from omnibias.pinn.characteristic import (
    DISCLAIMER,
    constant_v,
    gaussian_u0,
    honesty_payload,
    learn_v_skill,
    shock_flag_report,
    worked_example,
)
from omnibias.pinn.jax.characteristic import characteristic_eval as jax_eval
from omnibias.pinn.jax.characteristic import worked_example as jax_ex
from omnibias.pinn.torch.characteristic import characteristic_eval as torch_eval
from omnibias.pinn.torch.characteristic import worked_example as torch_ex

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
    g1 = bool(ex["err"] < 1e-12 and ex["crossed"] == 0.0)
    skill = learn_v_skill()
    g2 = bool(skill["g2_earned"] and skill["v_below_1e3"])
    shock = shock_flag_report()
    hon = honesty_payload()
    g3 = bool(shock["crossed_any"] and hon["unique_after_shock_claimed"] is False)
    jax.config.update("jax_enable_x64", True)
    torch.set_default_dtype(torch.float64)
    tu, tc = torch_eval(torch.tensor(0.0), 0.2, constant_v, gaussian_u0)
    ju, jc = jax_eval(jnp.asarray(0.0), 0.2, constant_v, gaussian_u0)
    g4 = bool(
        torch_ex()["u"] == jax_ex()["u"]
        and float(tu) == float(ju)
        and float(tc) == float(jc)
    )
    entries = [
        {"name": "g1_constant_v", "passed": g1, "u": ex["u"], "err": ex["err"]},
        {
            "name": "g2_skill",
            "passed": g2,
            "char_median": skill["char_median"],
            "pinn_median": skill["pinn_median"],
            "v_dev_median": skill["v_dev_median"],
            "g2_earned": skill["g2_earned"],
        },
        {
            "name": "g3_shock_flag",
            "passed": g3,
            "crossed_any": shock["crossed_any"],
            "unique_after_shock_claimed": False,
        },
        {
            "name": "g4_parity",
            "passed": g4,
            "torch_u": float(tu),
            "jax_u": float(ju),
        },
    ]
    for entry in entries:
        print(entry["name"], "ok" if entry["passed"] else "FAIL")
        if not entry["passed"]:
            raise AssertionError(f"{entry['name']} failed: {entry}")
    payload = {
        **provenance(
            schema="omnibias.benchmarks.characteristic_net.v1",
            config={"family": "characteristic_net", "full": full, "honesty": hon},
        ),
        "gates": dict(gates_block(entries)),
        "disclaimer": DISCLAIMER,
        "wall_seconds": time.perf_counter() - t0,
    }
    name = "characteristic_net.json" if full else "characteristic_net_smoke.json"
    if full:
        dest = SCRATCH / "inventions" / "characteristic"
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / name
        path.write_text(__import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8")
    else:
        path = write_json(name, payload)
    print(f"wrote {path}")
    return payload


if __name__ == "__main__":
    main()
