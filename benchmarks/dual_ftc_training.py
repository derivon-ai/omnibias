# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Frontier 09-17: dual-FTC training (not a VPINN).

G1: residuals vanish when ``f = dI/dx``. G2: dual loss beats a
derivative-only ablation on ``r_I``. G3 keeps ``claimed_vpinn`` false.
G4 is torch/jax bit-identity on G1. Jets are founding bias collapse,
not temperature collapse.
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
from omnibias.core.ftc import (
    DISCLAIMER,
    dual_ftc_loss,
    dual_skill_report,
    ftc_block,
    honesty_payload,
)
from omnibias.jax.architectures.ftc_net import dual_ftc_loss as jax_dual
from omnibias.jax.architectures.ftc_net import ftc_block as jax_ftc
from omnibias.torch.architectures.ftc_net import dual_ftc_loss as torch_dual
from omnibias.torch.architectures.ftc_net import ftc_block as torch_ftc

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
    integral, deriv, _collapse = ftc_block(0.0, 1.0, -0.1, 0.1)
    ident = dual_ftc_loss([integral], [deriv], [deriv], [0.0], I_a=integral)
    g1 = ident.max_r_D < 1e-12 and ident.max_r_I < 1e-12
    skill = dual_skill_report()
    g2 = bool(skill["below_1e4"] and skill["ri_below_deriv_only"] and float(skill["skill"]) > 0.0)
    g3 = honesty_payload()["claimed_vpinn"] is False
    jax.config.update("jax_enable_x64", True)
    torch.set_default_dtype(torch.float64)
    ti, td, tc = torch_ftc(
        torch.tensor(0.0), torch.tensor(1.0), torch.tensor(-0.1), torch.tensor(0.1)
    )
    ji, jd, jc = jax_ftc(
        jnp.asarray(0.0), jnp.asarray(1.0), jnp.asarray(-0.1), jnp.asarray(0.1)
    )
    t_loss = torch_dual([float(ti)], [float(td)], [float(td)], [0.0], I_a=float(ti))
    j_loss = jax_dual([float(ji)], [float(jd)], [float(jd)], [0.0], I_a=float(ji))
    g4 = bool(
        float(ti) == float(ji)
        and float(td) == float(jd)
        and float(tc) == float(jc)
        and t_loss.max_r_D == j_loss.max_r_D
        and t_loss.max_r_I == j_loss.max_r_I
    )
    entries = [
        {"name": "g1_identity", "passed": g1, "max_r_D": ident.max_r_D, "max_r_I": ident.max_r_I},
        {
            "name": "g2_skill",
            "passed": g2,
            "median_r_I": skill["median_r_I"],
            "median_deriv_only_r_I": skill["median_deriv_only_r_I"],
            "skill": skill["skill"],
        },
        {"name": "g3_not_vpinn", "passed": g3, "claimed_vpinn": False},
        {
            "name": "g4_parity",
            "passed": g4,
            "torch_r_I": t_loss.max_r_I,
            "jax_r_I": j_loss.max_r_I,
        },
    ]
    for entry in entries:
        print(entry["name"], "ok" if entry["passed"] else "FAIL")
        if not entry["passed"]:
            raise AssertionError(f"{entry['name']} failed: {entry}")
    payload = {
        **provenance(
            schema="omnibias.benchmarks.dual_ftc_training.v1",
            config={"family": "dual_ftc_training", "full": full, "honesty": honesty_payload()},
        ),
        "gates": dict(gates_block(entries)),
        "disclaimer": DISCLAIMER,
        "wall_seconds": time.perf_counter() - t0,
    }
    name = "dual_ftc_training.json" if full else "dual_ftc_training_smoke.json"
    if full:
        dest = SCRATCH / "inventions" / "dual_ftc"
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / name
        path.write_text(__import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8")
    else:
        path = write_json(name, payload)
    print(f"wrote {path}")
    return payload


if __name__ == "__main__":
    main()
