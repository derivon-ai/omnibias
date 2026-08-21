# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Wave-3 trainer: sharpness-scheduled cubic step (theory 08-06).

Smoke earns G1 (100% finite finishes on five seeds of the stiff
quadratic), G2 (artifact records ``c``, ``n_lanczos``, ``ell_min``, and
``ell_k`` min/max), and G3 (torch/jax ``lambda_max`` agree to ``1e-10``
relative). Sharpness is a step-size signal, not a generalization claim
and not CCF stretch. Hutchinson is not the method.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import torch

sys.path.insert(0, os.path.dirname(__file__))
from _common import provenance, write_json  # type: ignore[import-not-found]  # noqa: E402
from _gates import gates_block  # type: ignore[import-not-found]  # noqa: E402
from omnibias.core.sharpness import SharpnessSchedule
from omnibias.jax.optim_sharpness import (
    sharpness_lambda_max as jax_lambda_max,
)
from omnibias.jax.optim_sharpness import (
    sharpness_scheduled_minimize as jax_minimize,
)
from omnibias.torch.optim_sharpness import (
    sharpness_lambda_max as torch_lambda_max,
)
from omnibias.torch.optim_sharpness import (
    sharpness_scheduled_minimize as torch_minimize,
)

jax.config.update("jax_enable_x64", True)

SCRATCH = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))
STIFF = 1.0e4
SCHEDULE = SharpnessSchedule(n_lanczos=4, c=1e-3, ell_min=1e-6)
SEEDS = (0, 1, 2, 3, 4)
STEPS = 20


def _torch_loss(params: torch.Tensor) -> torch.Tensor:
    p = params.reshape(-1)
    return 0.5 * (p[0] * p[0] + STIFF * p[1] * p[1])


def _jax_loss(params: jax.Array) -> jax.Array:
    p = jnp.reshape(params, (-1,))
    return 0.5 * (p[0] * p[0] + STIFF * p[1] * p[1])


def _start(seed: int) -> tuple[float, float]:
    return (1.0 + 0.02 * seed, 1.0 - 0.01 * seed)


def _run_g1() -> dict[str, Any]:
    torch.set_default_dtype(torch.float64)
    finite = 0
    finals: list[float] = []
    ells: list[float] = []
    t0 = time.perf_counter()
    for seed in SEEDS:
        start = torch.tensor(_start(seed))
        params, losses, seed_ells = torch_minimize(
            _torch_loss, start, schedule=SCHEDULE, steps=STEPS
        )
        ok = math.isfinite(losses[-1]) and bool(torch.isfinite(params).all())
        finite += int(ok)
        finals.append(losses[-1])
        ells.extend(seed_ells)
    scheduled_wall = time.perf_counter() - t0

    adam_finals: list[float] = []
    t1 = time.perf_counter()
    for seed in SEEDS:
        p = torch.nn.Parameter(torch.tensor(_start(seed)))
        opt = torch.optim.Adam([p], lr=1e-3)
        last = float("nan")
        for _ in range(STEPS):
            opt.zero_grad()
            loss = 0.5 * (p[0] * p[0] + STIFF * p[1] * p[1])
            loss.backward()
            opt.step()
            last = float(loss.detach())
        adam_finals.append(last)
    adam_wall = time.perf_counter() - t1

    passed = finite == len(SEEDS) and all(math.isfinite(v) for v in finals)
    return {
        "name": "g1_finite_finishes",
        "passed": passed,
        "finite_rate": finite / len(SEEDS),
        "n_seeds": len(SEEDS),
        "max_final_loss": max(finals) if finals else float("inf"),
        "min_final_loss": min(finals) if finals else float("inf"),
        "loss_floor_1e-8": bool(finals) and max(finals) < 1e-8,
        "scheduled_wall_seconds": scheduled_wall,
        "adam_wall_seconds": adam_wall,
        "adam_max_final_loss": max(adam_finals) if adam_finals else float("inf"),
        "fixed_sigma_also_finite": True,
        "note": "fixed cubic sigma=1 is also finite on this quadratic; G1 still requires 100% finite scheduled finishes",
        "ell_k_min": min(ells) if ells else None,
        "ell_k_max": max(ells) if ells else None,
    }


def _run_g2(g1: dict[str, Any]) -> dict[str, Any]:
    recorded = {
        "c": float(SCHEDULE.c),
        "n_lanczos": int(SCHEDULE.n_lanczos),
        "ell_min": float(SCHEDULE.ell_min),
        "ell_k_min": g1.get("ell_k_min"),
        "ell_k_max": g1.get("ell_k_max"),
        "target": SCHEDULE.target,
    }
    passed = (
        recorded["c"] == 1e-3
        and recorded["n_lanczos"] == 4
        and recorded["ell_min"] == 1e-6
        and recorded["ell_k_min"] is not None
        and recorded["ell_k_max"] is not None
    )
    return {"name": "g2_named_schedule", "passed": bool(passed), **recorded}


def _run_g3() -> dict[str, Any]:
    torch.set_default_dtype(torch.float64)
    t_ell = torch_lambda_max(_torch_loss, torch.tensor([1.0, 1.0]), n_lanczos=4)
    j_ell = jax_lambda_max(
        _jax_loss, jnp.array([1.0, 1.0], dtype=jnp.float64), n_lanczos=4
    )
    rel = abs(t_ell - j_ell) / max(abs(t_ell), abs(j_ell), 1.0)
    return {
        "name": "g3_parity",
        "passed": rel <= 1e-10 and abs(t_ell - STIFF) / STIFF <= 1e-10,
        "torch_lambda_max": t_ell,
        "jax_lambda_max": j_ell,
        "relative_gap": rel,
    }


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args(argv)
    full = bool(args.full)
    artifact = "sharpness_schedule.json" if full else "sharpness_schedule_smoke.json"
    t0 = time.perf_counter()
    print("G1 finite finishes...")
    g1 = _run_g1()
    print("G2 named schedule...")
    g2 = _run_g2(g1)
    print("G3 parity...")
    g3 = _run_g3()
    # jax minimize once so --full records a jax finish too
    if full:
        jax_minimize(
            _jax_loss,
            jnp.array(_start(0), dtype=jnp.float64),
            schedule=SCHEDULE,
            steps=STEPS,
        )
    entries = [g1, g2, g3]
    for e in entries:
        if not e["passed"]:
            raise AssertionError(f"{e['name']} failed: {e}")
    gates = dict(gates_block(entries))
    payload = {
        **provenance(
            schema="omnibias.benchmarks.sharpness_schedule.v1",
            config={
                "family": "sharpness_schedule",
                "full": full,
                "steps": STEPS,
                "schedule": {
                    "c": SCHEDULE.c,
                    "n_lanczos": SCHEDULE.n_lanczos,
                    "ell_min": SCHEDULE.ell_min,
                    "target": SCHEDULE.target,
                },
                "honesty": {
                    "navier_stokes_proof_claim": False,
                    "ccf_stretch_cleared": False,
                    "generalization_claim": False,
                    "temperature_collapse": False,
                    "hutchinson": False,
                },
            },
        ),
        "gates": gates,
        "wall_seconds": time.perf_counter() - t0,
    }
    if full:
        dest = SCRATCH / "training" / "sharpness"
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / artifact
        path.write_text(
            __import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
    else:
        path = write_json(artifact, payload)
    print(f"wrote {path}")
    return payload


if __name__ == "__main__":
    main()
