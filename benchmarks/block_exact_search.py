# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Wave-3 trainer: block / coordinate exact search (theory 08-07).

Smoke earns G1 (last-layer LS vs 20 Adam on the same block, five
seeds, skill vs the zero predictor), G2 (section-5 quadratic ``s*=1``),
G3 (never-worse with ``verify=True``), and G4 (torch/jax G2 parity).
A special case of 03-12, not a global solver and not CCF stretch.
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

sys.path.insert(0, os.path.dirname(__file__))
from _common import provenance, write_json  # type: ignore[import-not-found]  # noqa: E402
from _gates import gates_block  # type: ignore[import-not-found]  # noqa: E402
from omnibias.jax.optim_block_search import (
    block_exact_search as jax_search,
)
from omnibias.torch.optim_block_search import (
    block_exact_search as torch_search,
)
from omnibias.torch.optim_block_search import last_linear_block

jax.config.update("jax_enable_x64", True)

SCRATCH = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))
SEEDS = (0, 1, 2, 3, 4)


def _section5_torch(params: torch.Tensor) -> torch.Tensor:
    p = params.reshape(-1)
    return (p[0] * 1.0 + p[1] * 0.5 - 1.0) ** 2


def _section5_jax(params: jax.Array) -> jax.Array:
    p = jnp.reshape(params, (-1,))
    return (p[0] * 1.0 + p[1] * 0.5 - 1.0) ** 2


def _run_g1() -> dict[str, Any]:
    torch.set_default_dtype(torch.float64)
    wins = 0
    skills = 0
    block_losses: list[float] = []
    adam_losses: list[float] = []
    t0 = time.perf_counter()
    for seed in SEEDS:
        g = torch.Generator().manual_seed(seed)
        hidden = torch.randn(12, 3, generator=g)
        v_true = torch.tensor([0.4, -0.3, 0.7])
        target = hidden @ v_true

        def loss(params: torch.Tensor, h: torch.Tensor = hidden, y: torch.Tensor = target) -> torch.Tensor:
            r = h @ params.reshape(-1) - y
            return 0.5 * torch.dot(r, r)

        v0 = torch.zeros(3)
        l0 = float(loss(v0))
        new, _report = torch_search(
            loss, v0, spec=last_linear_block(3), exact_quadratic=True
        )
        block_loss = float(loss(new))
        p = torch.nn.Parameter(v0.clone())
        opt = torch.optim.Adam([p], lr=1e-3)
        for _ in range(20):
            opt.zero_grad()
            value = loss(p)
            value.backward()
            opt.step()
        adam_loss = float(loss(p.detach()))
        wins += int(block_loss <= adam_loss + 1e-12)
        skills += int(block_loss < l0)
        block_losses.append(block_loss)
        adam_losses.append(adam_loss)
    wall = time.perf_counter() - t0
    return {
        "name": "g1_block_win",
        "passed": wins == len(SEEDS) and skills == len(SEEDS),
        "win_rate_vs_adam20": wins / len(SEEDS),
        "skill_vs_zero": skills / len(SEEDS),
        "max_block_loss": max(block_losses),
        "max_adam_loss": max(adam_losses),
        "wall_seconds": wall,
        "note": "last-layer least squares with frozen hidden; one block step vs Adam lr=1e-3",
    }


def _run_g2() -> dict[str, Any]:
    torch.set_default_dtype(torch.float64)
    _new, result = torch_search(
        _section5_torch,
        torch.tensor([0.0, 0.0]),
        mask=(True, False),
        exact_quadratic=True,
    )
    passed = (
        result.fell_back is False
        and abs(result.step - 1.0) <= 1e-12
        and result.actual_value is not None
        and abs(result.actual_value) <= 1e-12
    )
    return {
        "name": "g2_exact_quadratic",
        "passed": bool(passed),
        "step": result.step,
        "fell_back": result.fell_back,
        "actual_value": result.actual_value,
    }


def _run_g3() -> dict[str, Any]:
    torch.set_default_dtype(torch.float64)
    never_worse = 0
    n = 20
    for seed in range(n):
        g = torch.Generator().manual_seed(seed + 11)
        q = torch.randn(4, generator=g)
        diag = torch.tensor([1.0, 4.0, 0.5, 2.0])

        def loss(params: torch.Tensor, scale: torch.Tensor = diag) -> torch.Tensor:
            p = params.reshape(-1)
            return 0.5 * torch.dot(p * scale, p * scale)

        mask = [(int(torch.randint(0, 2, (1,), generator=g)) == 1) for _ in range(4)]
        if not any(mask):
            mask[0] = True
        l0 = float(loss(q))
        new, _result = torch_search(loss, q, mask=mask, exact_quadratic=True)
        never_worse += int(float(loss(new)) <= l0 + 1e-12)
    return {
        "name": "g3_never_worse",
        "passed": never_worse == n,
        "rate": never_worse / n,
        "n": n,
    }


def _run_g4() -> dict[str, Any]:
    torch.set_default_dtype(torch.float64)
    _t_new, t_res = torch_search(
        _section5_torch, torch.tensor([0.0, 0.0]), mask=(True, False), exact_quadratic=True
    )
    _j_new, j_res = jax_search(
        _section5_jax,
        jnp.array([0.0, 0.0], dtype=jnp.float64),
        mask=(True, False),
        exact_quadratic=True,
    )
    gap = abs(t_res.step - j_res.step)
    return {
        "name": "g4_parity",
        "passed": gap <= 1e-12 and abs(t_res.step - 1.0) <= 1e-12,
        "step_gap": gap,
        "torch_step": t_res.step,
        "jax_step": j_res.step,
    }


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args(argv)
    full = bool(args.full)
    artifact = "block_exact_search.json" if full else "block_exact_search_smoke.json"
    t0 = time.perf_counter()
    print("G1 block win...")
    g1 = _run_g1()
    print("G2 exact quadratic...")
    g2 = _run_g2()
    print("G3 never-worse...")
    g3 = _run_g3()
    print("G4 parity...")
    g4 = _run_g4()
    entries = [g1, g2, g3, g4]
    for e in entries:
        if not e["passed"]:
            raise AssertionError(f"{e['name']} failed: {e}")
    payload = {
        **provenance(
            schema="omnibias.benchmarks.block_exact_search.v1",
            config={
                "family": "block_exact_search",
                "full": full,
                "honesty": {
                    "navier_stokes_proof_claim": False,
                    "ccf_stretch_cleared": False,
                    "global_solver_claim": False,
                    "temperature_collapse": False,
                },
            },
        ),
        "gates": dict(gates_block(entries)),
        "wall_seconds": time.perf_counter() - t0,
    }
    if full:
        dest = SCRATCH / "training" / "block_search"
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
