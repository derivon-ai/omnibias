# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Wave-3 trainer: depth-causal residual (theory 08-05).

Smoke earns G1 (named 1-D Poisson, five seeds: depth-causal mean residual
at most last-layer-only, skill vs ``u = 0``, named MSE below the zero
field), G2 (honesty keys sealed), and G3 (torch/jax parity on a tiny
Poisson). Local GN is greedy, not a global min, and not CCF stretch.
Bias collapse (``delta -> 0``) supplies the jet. This is not time
marching (``omnibias.pinn.train.march``) and not the 08-03 proxy residual.
Hilbert is out of scope.
"""

from __future__ import annotations

import argparse
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from _common import provenance, write_json  # type: ignore[import-not-found]  # noqa: E402
from _gates import gates_block, skill_score  # type: ignore[import-not-found]  # noqa: E402
from omnibias.pinn.train import DepthResidualConfig, DepthResidualForbidden, honesty_payload
from omnibias.pinn.train.jax.depth_residual import (
    depth_residual_sweep as jax_sweep,
)
from omnibias.pinn.train.jax.depth_residual import (
    poisson_1d_residual as jax_poisson,
)
from omnibias.pinn.train.torch.depth_residual import (
    depth_residual_sweep as torch_sweep,
)
from omnibias.pinn.train.torch.depth_residual import (
    field_tower as torch_field_tower,
)
from omnibias.pinn.train.torch.depth_residual import (
    poisson_1d_residual as torch_poisson,
)

jax.config.update("jax_enable_x64", True)

SCRATCH = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))
SEEDS = (0, 1, 2, 3, 4)
WIDTHS = (8, 2)
N_COLLOC = 21
STEPS = 10
DAMPING = 1e-3
PDE_MSE_MAX = 50.0


def _collocation() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    xs = torch.linspace(0.0, 1.0, N_COLLOC + 2)[1:-1]
    u_star = torch.sin(math.pi * xs)
    force = (math.pi**2) * u_star
    return xs, u_star, force


def _init(seed: int) -> tuple[list[Any], list[Any]]:
    g = torch.Generator().manual_seed(seed)
    w1, w2 = WIDTHS
    layers = [
        (1.0 * torch.randn(w1, 1, generator=g), 0.3 * torch.randn(w1, generator=g), "tanh"),
        (0.3 * torch.randn(w2, w1, generator=g), 0.3 * torch.randn(w2, generator=g), "tanh"),
    ]
    decodes = [
        (1.0 * torch.randn(1, w1, generator=g), torch.zeros(1)),
        (0.5 * torch.randn(1, w2, generator=g), torch.zeros(1)),
    ]
    return layers, decodes


def _metrics(
    layers: list[Any],
    decodes: list[Any],
    xs: torch.Tensor,
    u_star: torch.Tensor,
    force: torch.Tensor,
    cfg: DepthResidualConfig,
) -> tuple[float, float]:
    tower = torch_field_tower(layers, decodes, xs, config=cfg)
    residual = torch_poisson(force)(tower)
    field = tower[0]
    mse = float(torch.mean(residual**2))
    skill = skill_score(field.detach().cpu().numpy(), u_star.detach().cpu().numpy())
    return mse, skill


def _run_g1() -> dict[str, Any]:
    torch.set_default_dtype(torch.float64)
    xs, u_star, force = _collocation()
    zero_mse = float(torch.mean(force**2))
    pde = torch_poisson(force)
    cfg_d = DepthResidualConfig(
        n_directions=1, jet_order=2, damping=DAMPING, steps_per_layer=STEPS
    )
    cfg_l = DepthResidualConfig(
        n_directions=1,
        jet_order=2,
        damping=DAMPING,
        steps_per_layer=STEPS,
        last_layer_only=True,
    )
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        layers, decodes = _init(seed)
        ld, dd, report_d = torch_sweep(layers, decodes, pde, xs, config=cfg_d)
        ll, dl, report_l = torch_sweep(layers, decodes, pde, xs, config=cfg_l)
        md, sd = _metrics(ld, dd, xs, u_star, force, cfg_d)
        ml, sl = _metrics(ll, dl, xs, u_star, force, cfg_l)
        rows.append(
            {
                "seed": seed,
                "depth_mse": md,
                "last_mse": ml,
                "skill_depth": sd,
                "skill_last": sl,
                "depth_leq_last": md <= ml,
                "n_gn_steps": report_d.n_gn_steps,
            }
        )
        assert report_l.n_gn_steps == report_d.n_gn_steps
    depth_mses = [float(r["depth_mse"]) for r in rows]
    last_mses = [float(r["last_mse"]) for r in rows]
    n_leq = sum(int(r["depth_leq_last"]) for r in rows)
    passed = (
        all(float(r["skill_depth"]) > 0.0 for r in rows)
        and max(depth_mses) < PDE_MSE_MAX
        and zero_mse > PDE_MSE_MAX
        and n_leq >= 2
        and statistics.mean(depth_mses) <= statistics.mean(last_mses)
        and all(r["n_gn_steps"] == 2 * STEPS for r in rows)
    )
    return {
        "name": "g1_last_layer",
        "passed": bool(passed),
        "n_seeds": len(rows),
        "n_depth_leq_last": n_leq,
        "mean_depth_mse": statistics.mean(depth_mses),
        "mean_last_mse": statistics.mean(last_mses),
        "max_depth_mse": max(depth_mses),
        "pde_mse_max": PDE_MSE_MAX,
        "zero_field_mse": zero_mse,
        "gn_steps_per_arm": 2 * STEPS,
        "rows": rows,
        "note": (
            "Named 1-D Poisson -u_xx = pi^2 sin(pi x) on (0, 1) with hard "
            "BC x(1-x) * decode(h). Two tanh hidden layers, widths 8 then 2. "
            f"{STEPS} GN steps per layer; last-layer-only gets the matched "
            "budget on the last block. Skill vs u=0. Not a global min and "
            "not CCF stretch."
        ),
    }


def _run_g2() -> dict[str, Any]:
    payload = honesty_payload()
    cfg = DepthResidualConfig(n_directions=1, jet_order=2)
    torch.set_default_dtype(torch.float64)
    xs = torch.linspace(0.2, 0.8, 5)
    layers = [(0.2 * torch.ones(2, 1), torch.zeros(2), "tanh")]
    decode = (0.1 * torch.ones(1, 2), torch.zeros(1))
    _new, _dec, report = torch_sweep(
        layers,
        decode,
        torch_poisson((math.pi**2) * torch.sin(math.pi * xs)),
        xs,
        config=cfg,
    )
    raised = False
    try:
        torch_sweep(
            [(torch.ones(1, 1), None, None)],
            (torch.ones(1, 1), None),
            torch_poisson(torch.ones(2)),
            torch.tensor([0.3, 0.7]),
            config=DepthResidualConfig(n_directions=2, jet_order=2),
        )
    except DepthResidualForbidden:
        raised = True
    passed = (
        payload == {
            "navier_stokes_proof_claim": False,
            "stretch_1e-13_cleared": False,
            "hilbert_not_in_scope": True,
            "greedy_only_claimed_optimal": False,
        }
        and report.navier_stokes_proof_claim is False
        and report.stretch_cleared is False
        and report.hilbert_not_in_scope is True
        and report.greedy_only_claimed_optimal is False
        and raised
    )
    return {
        "name": "g2_honesty",
        "passed": bool(passed),
        "honesty": payload,
        "flood_raises": raised,
        "note": (
            "Artifact honesty keys are sealed. Hilbert is out of scope. "
            "n_directions >= n_params raises DepthResidualForbidden."
        ),
    }


def _run_g3() -> dict[str, Any]:
    torch.set_default_dtype(torch.float64)
    xs_np = np.linspace(0.15, 0.85, 5)
    w = np.array([[0.4], [-0.25]])
    b = np.array([0.1, -0.05])
    v = np.array([[0.3, -0.2]])
    c = np.array([0.0])
    cfg = DepthResidualConfig(n_directions=1, jet_order=2, damping=1e-4)
    force_np = (np.pi**2) * np.sin(np.pi * xs_np)
    t_new, t_dec, t_rep = torch_sweep(
        [(torch.tensor(w), torch.tensor(b), "tanh")],
        (torch.tensor(v), torch.tensor(c)),
        torch_poisson(torch.tensor(force_np)),
        torch.tensor(xs_np),
        config=cfg,
    )
    j_new, j_dec, j_rep = jax_sweep(
        [(jnp.asarray(w), jnp.asarray(b), "tanh")],
        (jnp.asarray(v), jnp.asarray(c)),
        jax_poisson(jnp.asarray(force_np)),
        jnp.asarray(xs_np),
        config=cfg,
    )
    gap_w = abs(float(t_new[0][0][0, 0]) - float(j_new[0][0][0, 0].reshape(-1)[0]))
    gap_v = abs(float(t_dec[0][0][0, 0]) - float(j_dec[0][0][0, 0].reshape(-1)[0]))
    gap_r = abs(t_rep.residual_norms[-1] - j_rep.residual_norms[-1])
    passed = gap_w <= 1e-10 and gap_v <= 1e-10 and gap_r <= 1e-10
    return {
        "name": "g3_parity",
        "passed": bool(passed),
        "w_gap": gap_w,
        "v_gap": gap_v,
        "residual_gap": gap_r,
    }


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args(argv)
    full = bool(args.full)
    artifact = (
        "depth_causal_residual.json" if full else "depth_causal_residual_smoke.json"
    )
    t0 = time.perf_counter()
    print("G1 last-layer comparison...")
    g1 = _run_g1()
    print("G2 honesty...")
    g2 = _run_g2()
    print("G3 parity...")
    g3 = _run_g3()
    entries = [g1, g2, g3]
    for e in entries:
        if not e["passed"]:
            raise AssertionError(f"{e['name']} failed: {e}")
    payload = {
        **provenance(
            schema="omnibias.benchmarks.depth_causal_residual.v1",
            config={
                "family": "depth_causal_residual",
                "full": full,
                "honesty": honesty_payload(),
            },
        ),
        "gates": dict(gates_block(entries)),
        "wall_seconds": time.perf_counter() - t0,
    }
    if full:
        dest = SCRATCH / "training" / "depth_residual"
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
