# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Wave-3 trainer: depth-causal local jet (theory 08-03).

Smoke earns G1 (full-parameter flood raises ``LocalJetForbidden``),
G2 (1-D Poisson warm-start + 50 Gauss–Newton, five seeds, skill vs
``u = 0``), G3 (greedy-only is not claimed optimal), and G4
(torch/jax parity on the spec §5 two-layer toy). Local GN is greedy,
not a global min, and not CCF stretch. Bias collapse (``delta -> 0``)
supplies the jet. This is not time marching
(``omnibias.pinn.train``).
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
from _gates import gates_block, skill_score  # type: ignore[import-not-found]  # noqa: E402
from omnibias.core.local_jet import LocalJetConfig, LocalJetForbidden
from omnibias.jax.train_local import local_jet_step as jax_step
from omnibias.torch.activations.registry import get_activation
from omnibias.torch.optim import GaussNewton
from omnibias.torch.train_local import local_jet_step as torch_step

jax.config.update("jax_enable_x64", True)

SCRATCH = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))
SEEDS = (0, 1, 2, 3, 4)
HIDDEN = 8
N_COLLOC = 21
INIT_SCALE = 2.0
GN_STEPS = 50
PDE_MSE_MAX = 1e-3


def _section5_torch() -> tuple[list[Any], torch.Tensor, torch.Tensor]:
    layers = [
        (torch.tensor([[0.5]]), None, "tanh"),
        (torch.tensor([[0.2]]), None, None),
    ]
    return layers, torch.tensor([1.0]), torch.tensor([1.0])


def _run_g1() -> dict[str, Any]:
    torch.set_default_dtype(torch.float64)
    layers, x, y = _section5_torch()
    raised = False
    try:
        torch_step(
            layers,
            x,
            config=LocalJetConfig(n_directions=2, damping=0.0),
            target=y,
        )
    except LocalJetForbidden as exc:
        raised = "allow_full" in str(exc)
    return {
        "name": "g1_forbid_flood",
        "passed": bool(raised),
        "note": "n_directions >= n_params without allow_full raises LocalJetForbidden",
    }


def _poisson_target() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    xs = torch.linspace(-1.0, 1.0, N_COLLOC)
    u_star = torch.sin(math.pi * xs)
    s = 1.0 - xs**2
    n_star = torch.where(
        s > 1e-12,
        u_star / s,
        (math.pi / 2.0) * torch.sign(xs),
    )
    return xs, u_star, n_star


def _pde_residual_fn(xs: torch.Tensor, force: torch.Tensor, width: int):
    spec = get_activation("tanh")
    fp = spec.fastpath
    if fp is None:
        raise RuntimeError("tanh fastpath is required for the named Poisson residual")

    def pde_res(
        params: torch.Tensor,
        xs: torch.Tensor = xs,
        force: torch.Tensor = force,
        width: int = width,
        deriv2=fp,
    ) -> torch.Tensor:
        W = params[:width].reshape(width, 1)
        b = params[width : 2 * width]
        v = params[2 * width :]
        z = xs.unsqueeze(-1) * W.reshape(1, width) + b
        phi = torch.tanh(z)
        phip = deriv2(z, 1) * W.reshape(1, width)
        phipp = deriv2(z, 2) * (W.reshape(1, width) ** 2)
        N = phi @ v
        Np = phip @ v
        Npp = phipp @ v
        s = 1.0 - xs**2
        uxx = (-2.0) * N + 2.0 * (-2.0 * xs) * Np + s * Npp
        return uxx + force

    return pde_res


def _field(
    params: torch.Tensor, xs: torch.Tensor, width: int
) -> torch.Tensor:
    W = params[:width].reshape(width, 1)
    b = params[width : 2 * width]
    v = params[2 * width :]
    h = torch.tanh(xs.unsqueeze(-1) @ W.T + b)
    return (1.0 - xs**2) * (h @ v)


def _run_g2(*, full: bool) -> dict[str, Any]:
    torch.set_default_dtype(torch.float64)
    xs, u_star, n_star = _poisson_target()
    x = xs.unsqueeze(-1)
    force = (math.pi**2) * u_star
    pde_res = _pde_residual_fn(xs, force, HIDDEN)
    cfg = LocalJetConfig(n_directions=1, jet_order=1, damping=1e-8, variant="readout")
    seeds = SEEDS if full else SEEDS
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        g = torch.Generator().manual_seed(seed)
        W = INIT_SCALE * torch.randn(HIDDEN, 1, generator=g)
        b = INIT_SCALE * torch.randn(HIDDEN, generator=g)
        v = INIT_SCALE * torch.randn(HIDDEN, generator=g)
        p0 = torch.cat([W.reshape(-1), b, v])
        layers0 = [(W, b, "tanh"), (v.reshape(1, HIDDEN), None, None)]
        layers1, report = torch_step(layers0, x, config=cfg, target=n_star)
        W1, b1, _spec = layers1[0]
        v1 = layers1[1][0].reshape(-1)
        p1 = torch.cat([W1.reshape(-1), b1, v1])
        greedy_mse = float(torch.mean(pde_res(p1) ** 2))
        opt_w = GaussNewton(damping=1e-3, solver="qr")
        pw, hist_w = opt_w.minimize(pde_res, p1.clone(), steps=GN_STEPS)
        opt_c = GaussNewton(damping=1e-3, solver="qr")
        pc, hist_c = opt_c.minimize(pde_res, p0.clone(), steps=GN_STEPS)
        uw = _field(pw, xs, HIDDEN)
        uc = _field(pc, xs, HIDDEN)
        ug = _field(p1, xs, HIDDEN)
        rows.append(
            {
                "seed": seed,
                "warm_mse": hist_w[-1],
                "cold_mse": hist_c[-1],
                "greedy_mse": greedy_mse,
                "skill_warm": skill_score(uw.detach().cpu().numpy(), u_star.detach().cpu().numpy()),
                "skill_cold": skill_score(uc.detach().cpu().numpy(), u_star.detach().cpu().numpy()),
                "skill_greedy": skill_score(ug.detach().cpu().numpy(), u_star.detach().cpu().numpy()),
                "warm_better": hist_w[-1] < hist_c[-1],
                "greedy_only_claimed_optimal": report.greedy_only_claimed_optimal,
            }
        )
    n_better = sum(int(r["warm_better"]) for r in rows)
    max_warm = max(r["warm_mse"] for r in rows)
    max_cold = max(r["cold_mse"] for r in rows)
    passed = (
        n_better >= 1
        and all(r["skill_warm"] > 0.0 and r["skill_cold"] > 0.0 for r in rows)
        and max_warm < PDE_MSE_MAX
        and max_cold < PDE_MSE_MAX
        and all(r["greedy_only_claimed_optimal"] is False for r in rows)
    )
    return {
        "name": "g2_warm_start",
        "passed": bool(passed),
        "n_seeds": len(rows),
        "n_warm_better": n_better,
        "max_warm_mse": max_warm,
        "max_cold_mse": max_cold,
        "pde_mse_max": PDE_MSE_MAX,
        "gn_steps": GN_STEPS,
        "rows": rows,
        "note": (
            "Named 1-D Poisson -u_xx = pi^2 sin(pi x) on [-1,1] with hard "
            "BC (1-x^2) * MLP. Local-jet readout matches u*/(1-x^2), then "
            f"{GN_STEPS} QR GaussNewton. Skill vs u=0. Not a global min "
            "and not CCF stretch."
        ),
    }


def _run_g3(g2: dict[str, Any]) -> dict[str, Any]:
    greedy_mses = [float(r["greedy_mse"]) for r in g2["rows"]]
    warm_mses = [float(r["warm_mse"]) for r in g2["rows"]]
    claimed = any(bool(r["greedy_only_claimed_optimal"]) for r in g2["rows"])
    # Local-only residual is reported and is not the 50-step residual.
    passed = (
        claimed is False
        and all(g > w for g, w in zip(greedy_mses, warm_mses, strict=True))
    )
    return {
        "name": "g3_greedy_honesty",
        "passed": bool(passed),
        "greedy_only_claimed_optimal": False,
        "max_greedy_mse": max(greedy_mses),
        "max_warm_mse": max(warm_mses),
        "note": "A local-only arm is reported and is not written as optimal.",
    }


def _run_g4() -> dict[str, Any]:
    torch.set_default_dtype(torch.float64)
    cfg = LocalJetConfig(n_directions=1, damping=0.0, variant="readout")
    t_layers, t_x, t_y = _section5_torch()
    t_out, t_rep = torch_step(t_layers, t_x, config=cfg, target=t_y)
    j_out, j_rep = jax_step(
        [(jnp.asarray([[0.5]]), None, "tanh"), (jnp.asarray([[0.2]]), None, None)],
        jnp.asarray([1.0]),
        config=cfg,
        target=jnp.asarray([1.0]),
    )
    gap_w = abs(float(t_out[0][0].reshape(-1)[0]) - float(j_out[0][0].reshape(-1)[0]))
    gap_v = abs(float(t_out[1][0].reshape(-1)[0]) - float(j_out[1][0].reshape(-1)[0]))
    yhat = float(t_out[1][0].reshape(-1)[0] * torch.tanh(t_out[0][0].reshape(-1)[0]))
    passed = (
        gap_w <= 1e-12
        and gap_v <= 1e-12
        and abs(yhat - 1.0) <= 1e-12
        and t_rep.greedy_only_claimed_optimal is False
        and j_rep.greedy_only_claimed_optimal is False
    )
    return {
        "name": "g4_parity",
        "passed": bool(passed),
        "w_gap": gap_w,
        "v_gap": gap_v,
        "yhat": yhat,
    }


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args(argv)
    full = bool(args.full)
    artifact = (
        "depth_causal_local_jet.json" if full else "depth_causal_local_jet_smoke.json"
    )
    t0 = time.perf_counter()
    print("G1 forbid flood...")
    g1 = _run_g1()
    print("G2 warm-start...")
    g2 = _run_g2(full=full)
    print("G3 greedy honesty...")
    g3 = _run_g3(g2)
    print("G4 parity...")
    g4 = _run_g4()
    entries = [g1, g2, g3, g4]
    for e in entries:
        if not e["passed"]:
            raise AssertionError(f"{e['name']} failed: {e}")
    payload = {
        **provenance(
            schema="omnibias.benchmarks.depth_causal_local_jet.v1",
            config={
                "family": "depth_causal_local_jet",
                "full": full,
                "honesty": {
                    "navier_stokes_proof_claim": False,
                    "ccf_stretch_cleared": False,
                    "global_min_claim": False,
                    "greedy_only_claimed_optimal": False,
                    "temperature_collapse": False,
                },
            },
        ),
        "gates": dict(gates_block(entries)),
        "wall_seconds": time.perf_counter() - t0,
    }
    if full:
        dest = SCRATCH / "training" / "local_jet"
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
