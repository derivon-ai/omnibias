# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Jet vs nested AD + GN vs Adam on 1-D Poisson (theory 06-05 obligation 3).

Local PDE only. Not CCF, not Group 09, not ImageNet, not a package extract
and not a paper. ``delta -> 0`` (founding bias collapse) supplies
``sigma^(n)`` inside ``mlp_jet``; this is not temperature collapse.

(a) Closed-form ``mlp_jet`` residual vs nested ``autograd``: agreement
plus wall at a cost-sensitive order (order 6). Order 2 is reported and
often loses to AD on a small net.
(b) Exact-J ``GaussNewton`` vs named Adam on the same jet residual,
multi-seed.

    uv run python benchmarks/jet_vs_nested_ad.py
    uv run python benchmarks/jet_vs_nested_ad.py --full
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from _common import (  # type: ignore[import-not-found]  # noqa: E402
    median_time_ms,
    provenance,
    rss_mb,
    write_json,
)
from _gates import gates_block, require_cost_parity  # type: ignore[import-not-found]  # noqa: E402
from omnibias.torch.jet import jet_to_tower, mlp_jet  # noqa: E402
from omnibias.torch.optim import GaussNewton, functional_residual_fn  # noqa: E402

SCRATCH = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))

torch.set_default_dtype(torch.float64)

BC_WEIGHT = 20.0
HIDDEN = 8
N_INT = 32
COST_POINTS = 2048
COST_ORDER = 6
COST_DIMS = (1, 16, 16, 1)
ADAM_LR = 1e-2
BASELINE_NAME = "nested torch.autograd + Adam"


class JetPoisson(nn.Module):
    """One hidden tanh layer; Laplacian from ``mlp_jet``, not a hand-rolled ``sigma''``."""

    def __init__(self, hidden: int = HIDDEN) -> None:
        super().__init__()
        self.W = nn.Parameter(torch.randn(hidden, 1) * 0.5)
        self.beta = nn.Parameter(torch.randn(hidden) * 0.1)
        self.c = nn.Parameter(torch.randn(hidden) * 0.1)
        self.b = nn.Parameter(torch.zeros(1))

    def layers(self) -> tuple[tuple[torch.Tensor, torch.Tensor, str | None], ...]:
        return (
            (self.W, self.beta, "tanh"),
            (self.c.reshape(1, -1), self.b, None),
        )

    def tower(self, x: torch.Tensor, order: int) -> torch.Tensor:
        return jet_to_tower(mlp_jet(x, torch.ones_like(x), self.layers(), order))

    def value(self, x: torch.Tensor) -> torch.Tensor:
        z = x @ self.W.T + self.beta
        return (torch.tanh(z) * self.c).sum(-1) + self.b.reshape(())

    def nested_uxx(self, x: torch.Tensor) -> torch.Tensor:
        xx = x.detach().clone().requires_grad_(True)
        u = self.value(xx)
        du = torch.autograd.grad(u.sum(), xx, create_graph=True)[0]
        d2 = torch.autograd.grad(du.sum(), xx, create_graph=True)[0]
        return d2.reshape(-1)

    def forward(self, x_int: torch.Tensor, x_bc: torch.Tensor, f_int: torch.Tensor) -> torch.Tensor:
        pde = -self.tower(x_int, 2)[2].reshape(-1) - f_int
        bc = BC_WEIGHT * self.tower(x_bc, 0)[0].reshape(-1)
        return torch.cat([pde, bc], dim=0)


class DeepTanh(nn.Module):
    """Two hidden tanh layers for the high-order wall comparison."""

    def __init__(self, dims: tuple[int, ...] = COST_DIMS) -> None:
        super().__init__()
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        for din, dout in zip(dims[:-1], dims[1:], strict=True):
            self.weights.append(nn.Parameter(torch.randn(dout, din) * 0.4))
            self.biases.append(nn.Parameter(torch.randn(dout) * 0.1))

    def layers(self) -> tuple[tuple[torch.Tensor, torch.Tensor, str | None], ...]:
        last = len(self.weights) - 1
        return tuple(
            (w, b, None if i == last else "tanh")
            for i, (w, b) in enumerate(zip(self.weights, self.biases, strict=True))
        )

    def value(self, x: torch.Tensor) -> torch.Tensor:
        z = x
        last = len(self.weights) - 1
        for i, (w, b) in enumerate(zip(self.weights, self.biases, strict=True)):
            z = z @ w.T + b
            if i < last:
                z = torch.tanh(z)
        return z.reshape(-1)

    def jet_deriv(self, x: torch.Tensor, order: int) -> torch.Tensor:
        return jet_to_tower(mlp_jet(x, torch.ones_like(x), self.layers(), order))[order].reshape(-1)

    def nested_deriv(self, x: torch.Tensor, order: int) -> torch.Tensor:
        t = x.detach().clone().requires_grad_(True)
        u = self.value(t)
        g = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
        for _ in range(int(order) - 1):
            g = torch.autograd.grad(g.sum(), t, create_graph=True)[0]
        return g.reshape(-1)


def _points(n_int: int = N_INT) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_int = torch.linspace(0.0, 1.0, n_int + 2)[1:-1].reshape(-1, 1)
    x_bc = torch.tensor([[0.0], [1.0]])
    f_int = (math.pi**2) * torch.sin(math.pi * x_int).reshape(-1)
    return x_int, x_bc, f_int


def _rel_l2(model: JetPoisson, n: int = 200) -> float:
    xs = torch.linspace(0.0, 1.0, n).unsqueeze(-1)
    with torch.no_grad():
        pred = model.value(xs)
        exact = torch.sin(math.pi * xs.squeeze(-1))
        return float(torch.linalg.norm(pred - exact) / torch.linalg.norm(exact))


def _run_agreement() -> dict[str, Any]:
    torch.manual_seed(0)
    model = JetPoisson()
    x_int, x_bc, f_int = _points()
    r_jet = model(x_int, x_bc, f_int).detach()
    uxx_ad = model.nested_uxx(x_int).detach()
    pde_ad = -uxx_ad - f_int
    u_bc = model.value(x_bc).detach()
    r_ad = torch.cat([pde_ad, BC_WEIGHT * u_bc], dim=0)
    err = float((r_jet - r_ad).abs().max())
    if err > 1e-12:
        raise AssertionError(f"jet_vs_ad_residual: max_abs={err:.4e} exceeds 1e-12")
    return {
        "name": "jet_ad_agreement",
        "passed": err <= 1e-12,
        "max_abs": err,
        "max_abs_cap": 1e-12,
        "rss_mb": rss_mb(),
    }


def _time_deriv(model: DeepTanh, x: torch.Tensor, order: int, *, jet: bool) -> float:
    if jet:

        def fn() -> torch.Tensor:
            return model.jet_deriv(x, order)
    else:

        def fn() -> torch.Tensor:
            return model.nested_deriv(x, order)

    return median_time_ms(fn, warmup=1, repeats=3)


def _run_wall() -> dict[str, Any]:
    torch.manual_seed(1)
    model = DeepTanh()
    x = torch.linspace(0.0, 1.0, COST_POINTS).reshape(-1, 1)
    # Order-2 is the Poisson residual order; often AD wins on a small net.
    t_jet2 = _time_deriv(model, x, 2, jet=True)
    t_ad2 = _time_deriv(model, x, 2, jet=False)
    t_jet = _time_deriv(model, x, COST_ORDER, jet=True)
    t_ad = _time_deriv(model, x, COST_ORDER, jet=False)
    agree = float(
        (model.jet_deriv(x, COST_ORDER) - model.nested_deriv(x, COST_ORDER)).detach().abs().max()
    )
    cost = require_cost_parity(t_jet, t_ad, max_ratio=0.5, name="jet_vs_ad_order6")
    return {
        "name": "jet_wall_order6",
        "passed": bool(cost["passed"]) and agree <= 1e-10,
        "order": COST_ORDER,
        "n_points": COST_POINTS,
        "jet_ms": t_jet,
        "nested_ad_ms": t_ad,
        "speedup": float(t_ad / t_jet) if t_jet > 0.0 else float("inf"),
        "order2_jet_ms": t_jet2,
        "order2_nested_ad_ms": t_ad2,
        "order2_speedup": float(t_ad2 / t_jet2) if t_jet2 > 0.0 else float("inf"),
        "order2_note": "order-2 mlp_jet often loses to AD on a small net; gated order is 6",
        "max_abs_order6": agree,
        "rss_mb": rss_mb(),
    }


def _run_adam(seed: int, *, steps: int) -> dict[str, float]:
    torch.manual_seed(seed)
    model = JetPoisson()
    x_int, x_bc, f_int = _points()
    opt = torch.optim.Adam(model.parameters(), lr=ADAM_LR)
    loss = torch.tensor(0.0)
    t0 = time.perf_counter()
    for _ in range(int(steps)):
        opt.zero_grad(set_to_none=True)
        r = model(x_int, x_bc, f_int)
        loss = 0.5 * (r * r).mean()
        loss.backward()
        opt.step()
    return {
        "wall_s": time.perf_counter() - t0,
        "rel_l2": _rel_l2(model),
        "final_loss": float(loss.detach()),
    }


def _run_gn(seed: int, *, steps: int) -> dict[str, float]:
    torch.manual_seed(seed)
    model = JetPoisson()
    x_int, x_bc, f_int = _points()
    flat0, residual_fn = functional_residual_fn(model, x_int, x_bc, f_int)
    opt = GaussNewton(damping=1e-3, solver="qr", damping_strategy="nielsen")
    t0 = time.perf_counter()
    theta, hist = opt.minimize(residual_fn, flat0, steps=int(steps))
    torch.nn.utils.vector_to_parameters(theta.detach(), model.parameters())
    return {
        "wall_s": time.perf_counter() - t0,
        "rel_l2": _rel_l2(model),
        "final_loss": float(hist[-1]) if hist else float("nan"),
    }


def _run_gn_vs_adam(*, n_seeds: int, adam_steps: int, gn_steps: int) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for seed in range(int(n_seeds)):
        adam = _run_adam(seed, steps=adam_steps)
        gn = _run_gn(seed, steps=gn_steps)
        rows.append(
            {
                "seed": seed,
                "adam_rel_l2": adam["rel_l2"],
                "gn_rel_l2": gn["rel_l2"],
                "adam_loss": adam["final_loss"],
                "gn_loss": gn["final_loss"],
                "adam_wall_s": adam["wall_s"],
                "gn_wall_s": gn["wall_s"],
                "gn_better": gn["rel_l2"] < adam["rel_l2"],
            }
        )
    wins = sum(1 for r in rows if r["gn_better"])
    passed = wins == len(rows) and len(rows) > 0
    return {
        "name": "gn_beats_adam",
        "passed": passed,
        "n_seeds": len(rows),
        "wins": wins,
        "median_adam_rel_l2": float(torch.median(torch.tensor([r["adam_rel_l2"] for r in rows]))),
        "median_gn_rel_l2": float(torch.median(torch.tensor([r["gn_rel_l2"] for r in rows]))),
        "adam_steps": adam_steps,
        "gn_steps": gn_steps,
        "per_seed": rows,
        "baseline": BASELINE_NAME,
    }


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args(argv)
    full = bool(args.full)
    artifact = "jet_vs_nested_ad.json" if full else "jet_vs_nested_ad_smoke.json"
    t0 = time.perf_counter()
    g1 = _run_agreement()
    g2 = _run_wall()
    g3 = _run_gn_vs_adam(
        n_seeds=5 if full else 3,
        adam_steps=120 if full else 80,
        gn_steps=16 if full else 12,
    )
    entries = [g1, g2, g3]
    for e in entries:
        print(e["name"], "ok" if e["passed"] else "FAIL")
        if not e["passed"]:
            raise AssertionError(f"{e['name']} failed: {e}")
    payload = {
        **provenance(
            schema="omnibias.benchmarks.jet_vs_nested_ad.v1",
            config={
                "family": "jet_vs_nested_ad",
                "full": full,
                "pde": "1d_poisson",
                "honesty": {
                    "ccf_showcase": False,
                    "group_09_vehicle": False,
                    "package_extract": False,
                    "paper": False,
                    "founding_bias_collapse": True,
                    "temperature_collapse": False,
                    "public_primitive": "omnibias.torch.jet.mlp_jet",
                },
            },
        ),
        "baseline": {"name": BASELINE_NAME},
        "seeds": list(range(int(g3["n_seeds"]))),
        "gates": dict(gates_block(entries)),
        "wall_seconds": time.perf_counter() - t0,
    }
    if full:
        dest = SCRATCH / "citation" / "jet_vs_ad"
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / artifact
        path.write_text(__import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8")
    else:
        path = write_json(artifact, payload)
    print(f"wrote {path}")
    return payload


if __name__ == "__main__":
    main()
