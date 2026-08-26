# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Wave-3 trainer: implicit / DEQ Newton (theory 08-08).

Smoke earns G1 (IFT VJP vs central FD on the section-5 scalar DEQ,
``||F(u*)|| <= 1e-10``, relative gap ``<= 1e-8``), G2 (named implicit
residual ``u*[:,0] - sin(pi x)`` trained with IFT-GN, five seeds, skill
vs ``u = 0``), G3 (torch/jax parity on G1), and G4 (docs / docstring
name ``lax.while_loop`` vs a Python ``while``). The IFT is the chain
rule at a fixed point, not unrolled BPTT. Not a global min and not CCF
stretch. Bias collapse (``delta -> 0``) supplies ``sigma'``.
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
import torch

sys.path.insert(0, os.path.dirname(__file__))
from _common import provenance, write_json  # type: ignore[import-not-found]  # noqa: E402
from _gates import gates_block, skill_score  # type: ignore[import-not-found]  # noqa: E402
from omnibias.core.implicit import DEQConfig, DEQNotContractive, honesty_payload
from omnibias.jax.implicit import deq_solve as jax_solve
from omnibias.jax.implicit import deq_vjp as jax_vjp
from omnibias.torch.implicit import deq_du_dW, deq_solve, deq_vjp
from omnibias.torch.optim import gauss_newton_direction

jax.config.update("jax_enable_x64", True)

SCRATCH = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))
SEEDS = (0, 1, 2, 3, 4)
N_COLLOC = 21
STEPS = 15
DAMPING = 1e-3
INIT_SCALE = 0.15
MSE_MAX = 0.25
MSE_MEAN_MAX = 0.20


def _run_g1() -> dict[str, Any]:
    torch.set_default_dtype(torch.float64)
    cfg = DEQConfig(solver="newton", tol=1e-14, max_iter=30)
    w = torch.tensor([[0.2]])
    x = torch.tensor([0.5])
    vjp = float(deq_vjp(w, x, "tanh", torch.tensor([1.0]), config=cfg).reshape(-1)[0])
    eps = 1e-6
    up = float(deq_solve(w + eps, x, "tanh", config=cfg).u.reshape(-1)[0])
    um = float(deq_solve(w - eps, x, "tanh", config=cfg).u.reshape(-1)[0])
    fd = (up - um) / (2.0 * eps)
    rel = abs(vjp - fd) / max(abs(fd), 1e-12)
    residual = deq_solve(w, x, "tanh", config=cfg).residual
    passed = rel <= 1e-8 and residual <= 1e-10
    return {
        "name": "g1_ift_fd",
        "passed": bool(passed),
        "vjp": vjp,
        "fd": fd,
        "rel_gap": rel,
        "residual": residual,
        "note": (
            "Section-5 scalar DEQ. IFT uses exact sigma'. "
            "Solver residual must sit below the FD gate."
        ),
    }


def _run_g2() -> dict[str, Any]:
    torch.set_default_dtype(torch.float64)
    xs = torch.linspace(0.0, 1.0, N_COLLOC)
    target = torch.sin(math.pi * xs)
    x_inj = torch.stack([xs, torch.ones_like(xs)], dim=-1)
    cfg = DEQConfig(tol=1e-12, max_iter=30)
    zero = torch.zeros_like(target)
    zero_mse = float(((zero - target) ** 2).mean())
    mses: list[float] = []
    skills: list[float] = []
    for seed in SEEDS:
        gen = torch.Generator().manual_seed(seed)
        w = INIT_SCALE * torch.randn(2, 2, generator=gen)
        sol = deq_solve(w, x_inj, "tanh", config=cfg)
        pred = sol.u[:, 0]
        best = float(((pred - target) ** 2).mean())
        for _ in range(STEPS):
            res = sol.u[:, 0] - target
            jac = deq_du_dW(w, x_inj, "tanh", config=cfg, u=sol.u)
            j = jac[:, 0, :, :].reshape(N_COLLOC, 4)
            trial = w + gauss_newton_direction(j, res, DAMPING).reshape(2, 2)
            trial_sol = deq_solve(trial, x_inj, "tanh", config=cfg)
            trial_pred = trial_sol.u[:, 0]
            trial_mse = float(((trial_pred - target) ** 2).mean())
            if trial_mse <= best:
                w, sol, pred, best = trial, trial_sol, trial_pred, trial_mse
        mses.append(best)
        skills.append(float(skill_score(pred.detach().numpy(), target.numpy())))
    mean_mse = statistics.mean(mses)
    max_mse = max(mses)
    passed = (
        all(s > 0.0 for s in skills)
        and mean_mse <= MSE_MEAN_MAX
        and max_mse <= MSE_MAX
        and zero_mse > MSE_MAX
    )
    payload = honesty_payload()
    raised = False
    try:
        deq_solve(
            torch.tensor([[3.0]]),
            torch.tensor([0.1]),
            "tanh",
            config=DEQConfig(require_contraction=True, max_iter=2),
        )
    except DEQNotContractive:
        raised = True
    passed = passed and raised and payload["unrolled_bptt_claim"] is False
    return {
        "name": "g2_implicit_residual",
        "passed": bool(passed),
        "n_seeds": len(SEEDS),
        "mses": mses,
        "skills": skills,
        "mean_mse": mean_mse,
        "max_mse": max_mse,
        "zero_mse": zero_mse,
        "mse_max": MSE_MAX,
        "contraction_raises": raised,
        "honesty": payload,
        "note": (
            "Named implicit residual u*[:,0] - sin(pi x) on a width-2 "
            "tanh DEQ. IFT-GN with a never-worse accept, not unrolled "
            "BPTT. u=0 fails the MSE cap. require_contraction raises "
            "instead of unrolling."
        ),
    }


def _run_g3() -> dict[str, Any]:
    torch.set_default_dtype(torch.float64)
    cfg = DEQConfig(solver="newton", tol=1e-14, max_iter=30)
    t_res = deq_solve(torch.tensor([[0.2]]), torch.tensor([0.5]), "tanh", config=cfg)
    j_res = jax_solve(jnp.asarray([[0.2]]), jnp.asarray([0.5]), "tanh", config=cfg)
    t_g = float(
        deq_vjp(
            torch.tensor([[0.2]]),
            torch.tensor([0.5]),
            "tanh",
            torch.tensor([1.0]),
            config=cfg,
        ).reshape(-1)[0]
    )
    j_g = float(
        jax_vjp(
            jnp.asarray([[0.2]]),
            jnp.asarray([0.5]),
            "tanh",
            jnp.asarray([1.0]),
            config=cfg,
        ).reshape(-1)[0]
    )
    gap_u = abs(float(t_res.u.reshape(-1)[0]) - float(j_res.u.reshape(-1)[0]))
    gap_g = abs(t_g - j_g)
    passed = gap_u <= 1e-12 and gap_g <= 1e-12
    return {
        "name": "g3_parity",
        "passed": bool(passed),
        "u_gap": gap_u,
        "vjp_gap": gap_g,
        "n_iter_torch": t_res.n_iter,
        "n_iter_jax": j_res.n_iter,
    }


def _run_g4() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    jax_src = (root / "packages/omnibias-jax/src/omnibias/jax/implicit.py").read_text(
        encoding="utf-8"
    )
    torch_src = (
        root / "packages/omnibias-torch/src/omnibias/torch/implicit.py"
    ).read_text(encoding="utf-8")
    api = (root / "docs/api/implicit.md").read_text(encoding="utf-8")
    jax_named = "lax.while_loop" in jax_src and "lax.while_loop" in api
    torch_named = "Python ``while``" in torch_src or "Python `while`" in api
    passed = jax_named and torch_named
    return {
        "name": "g4_tracing",
        "passed": bool(passed),
        "jax_while_loop": "lax.while_loop" in jax_src,
        "docs_while_loop": "lax.while_loop" in api,
        "torch_python_while": "while" in torch_src and "torch.compile" in torch_src,
    }


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args(argv)
    full = bool(args.full)
    artifact = "implicit_deq.json" if full else "implicit_deq_smoke.json"
    t0 = time.perf_counter()
    print("G1 IFT vs FD...")
    g1 = _run_g1()
    print("G2 implicit residual...")
    g2 = _run_g2()
    print("G3 parity...")
    g3 = _run_g3()
    print("G4 tracing note...")
    g4 = _run_g4()
    entries = [g1, g2, g3, g4]
    for e in entries:
        if not e["passed"]:
            raise AssertionError(f"{e['name']} failed: {e}")
    payload = {
        **provenance(
            schema="omnibias.benchmarks.implicit_deq.v1",
            config={
                "family": "implicit_deq",
                "full": full,
                "honesty": honesty_payload(),
            },
        ),
        "gates": dict(gates_block(entries)),
        "wall_seconds": time.perf_counter() - t0,
    }
    if full:
        dest = SCRATCH / "training" / "implicit_deq"
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
