# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Wave-3 trainer: composed-curvature joint Newton (theory 08-02).

Smoke earns G1 (chain-rule / HVP Hessian vs high-precision FD), G2
(slice-PD + joint-saddle escape on a 1-D Poisson nest with skill > 0
vs ``u = 0``), G3 (torch/jax parity), and G4 (reject full-parameter
Jacobian). Escape is from a *slice* critical point, not a global min
and not CCF stretch. ``sigma''`` comes from bias collapse
(``delta -> 0``). No temperature collapse.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from _common import provenance, write_json  # type: ignore[import-not-found]  # noqa: E402
from _gates import gates_block, skill_score  # type: ignore[import-not-found]  # noqa: E402

SCRATCH = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))


def _ulp_error(a: float, b: float) -> float:
    if not (math.isfinite(a) and math.isfinite(b)):
        return float("inf")
    if a == b:
        return 0.0
    scale = max(abs(a), abs(b), 1.0)
    return abs(a - b) / (np.finfo(np.float64).eps * scale)


def _rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1.0)


def _run_g1() -> dict[str, Any]:
    import torch
    from omnibias.core.composed_curvature import (
        ComposedCurvatureConfig,
        scalar_nest_hessian,
    )
    from omnibias.torch.optim_composed import composed_block_hessian
    from omnibias.torch.optim_composed import scalar_nest_hessian as torch_nest

    torch.set_default_dtype(torch.float64)
    closed = scalar_nest_hessian(0.2, 0.1, 1.0)
    jet = torch_nest(torch.tensor(0.2), torch.tensor(0.1), 1.0)

    def residual(prev: torch.Tensor, curr: torch.Tensor) -> torch.Tensor:
        return (curr * torch.tanh(prev) - 1.0).reshape(1)

    cfg = ComposedCurvatureConfig(n_directions=2, allow_full=True)
    _slice, joint, cross = composed_block_hessian(
        residual,
        torch.tensor(0.2),
        torch.tensor(0.1),
        directions=(
            (torch.tensor(1.0), torch.tensor(0.0)),
            (torch.tensor(0.0), torch.tensor(1.0)),
        ),
        config=cfg,
    )
    rels = [
        _rel(closed[0], float(joint[0, 0])),
        _rel(closed[1], float(joint[0, 1])),
        _rel(closed[2], float(joint[1, 1])),
        _rel(closed[0], jet[0]),
        _rel(closed[1], jet[1]),
        _rel(closed[2], jet[2]),
        _rel(closed[1], float(cross[0, 0])),
    ]
    worst = float(max(rels))
    return {
        "name": "g1_identity",
        "passed": worst <= 1e-10,
        "worst_rel": worst,
        "expected": 1e-10,
        "blocks": {
            "H_ww": closed[0],
            "H_wv": closed[1],
            "H_vv": closed[2],
        },
    }


def _poisson_setup(w: float):
    import torch
    from omnibias.torch.activations.registry import get_activation

    torch.set_default_dtype(torch.float64)
    xs = torch.linspace(-1.0, 1.0, 41)
    target = torch.sin(math.pi * xs)
    force = (math.pi**2) * target
    spec = get_activation("tanh")
    fp = spec.fastpath
    assert fp is not None
    w0 = torch.tensor(w)

    def residual(prev: torch.Tensor, curr: torch.Tensor) -> torch.Tensor:
        z = prev * xs
        return curr * (prev**2) * fp(z, 2) + force

    a = residual(w0, torch.tensor(1.0)) - force
    v0 = torch.tensor(-float(torch.dot(a, force)) / float(torch.dot(a, a)))
    field = v0 * torch.tanh(w0 * xs)
    skill = skill_score(field.detach().cpu().numpy(), target.detach().cpu().numpy())
    return residual, w0, v0, xs, target, skill


def _run_g2(*, full: bool) -> dict[str, Any]:
    import torch
    from omnibias.core.composed_curvature import ComposedCurvatureConfig
    from omnibias.torch.optim_composed import composed_curvature_step

    torch.set_default_dtype(torch.float64)
    seeds = (3.0, 3.05, 3.1, 2.95, 3.15) if full else (3.0,)
    cfg = ComposedCurvatureConfig(n_directions=2, allow_full=True)
    rows: list[dict[str, Any]] = []
    for w in seeds:
        residual, w0, v0, _xs, _target, skill = _poisson_setup(w)

        l0 = float(0.5 * torch.dot(residual(w0, v0), residual(w0, v0)))
        new_w, new_v, report = composed_curvature_step(residual, w0, v0, config=cfg)
        r1 = residual(new_w, new_v)
        l1 = float(0.5 * torch.dot(r1, r1))
        rows.append(
            {
                "w": w,
                "skill": float(skill),
                "lambda_min_slice": report.lambda_min_slice,
                "lambda_min_joint": report.lambda_min_joint,
                "escaped": report.escaped,
                "loss_before": l0,
                "loss_after": l1,
                "decreased": l1 < l0,
                "step_norm": report.step_norm,
            }
        )
    passed = all(
        r["skill"] > 0.0
        and r["lambda_min_slice"] >= 0.0
        and r["lambda_min_joint"] < 0.0
        and r["escaped"]
        and r["decreased"]
        for r in rows
    )
    return {
        "name": "g2_escape",
        "passed": bool(passed),
        "n_seeds": len(rows),
        "rows": rows,
        "note": (
            "Equivalent constructed residual: closed-form u_xx of "
            "v tanh(w x) on -u_xx = pi^2 sin(pi x). Skill is vs u=0 "
            "at the stalled slice min. Not a global min and not CCF stretch."
        ),
    }


def _run_g3() -> dict[str, Any]:
    import jax
    import jax.numpy as jnp
    import torch
    from omnibias.core.composed_curvature import ComposedCurvatureConfig
    from omnibias.jax.optim_composed import composed_block_hessian as jax_hess
    from omnibias.torch.optim_composed import composed_block_hessian as torch_hess

    jax.config.update("jax_enable_x64", True)
    torch.set_default_dtype(torch.float64)

    def t_res(prev: torch.Tensor, curr: torch.Tensor) -> torch.Tensor:
        return (curr * torch.tanh(prev) - 1.0).reshape(1)

    def j_res(prev: jax.Array, curr: jax.Array) -> jax.Array:
        return jnp.reshape(curr * jnp.tanh(prev) - 1.0, (1,))

    cfg = ComposedCurvatureConfig(n_directions=2, allow_full=True)
    t_dirs = (
        (torch.tensor(1.0), torch.tensor(0.0)),
        (torch.tensor(0.0), torch.tensor(1.0)),
    )
    j_dirs = (
        (jnp.asarray(1.0), jnp.asarray(0.0)),
        (jnp.asarray(0.0), jnp.asarray(1.0)),
    )
    _ts, t_joint, t_cross = torch_hess(
        t_res, torch.tensor(0.2), torch.tensor(0.1), t_dirs, config=cfg
    )
    _js, j_joint, j_cross = jax_hess(
        j_res, jnp.asarray(0.2), jnp.asarray(0.1), j_dirs, config=cfg
    )
    worst = 0.0
    for i in range(2):
        for j in range(2):
            worst = max(worst, _ulp_error(float(t_joint[i, j]), float(j_joint[i, j])))
    worst = max(worst, _ulp_error(float(t_cross[0, 0]), float(j_cross[0, 0])))
    return {
        "name": "g3_parity",
        "passed": worst <= 4.0,
        "worst_ulp": float(worst),
        "expected": 4.0,
    }


def _run_g4() -> dict[str, Any]:
    import torch
    from omnibias.core.composed_curvature import ComposedCurvatureConfig
    from omnibias.torch.optim_composed import composed_block_hessian

    torch.set_default_dtype(torch.float64)

    def residual(prev: torch.Tensor, curr: torch.Tensor) -> torch.Tensor:
        return (curr * torch.tanh(prev) - 1.0).reshape(1)

    raised = False
    try:
        composed_block_hessian(
            residual,
            torch.tensor(0.2),
            torch.tensor(0.1),
            config=ComposedCurvatureConfig(n_directions=2, allow_full=False),
        )
    except ValueError as exc:
        raised = "allow_full" in str(exc)
    return {
        "name": "g4_no_flood",
        "passed": bool(raised),
        "note": "n_directions >= n_params without allow_full raises ValueError",
    }


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args(argv)
    full = bool(args.full)
    artifact = "composed_curvature.json" if full else "composed_curvature_smoke.json"
    t0 = time.perf_counter()
    print("G1 identity...")
    g1 = _run_g1()
    print("G2 escape...")
    g2 = _run_g2(full=full)
    print("G3 parity...")
    g3 = _run_g3()
    print("G4 no flood...")
    g4 = _run_g4()
    entries = [g1, g2, g3, g4]
    for e in entries:
        if not e["passed"]:
            raise AssertionError(f"{e['name']} failed: {e}")
    gates = dict(gates_block(entries))
    payload = {
        **provenance(
            schema="omnibias.benchmarks.composed_curvature.v1",
            config={
                "family": "composed_curvature",
                "full": full,
                "honesty": {
                    "navier_stokes_proof_claim": False,
                    "ccf_stretch_cleared": False,
                    "global_min_claim": False,
                    "temperature_collapse": False,
                },
            },
        ),
        "gates": gates,
        "wall_seconds": time.perf_counter() - t0,
    }
    if full:
        dest = SCRATCH / "training" / "composed_curvature"
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
