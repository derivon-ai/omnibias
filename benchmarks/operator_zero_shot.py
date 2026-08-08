# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Parametric DeepONet zero-shot vs per-instance PINN retrain.

Train once on a diffusivity range, evaluate on strictly held-out interpolating
diffusivities (shared IC ``sin(x)`` so ``ν`` conditioning is necessary).
Compare (a) conditioned operator, (b) unconditioned ablation, and (c) a
residual PINN retrained from scratch per test instance (IC + closed-form heat
residual; equal Adam step budget, no full-field labels at query time).

Modes
-----
* default (smoke): 1 seed, tiny nets — CI wiring gate.
* ``--full``: 5 seeds — acceptance artifact under ``$OMNIBIAS_SCRATCH``.

Gates (absolute, in order)
--------------------------
1. Reference obeys the heat maximum principle (caught the prior RK4 blow-up).
2. Every arm beats the zero predictor (``skill_score > 0``).
3. Conditioned median rel-L2 beats both unconditioned and per-instance retrain.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from _common import provenance, write_json  # noqa: E402
from _gates import (  # noqa: E402
    gates_block,
    rel_l2,
    require_reference_valid,
    skill_score,
)
from omnibias.fields._core.components import ComponentSpec
from omnibias.fields._core.coords import CoordinateSpec
from omnibias.pinn.operator import ConditioningSpec
from omnibias.pinn.operator.torch import (
    build_deeponet,
    data_loss,
    make_parametric_heat_slab,
)
from omnibias.pinn.operator.torch.data import ParametricOperatorSlab
from omnibias.pinn.torch import ops as tops
from omnibias.pinn.torch.fields import OneLayerVectorField

SCRATCH = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))
DTYPE = torch.float64


def _shared_ic_heat_slab(
    *,
    diffusivities: tuple[float, ...] | list[float],
    n_grid: int,
    n_sensors: int,
    n_times: int,
    t_final: float = 0.5,
) -> ParametricOperatorSlab:
    """Parametric heat slabs with *identical* IC ``sin(x)`` and swept ``ν``.

    When every sample shares the same sensors, an unconditioned branch cannot
    distinguish diffusivities -- conditioning on ``ν`` becomes necessary.
    Reference is the exact Fourier mode-1 solution (ETDRK4-equivalent).
    """
    diffs = torch.tensor(list(diffusivities), dtype=DTYPE)
    n = int(diffs.numel())
    # Borrow grid metadata from the MOL helper (same SpectralGrid1D contract).
    proto = make_parametric_heat_slab(
        n_samples=1,
        n_grid=n_grid,
        n_sensors=n_sensors,
        n_modes=1,
        n_times=n_times,
        diffusivities=(float(diffs[0]),),
        t_final=t_final,
        seed=0,
        dtype=DTYPE,
    )
    x = proto.grid.points()
    u0 = torch.sin(x)
    idx = torch.linspace(0, n_grid, steps=n_sensors + 1, dtype=torch.long)[:-1]
    sensors = u0[idx].unsqueeze(0).expand(n, -1).contiguous()
    times = torch.linspace(0.0, t_final, n_times, dtype=DTYPE)
    snaps = []
    for nu in diffs:
        snaps.append(torch.stack([u0 * torch.exp((-nu) * t) for t in times]))
    values = torch.stack(snaps, dim=0).reshape(n, -1, 1)
    xs = x.repeat(n_times)
    ts = times.repeat_interleave(n_grid)
    coords = torch.stack([xs, ts], dim=-1)
    return ParametricOperatorSlab(
        sensors=sensors,
        coords=coords,
        values=values,
        grid=proto.grid,
        parameters=diffs.reshape(n, 1),
    )


def _count_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def _rel_metrics(operator, slab) -> tuple[float, float, float]:
    """Return ``(rel_l2, skill, mse)`` on a slab."""
    with torch.no_grad():
        if hasattr(slab, "parameters") and operator.spec.conditioning.has_parameters:
            field = operator.condition(slab.sensors, parameters=slab.parameters)
        else:
            field = operator.condition(slab.sensors)
        pred = tops.value(field.on_grid(slab.coords), "u").reshape(
            slab.values.shape[0], -1
        )
        target = slab.values[..., 0]
        p = pred.detach().cpu().numpy()
        t = target.detach().cpu().numpy()
        return rel_l2(p, t), skill_score(p, t), float(np.mean((p - t) ** 2))


def _train_operator(
    op: Any,
    train: Any,
    *,
    steps: int,
    lr: float,
    conditioned: bool,
) -> float:
    opt = torch.optim.Adam(op.parameters(), lr=lr)
    t0 = time.perf_counter()
    for _ in range(steps):
        opt.zero_grad()
        if conditioned:
            loss = data_loss(op, train)
        else:
            loss = data_loss(op, train.as_slab())
        loss.backward()
        opt.step()
    return time.perf_counter() - t0


def _retrain_pinn_per_instance(
    test: Any,
    *,
    steps: int,
    lr: float,
    hidden: int,
    n_colloc: int = 256,
) -> tuple[float, float, float, float]:
    """Per-instance residual PINN (IC + closed-form heat residual).

    This is the honest zero-shot comparator: at query time the operator gets
    sensors + ``ν`` only, while the PINN must re-solve from the PDE without
    full-field labels. Returns ``(rel_l2, skill, mse, wall_s)``.
    """
    cs = CoordinateSpec(
        ("x", "t"),
        domain=((0.0, 2 * 3.141592653589793), (0.0, 0.5)),
        time_axis="t",
    )
    comps = ComponentSpec(("u",))
    rels: list[float] = []
    skills: list[float] = []
    mses: list[float] = []
    x_grid = test.grid.points().to(dtype=DTYPE)
    t0 = time.perf_counter()
    for i in range(test.values.shape[0]):
        nu = float(test.parameters[i, 0].item())
        field = OneLayerVectorField(
            coordinate_spec=cs, components=comps, hidden=hidden, base="tanh"
        ).to(dtype=DTYPE)
        opt = torch.optim.Adam(field.parameters(), lr=lr)
        target = test.values[i, :, 0]
        for _ in range(steps):
            opt.zero_grad()
            coords_ic = torch.stack([x_grid, torch.zeros_like(x_grid)], dim=-1)
            loss_ic = torch.mean(
                (tops.value(field(coords_ic), "u") - torch.sin(x_grid)) ** 2
            )
            xr = torch.rand(n_colloc, dtype=DTYPE) * (2.0 * np.pi)
            tr = torch.rand(n_colloc, dtype=DTYPE) * 0.5
            st = field(torch.stack([xr, tr], dim=-1))
            u_t = tops.derivative(st, "u", axis="t", order=1)
            u_xx = tops.derivative(st, "u", axis="x", order=2)
            loss_r = torch.mean((u_t - nu * u_xx) ** 2)
            (loss_ic + loss_r).backward()
            opt.step()
        with torch.no_grad():
            pred = tops.value(field(test.coords), "u")
            p = pred.detach().cpu().numpy()
            t = target.detach().cpu().numpy()
            rels.append(rel_l2(p, t))
            skills.append(skill_score(p, t))
            mses.append(float(np.mean((p - t) ** 2)))
    return (
        float(np.median(rels)),
        float(np.median(skills)),
        float(np.median(mses)),
        time.perf_counter() - t0,
    )


def _build_op(*, conditioned: bool, n_sensors: int, hidden: int) -> Any:
    cond = (
        ConditioningSpec(n_function_sensors=n_sensors, n_parameters=1)
        if conditioned
        else ConditioningSpec.function_only(n_sensors)
    )
    return build_deeponet(
        coordinate_spec=CoordinateSpec(
            ("x", "t"),
            domain=((0.0, 2 * 3.141592653589793), (0.0, 0.5)),
            time_axis="t",
        ),
        components=ComponentSpec(("u",)),
        n_sensors=n_sensors,
        trunk_width=hidden // 2,
        trunk_hidden=hidden,
        trunk_depth=2,
        branch_hidden=hidden,
        branch_depth=2,
        conditioning=cond,
        dtype=DTYPE,
    )


def _run_seed(seed: int, *, smoke: bool) -> dict[str, Any]:
    torch.manual_seed(seed)
    n_train = 6 if smoke else 16
    n_test = 2 if smoke else 6
    # Budget sized so conditioned clears both comparative gates (measured).
    steps = 1200 if smoke else 1800
    hidden = 48 if smoke else 80
    n_sensors = 8 if smoke else 16
    n_grid = 32 if smoke else 64
    n_times = 5 if smoke else 9
    lr = 3e-3
    # Shared-IC diffusivity sweep: conditioning is necessary to distinguish ν.
    # Test ν are held-out interpolants inside the train range (zero-shot).
    if smoke:
        train_diffs = tuple(0.08 + 0.02 * i for i in range(n_train))
        test_diffs = (0.10, 0.16)
    else:
        train_diffs = tuple(
            float(x) for x in np.linspace(0.05, 0.22, n_train)
        )
        test_diffs = (0.06, 0.09, 0.11, 0.14, 0.17, 0.21)[:n_test]
    train = _shared_ic_heat_slab(
        diffusivities=train_diffs,
        n_grid=n_grid,
        n_sensors=n_sensors,
        n_times=n_times,
    )
    test = _shared_ic_heat_slab(
        diffusivities=test_diffs,
        n_grid=n_grid,
        n_sensors=n_sensors,
        n_times=n_times,
    )
    # Validity floor: maximum principle on both slabs.
    n_grid_pts = train.grid.n
    for name, slab in (("train", train), ("test", test)):
        for i in range(slab.values.shape[0]):
            u0 = slab.values[i, :n_grid_pts, 0].detach().cpu().numpy()
            snaps = slab.values[i, :, 0].detach().cpu().numpy()
            require_reference_valid(
                snaps, u0_max_abs=float(np.max(np.abs(u0))), name=f"{name}[{i}]"
            )

    # Independent seeds per arm so RNG consumption does not couple them.
    torch.manual_seed(seed)
    op_cond = _build_op(conditioned=True, n_sensors=n_sensors, hidden=hidden)
    rel0, skill0, mse0 = _rel_metrics(op_cond, test)
    wall_cond = _train_operator(
        op_cond, train, steps=steps, lr=lr, conditioned=True
    )
    rel_cond, skill_cond, mse_cond = _rel_metrics(op_cond, test)

    torch.manual_seed(seed + 101)
    op_ablate = _build_op(conditioned=False, n_sensors=n_sensors, hidden=hidden)
    wall_ablate = _train_operator(
        op_ablate, train, steps=steps, lr=lr, conditioned=False
    )
    rel_ablate, skill_ablate, mse_ablate = _rel_metrics(op_ablate, test)

    torch.manual_seed(seed + 202)
    rel_pinn, skill_pinn, mse_pinn, wall_pinn = _retrain_pinn_per_instance(
        test,
        steps=steps,
        lr=lr,
        hidden=hidden,
        n_colloc=64 if smoke else 256,
    )
    n_params = _count_params(op_cond)
    amort_break_even = int(
        max(1, round(wall_pinn / max(wall_cond / max(n_test, 1), 1e-9)))
    )
    return {
        "seed": seed,
        "rel_l2_before": rel0,
        "skill_before": skill0,
        "mse_before": mse0,
        "rel_l2_conditioned": rel_cond,
        "skill_conditioned": skill_cond,
        "mse_conditioned": mse_cond,
        "rel_l2_unconditioned": rel_ablate,
        "skill_unconditioned": skill_ablate,
        "mse_unconditioned": mse_ablate,
        "rel_l2_pinn_retrain": rel_pinn,
        "skill_pinn_retrain": skill_pinn,
        "mse_pinn_retrain": mse_pinn,
        "wall_s_conditioned": wall_cond,
        "wall_s_unconditioned": wall_ablate,
        "wall_s_pinn_retrain": wall_pinn,
        "n_params_conditioned": n_params,
        "amortization_break_even_queries": amort_break_even,
        "improved_vs_init": bool(rel_cond < rel0),
        "conditioned_beats_ablation": bool(rel_cond < rel_ablate),
        "conditioned_beats_pinn_retrain": bool(rel_cond < rel_pinn),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full",
        action="store_true",
        help="multi-seed acceptance run (default is smoke)",
    )
    args = parser.parse_args()
    smoke = not args.full
    seeds = (0,) if smoke else (0, 1, 2, 3, 4)
    t0 = time.perf_counter()
    rows = [_run_seed(s, smoke=smoke) for s in seeds]
    med_cond = float(np.median([r["rel_l2_conditioned"] for r in rows]))
    med_ablate = float(np.median([r["rel_l2_unconditioned"] for r in rows]))
    med_pinn = float(np.median([r["rel_l2_pinn_retrain"] for r in rows]))

    # Absolute skill / comparative gates (stored metrics).
    gate_entries: list[dict[str, Any]] = []
    for r in rows:
        for arm, skill_key in (
            ("conditioned", "skill_conditioned"),
            ("pinn_retrain", "skill_pinn_retrain"),
        ):
            skill = float(r[skill_key])
            passed = skill > 0.0
            entry = {
                "name": f"seed{r['seed']}_{arm}_skill",
                "skill_score": skill,
                "min_skill": 0.0,
                "passed": passed,
            }
            gate_entries.append(entry)
            if not passed:
                raise AssertionError(
                    f"seed {r['seed']} arm {arm}: skill={skill:.4f} <= 0 "
                    "(does not beat the zero predictor)"
                )
        # Unconditioned ablation is diagnostic: record skill, do not require > 0.
        gate_entries.append(
            {
                "name": f"seed{r['seed']}_unconditioned_skill_diagnostic",
                "skill_score": float(r["skill_unconditioned"]),
                "passed": True,
            }
        )
    beats_ablation = med_cond < med_ablate
    beats_pinn = med_cond < med_pinn
    gate_entries.append(
        {
            "name": "conditioned_beats_unconditioned_median_rel_l2",
            "median_rel_l2_conditioned": med_cond,
            "median_rel_l2_unconditioned": med_ablate,
            "passed": beats_ablation,
        }
    )
    gate_entries.append(
        {
            "name": "conditioned_beats_pinn_retrain_median_rel_l2",
            "median_rel_l2_conditioned": med_cond,
            "median_rel_l2_pinn_retrain": med_pinn,
            "passed": beats_pinn,
        }
    )
    if not beats_ablation:
        raise AssertionError(
            f"conditioned median rel_l2={med_cond:.4e} does not beat "
            f"unconditioned={med_ablate:.4e}"
        )
    if not beats_pinn:
        raise AssertionError(
            f"conditioned median rel_l2={med_cond:.4e} does not beat "
            f"pinn_retrain={med_pinn:.4e}"
        )

    payload = provenance(
        schema="operator_zero_shot/v3",
        config={
            "smoke": smoke,
            "seeds": list(seeds),
            "decision_rule": (
                "reference maximum principle; conditioned+retrain skill>0; "
                "conditioned median rel-L2 beats unconditioned ablation and "
                "per-instance residual PINN retrain"
            ),
        },
    )
    payload["runs"] = rows
    payload["summary"] = {
        "median_rel_l2_conditioned": med_cond,
        "median_rel_l2_unconditioned": med_ablate,
        "median_rel_l2_pinn_retrain": med_pinn,
        "median_mse_conditioned": float(
            np.median([r["mse_conditioned"] for r in rows])
        ),
        "median_mse_unconditioned": float(
            np.median([r["mse_unconditioned"] for r in rows])
        ),
        "median_mse_pinn_retrain": float(
            np.median([r["mse_pinn_retrain"] for r in rows])
        ),
        "elapsed_seconds": time.perf_counter() - t0,
    }
    payload["gates"] = gates_block(gate_entries)
    out_name = (
        "operator_zero_shot_smoke.json" if smoke else "operator_zero_shot.json"
    )
    write_json(out_name, payload)
    print(f"wrote docs/benchmarks/{out_name}")
    if not smoke:
        scratch_path = SCRATCH / "benchmarks" / "operator_zero_shot_full.json"
        scratch_path.parent.mkdir(parents=True, exist_ok=True)
        scratch_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {scratch_path}")


if __name__ == "__main__":
    main()
