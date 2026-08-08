# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Parametric DeepONet zero-shot vs per-instance PINN retrain.

Train once on a parameter range, evaluate on strictly held-out diffusivities.
Compare (a) conditioned operator, (b) unconditioned ablation, and (c) a PINN
retrained from scratch per test instance under an equal step budget.

Modes
-----
* ``--smoke`` (default): 1 seed, tiny nets — CI wiring gate.
* ``--full``: multiple seeds — acceptance artifact under ``$OMNIBIAS_SCRATCH``.

Guarantee level: operator accuracy is optimised on the training envelope, not
proven; the decision rule is pre-declared improvement on held-out MSE.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, os.path.dirname(__file__))
from _common import provenance, write_json  # noqa: E402

from omnibias.fields._core.components import ComponentSpec
from omnibias.fields._core.coords import CoordinateSpec
from omnibias.pinn.operator import ConditioningSpec
from omnibias.pinn.operator.torch import (
    build_deeponet,
    data_loss,
    make_parametric_heat_slab,
)
from omnibias.pinn.torch import ops as tops
from omnibias.pinn.torch.fields import OneLayerVectorField

SCRATCH = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))
DTYPE = torch.float64


def _count_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def _mse(operator, slab) -> float:
    with torch.no_grad():
        if hasattr(slab, "parameters") and operator.spec.conditioning.has_parameters:
            field = operator.condition(slab.sensors, parameters=slab.parameters)
        else:
            field = operator.condition(slab.sensors)
        pred = tops.value(field.on_grid(slab.coords), "u").reshape(
            slab.values.shape[0], -1
        )
        target = slab.values[..., 0]
        return float(torch.mean((pred - target) ** 2).detach())


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
) -> tuple[float, float]:
    """Per-instance one-layer PINN with equal step budget; returns (mse, wall_s)."""
    cs = CoordinateSpec(
        ("x", "t"),
        domain=((0.0, 2 * 3.141592653589793), (0.0, 0.5)),
        time_axis="t",
    )
    comps = ComponentSpec(("u",))
    mses: list[float] = []
    t0 = time.perf_counter()
    for i in range(test.values.shape[0]):
        field = OneLayerVectorField(
            coordinate_spec=cs, components=comps, hidden=hidden, base="tanh"
        ).to(dtype=DTYPE)
        opt = torch.optim.Adam(field.parameters(), lr=lr)
        target = test.values[i, :, 0]
        coords = test.coords
        for _ in range(steps):
            opt.zero_grad()
            pred = tops.value(field(coords), "u")
            loss = torch.mean((pred - target) ** 2)
            loss.backward()
            opt.step()
        with torch.no_grad():
            pred = tops.value(field(coords), "u")
            mses.append(float(torch.mean((pred - target) ** 2).detach()))
    return float(sum(mses) / len(mses)), time.perf_counter() - t0


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
    steps = 40 if smoke else 400
    hidden = 16 if smoke else 32
    n_sensors = 8 if smoke else 16
    train = make_parametric_heat_slab(
        n_samples=n_train,
        n_grid=32 if smoke else 64,
        n_sensors=n_sensors,
        n_modes=2,
        n_times=5 if smoke else 9,
        diffusivity_range=(0.08, 0.18),
        seed=seed,
        dtype=DTYPE,
    )
    test = make_parametric_heat_slab(
        n_samples=n_test,
        n_grid=32 if smoke else 64,
        n_sensors=n_sensors,
        n_modes=2,
        n_times=5 if smoke else 9,
        diffusivities=tuple(0.04 + 0.04 * i for i in range(n_test)),
        seed=seed + 1000,
        dtype=DTYPE,
    )
    op_cond = _build_op(conditioned=True, n_sensors=n_sensors, hidden=hidden)
    op_ablate = _build_op(conditioned=False, n_sensors=n_sensors, hidden=hidden)
    err0 = _mse(op_cond, test)
    wall_cond = _train_operator(
        op_cond, train, steps=steps, lr=1e-2, conditioned=True
    )
    err_cond = _mse(op_cond, test)
    wall_ablate = _train_operator(
        op_ablate, train, steps=steps, lr=1e-2, conditioned=False
    )
    err_ablate = _mse(op_ablate, test)
    err_pinn, wall_pinn = _retrain_pinn_per_instance(
        test, steps=steps, lr=1e-2, hidden=hidden
    )
    n_params = _count_params(op_cond)
    amort_break_even = int(
        max(1, round(wall_pinn / max(wall_cond / max(n_test, 1), 1e-9)))
    )
    return {
        "seed": seed,
        "mse_before": err0,
        "mse_conditioned": err_cond,
        "mse_unconditioned": err_ablate,
        "mse_pinn_retrain": err_pinn,
        "wall_s_conditioned": wall_cond,
        "wall_s_unconditioned": wall_ablate,
        "wall_s_pinn_retrain": wall_pinn,
        "n_params_conditioned": n_params,
        "amortization_break_even_queries": amort_break_even,
        "improved_vs_init": bool(err_cond < err0),
        "conditioned_beats_ablation": bool(err_cond < err_ablate),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full",
        action="store_true",
        help="multi-seed acceptance run (default is --smoke)",
    )
    args = parser.parse_args()
    smoke = not args.full
    seeds = (0,) if smoke else (0, 1, 2)
    t0 = time.perf_counter()
    rows = [_run_seed(s, smoke=smoke) for s in seeds]
    payload = provenance(
        schema="operator_zero_shot/v2",
        config={
            "smoke": smoke,
            "seeds": list(seeds),
            "decision_rule": (
                "held-out MSE after training must beat pre-training MSE; "
                "conditioned arm should beat unconditioned ablation on median MSE"
            ),
        },
    )
    payload["runs"] = rows
    payload["summary"] = {
        "median_mse_conditioned": float(
            sorted(r["mse_conditioned"] for r in rows)[len(rows) // 2]
        ),
        "median_mse_unconditioned": float(
            sorted(r["mse_unconditioned"] for r in rows)[len(rows) // 2]
        ),
        "median_mse_pinn_retrain": float(
            sorted(r["mse_pinn_retrain"] for r in rows)[len(rows) // 2]
        ),
        "elapsed_seconds": time.perf_counter() - t0,
    }
    write_json("operator_zero_shot.json", payload)
    print("wrote docs/benchmarks/operator_zero_shot.json")
    if not smoke:
        scratch_path = SCRATCH / "benchmarks" / "operator_zero_shot_full.json"
        scratch_path.parent.mkdir(parents=True, exist_ok=True)
        scratch_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {scratch_path}")
    assert all(r["improved_vs_init"] for r in rows)


if __name__ == "__main__":
    main()
