# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""CPU smoke: parametric DeepONet zero-shot on unseen diffusivities.

Decision rule (fixed before the run): post-training held-out MSE must be
strictly below the pre-training MSE (training improves zero-shot error).
"""

from __future__ import annotations

import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
from _common import provenance, write_json  # noqa: E402

from omnibias.pinn import ComponentSpec, CoordinateSpec
from omnibias.pinn.operator import ConditioningSpec
from omnibias.pinn.operator.torch import (
    build_deeponet,
    make_parametric_heat_slab,
)
from omnibias.pinn.torch import ops as tops

DTYPE = torch.float64


def main() -> None:
    t0 = time.perf_counter()
    train = make_parametric_heat_slab(
        n_samples=6,
        n_grid=32,
        n_sensors=8,
        n_modes=2,
        n_times=5,
        diffusivity_range=(0.08, 0.18),
        seed=0,
        dtype=DTYPE,
    )
    test = make_parametric_heat_slab(
        n_samples=2,
        n_grid=32,
        n_sensors=8,
        n_modes=2,
        n_times=5,
        diffusivities=(0.05, 0.22),
        seed=1,
        dtype=DTYPE,
    )
    cond = ConditioningSpec(n_function_sensors=8, n_parameters=1)
    op = build_deeponet(
        coordinate_spec=CoordinateSpec(
            ("x", "t"),
            domain=((0.0, 2 * 3.141592653589793), (0.0, 0.5)),
            time_axis="t",
        ),
        components=ComponentSpec(("u",)),
        n_sensors=8,
        trunk_width=8,
        trunk_hidden=16,
        trunk_depth=2,
        branch_hidden=16,
        branch_depth=2,
        conditioning=cond,
        dtype=DTYPE,
    )

    def mse(slab) -> float:
        field = op.condition(slab.sensors, parameters=slab.parameters)
        pred = tops.value(field.on_grid(slab.coords), "u").reshape(
            slab.values.shape[0], -1
        )
        return float(torch.mean((pred - slab.values[..., 0]) ** 2).detach())

    err0 = mse(test)
    opt = torch.optim.Adam(op.parameters(), lr=1e-2)
    for _ in range(40):
        opt.zero_grad()
        field = op.condition(train.sensors, parameters=train.parameters)
        pred = tops.value(field.on_grid(train.coords), "u").reshape(
            train.values.shape[0], -1
        )
        loss = torch.mean((pred - train.values[..., 0]) ** 2)
        loss.backward()
        opt.step()
    err1 = mse(test)
    payload = provenance(
        schema="operator_zero_shot/v1",
        config={"steps": 40, "n_train": 6, "n_test": 2},
    )
    payload.update(
        {
            "mse_before": err0,
            "mse_after": err1,
            "improved": bool(err1 < err0),
            "elapsed_seconds": time.perf_counter() - t0,
        }
    )
    assert err1 < err0
    write_json("operator_zero_shot.json", payload)
    print("wrote docs/benchmarks/operator_zero_shot.json")


if __name__ == "__main__":
    main()
