# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""CPU smoke: parametric DeepONet zero-shot vs a retrained single-instance PINN.

Headline experiment of Phase 3. Kept tiny for CI; the full calibrated
benchmark lives in ``benchmarks/operator_zero_shot.py``.
"""

from __future__ import annotations

import torch
from omnibias.pinn import ComponentSpec, CoordinateSpec
from omnibias.pinn.operator import ConditioningSpec
from omnibias.pinn.operator.torch import (
    build_deeponet,
    make_parametric_heat_slab,
)
from omnibias.pinn.torch import ops as tops

DTYPE = torch.float64


def test_parametric_deeponet_zero_shot_beats_random_init():
    """Train on a diffusivity range; held-out diffusivity error drops with training."""
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
        diffusivities=(0.05, 0.22),  # outside the train range
        seed=1,
        dtype=DTYPE,
    )
    cond = ConditioningSpec(n_function_sensors=8, n_parameters=1)
    op = build_deeponet(
        coordinate_spec=CoordinateSpec(
            ("x", "t"), domain=((0.0, 2 * 3.141592653589793), (0.0, 0.5)), time_axis="t"
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

    def _mse(operator, slab) -> float:
        field = operator.condition(slab.sensors, parameters=slab.parameters)
        state = field.on_grid(slab.coords)
        pred = tops.value(state, "u").reshape(slab.values.shape[0], -1)
        target = slab.values[..., 0]
        return float(torch.mean((pred - target) ** 2).detach())

    err0 = _mse(op, test)
    opt = torch.optim.Adam(op.parameters(), lr=1e-2)
    for _ in range(40):
        opt.zero_grad()
        field = op.condition(train.sensors, parameters=train.parameters)
        state = field.on_grid(train.coords)
        pred = tops.value(state, "u").reshape(train.values.shape[0], -1)
        loss = torch.mean((pred - train.values[..., 0]) ** 2)
        loss.backward()
        opt.step()
    err1 = _mse(op, test)
    # Training must improve zero-shot error on unseen diffusivities.
    assert err1 < err0
