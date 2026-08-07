# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""DeepONet contracts: readout-independence, linear-solver refusal, cage wrapping."""

from __future__ import annotations

import pytest
import torch
from omnibias.fields._core.field_base import READOUT_INDEPENDENT_ATTR
from omnibias.pinn import ComponentSpec, CoordinateSpec
from omnibias.pinn._core.constrained import HardCondition, dirichlet
from omnibias.pinn.operator.torch import build_deeponet
from omnibias.pinn.solver.torch.readout import readout_size
from omnibias.pinn.torch import ops as tops
from omnibias.pinn.torch.cage.conservation import HardBoundaryField
from omnibias.pinn.torch.cage.constrained import ConstrainedExpressionField

DTYPE = torch.float64
EXACT = 1e-12


def _op(seed: int = 0):
    torch.manual_seed(seed)
    return build_deeponet(
        coordinate_spec=CoordinateSpec(
            ("t", "x"), domain=((0.0, 1.0), (0.0, 1.0)), time_axis="t"
        ),
        components=ComponentSpec(("u",)),
        n_sensors=8,
        trunk_width=6,
        trunk_hidden=10,
        trunk_depth=2,
        branch_hidden=10,
        branch_depth=2,
        jet_order=2,
        dtype=DTYPE,
    )


def test_deeponet_field_declares_readout_independence() -> None:
    field = _op().condition(torch.randn(1, 8, dtype=DTYPE))
    assert getattr(field, READOUT_INDEPENDENT_ATTR) is True


def test_linear_solver_refuses_deeponet_with_named_error() -> None:
    field = _op().condition(torch.randn(1, 8, dtype=DTYPE))
    with pytest.raises(TypeError, match="solve_optimize"):
        readout_size(field)


def test_hard_boundary_cage_wraps_deeponet_and_stays_exact() -> None:
    op = _op()
    base = op.condition(torch.randn(1, 8, dtype=DTYPE))
    cage = HardBoundaryField(
        base=base,
        distance_fn=lambda c: c[:, 1] * (1.0 - c[:, 1]),
        boundary_value_fn=lambda c: {"u": torch.zeros(c.shape[0], dtype=DTYPE)},
        bounded_names=("u",),
    )
    # On the faces x=0 and x=1 the distance vanishes, so u must be exactly 0.
    faces = torch.tensor(
        [[0.3, 0.0], [0.7, 0.0], [0.2, 1.0], [0.9, 1.0]], dtype=DTYPE
    )
    state = cage(faces)
    u = tops.value(state, "u")
    assert float(u.detach().abs().max()) < EXACT


def test_constrained_expression_cage_wraps_deeponet_and_stays_exact() -> None:
    op = _op()
    base = op.condition(torch.randn(1, 8, dtype=DTYPE))
    cage = ConstrainedExpressionField(
        base=base,
        conditions=[
            HardCondition("u", 1, dirichlet(0.0), 0.0),
            HardCondition("u", 1, dirichlet(1.0), 0.0),
        ],
    )
    lo = torch.rand(16, 2, dtype=DTYPE)
    lo[:, 1] = 0.0
    hi = torch.rand(16, 2, dtype=DTYPE)
    hi[:, 1] = 1.0
    assert float(tops.value(cage(lo), "u").detach().abs().max()) < EXACT
    assert float(tops.value(cage(hi), "u").detach().abs().max()) < EXACT
