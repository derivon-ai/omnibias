# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""DeepONet trunk-jet seam (torch): exactness, caching, shared-grid path."""

from __future__ import annotations

import pytest
import torch
from omnibias.pinn import ComponentSpec, CoordinateSpec
from omnibias.pinn.operator.torch import build_deeponet
from omnibias.pinn.operator.torch.deeponet import (
    TRUNK_JET_CACHE_KEY,
    PerSampleReadoutError,
)
from omnibias.pinn.torch import ops as tops

TOL = 1e-12


@pytest.fixture
def specs():
    return CoordinateSpec(("x", "t")), ComponentSpec(("u",))


def _operator(specs, *, trunk_width: int = 8, jet_order: int = 3, seed: int = 0):
    torch.manual_seed(seed)
    cs, comps = specs
    return build_deeponet(
        coordinate_spec=cs,
        components=comps,
        n_sensors=16,
        trunk_width=trunk_width,
        trunk_hidden=12,
        trunk_depth=2,
        branch_hidden=12,
        branch_depth=2,
        base="tanh",
        jet_order=jet_order,
    )


def test_value_matches_manual_contraction(specs) -> None:
    op = _operator(specs)
    sensors = torch.randn(3, 16, dtype=torch.float64)
    field = op.condition(sensors)
    coords = torch.randn(3, 2, dtype=torch.float64)
    got = field.forward_values(coords)
    trunk = op.core.trunk.value(coords)
    want = torch.einsum("bp,bcp->bc", trunk, field.coeffs) + field.bias
    assert torch.allclose(got, want, atol=TOL, rtol=0.0)


def test_closed_form_derivatives_match_autograd_orders_1_to_3(specs) -> None:
    op = _operator(specs, jet_order=3)
    sensors = torch.randn(1, 16, dtype=torch.float64)
    field = op.condition(sensors)
    # Single sample so aligned path is F=1, B=Q.
    g = torch.Generator().manual_seed(3)
    coords = torch.randn(5, 2, generator=g, dtype=torch.float64, requires_grad=True)
    state = field(coords.detach())
    u = tops.value(state, "u")
    # Rebuild with grad-enabled coords for the AD reference.
    u_ad = field.forward_values(coords)[:, 0]
    for order in (1, 2, 3):
        for axis in (0, 1):
            closed = tops.derivative(state, "u", axis=axis, order=order)
            # Nested autograd along one axis.
            v = u_ad
            for _ in range(order):
                (grad,) = torch.autograd.grad(v.sum(), coords, create_graph=True)
                v = grad[:, axis]
            assert torch.allclose(closed, v.detach(), atol=1e-10, rtol=0.0), (
                f"order={order} axis={axis}"
            )
    assert u.shape == (5,)


def test_residual_costs_exactly_one_trunk_jet(specs) -> None:
    op = _operator(specs, jet_order=2)
    field = op.condition(torch.randn(1, 16, dtype=torch.float64))
    coords = torch.randn(7, 2, dtype=torch.float64)
    state = field(coords)
    tops.value(state, "u")  # value path must not populate the trunk-jet cache
    assert TRUNK_JET_CACHE_KEY not in state.extra or not state.extra[TRUNK_JET_CACHE_KEY]
    tops.gradient(state, "u")
    tops.laplacian(state, "u")
    cached = state.extra[TRUNK_JET_CACHE_KEY]
    assert sorted(cached) == [2]


def test_shared_grid_and_aligned_paths_agree(specs) -> None:
    op = _operator(specs)
    sensors = torch.randn(4, 16, dtype=torch.float64)
    field = op.condition(sensors)
    Q = 6
    query = torch.randn(Q, 2, dtype=torch.float64)
    # Shared-grid path.
    state_shared = field.on_grid(query)
    u_shared = tops.value(state_shared, "u").reshape(4, Q)
    ux_shared = tops.derivative(state_shared, "u", axis=0, order=1).reshape(4, Q)
    # Aligned path: one field per sample.
    for f in range(4):
        one = op.condition(sensors[f : f + 1])
        st = one(query)
        assert torch.allclose(tops.value(st, "u"), u_shared[f], atol=TOL, rtol=0.0)
        assert torch.allclose(
            tops.derivative(st, "u", axis=0, order=1), ux_shared[f], atol=TOL, rtol=0.0
        )


def test_shared_grid_trunk_jet_computed_once(specs) -> None:
    op = _operator(specs, jet_order=2)
    field = op.condition(torch.randn(5, 16, dtype=torch.float64))
    query = torch.randn(8, 2, dtype=torch.float64)
    state = field.on_grid(query)
    tops.gradient(state, "u")
    tops.laplacian(state, "u")
    cached = state.extra[TRUNK_JET_CACHE_KEY]
    assert sorted(cached) == [2]
    # Compact cache: Q rows, not F*Q.
    assert cached[2].shape[0] == 8


def test_apply_readout_jet_refuses_shared_readout(specs) -> None:
    op = _operator(specs)
    fake = torch.zeros(3, 8, dtype=torch.float64)
    with pytest.raises(PerSampleReadoutError):
        op.core._apply_readout_jet(fake)


def test_fastpath_refusal_on_order_cap(specs) -> None:
    """arctan caps at order 2; a jet_order=3 DeepONet must refuse at construction."""
    cs, comps = specs
    with pytest.raises(ValueError, match="does not support order 3"):
        build_deeponet(
            coordinate_spec=cs,
            components=comps,
            n_sensors=8,
            trunk_width=4,
            base="arctan",
            jet_order=3,
        )


def test_fastpath_refusal_on_missing_kernel(specs) -> None:
    """An activation with fastpath=None is rejected at construction."""
    import dataclasses

    from omnibias.torch.activations.registry import get_activation

    cs, comps = specs
    nofp = dataclasses.replace(get_activation("tanh"), name="tanh_nofp", fastpath=None)
    with pytest.raises(ValueError, match="closed-form derivative"):
        build_deeponet(
            coordinate_spec=cs,
            components=comps,
            n_sensors=8,
            trunk_width=4,
            base=nofp,
            jet_order=2,
        )
