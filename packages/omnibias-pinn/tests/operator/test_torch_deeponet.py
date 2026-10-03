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
    cached = state.extra[TRUNK_JET_CACHE_KEY]
    assert sorted(cached) == [2]
    tops.laplacian(state, "u")  # fast lane: no extra trunk-jet entry
    assert sorted(state.extra[TRUNK_JET_CACHE_KEY]) == [2]


def test_laplacian_fast_lane_leaves_trunk_jet_cache_unchanged(specs) -> None:
    from omnibias.pinn.torch.fields.jet_mlp import FAST_LANE_CACHE_KEY

    op = _operator(specs, jet_order=2)
    field = op.condition(torch.randn(1, 16, dtype=torch.float64))
    state = field(torch.randn(5, 2, dtype=torch.float64))
    tops.laplacian(state, "u")
    assert TRUNK_JET_CACHE_KEY not in state.extra or not state.extra[TRUNK_JET_CACHE_KEY]
    assert FAST_LANE_CACHE_KEY in state.extra


def test_laplacian_matches_manual_branch_contraction(specs) -> None:
    """The deep-field Laplacian fast lane must not silently read the raw trunk.

    ``G(u)(y) = bias + sum_k coeffs_k * t_k(y)`` with ``coeffs``/``bias`` constant
    in ``y``, so ``Delta_y G(u)(y) = sum_k coeffs_k * Delta_y t_k(y)``. The trunk
    Laplacian is taken independently via ``torch.func.hessian`` on the trunk's own
    ``value``, never touching the field's ``polylaplacian`` -- an oracle the fast
    lane cannot share a bug with. This pins
    :class:`~omnibias.pinn.operator.torch.deeponet.DeepONetField` against the
    generic ``_JetFieldBase.polylaplacian`` fast lane, which reads
    ``net._layer_specs()`` directly and would otherwise compute the Laplacian of
    a trunk basis column instead of the branch-contracted operator output.
    """
    op = _operator(specs, trunk_width=5, jet_order=2)
    sensors = torch.randn(4, 16, dtype=torch.float64)
    field = op.condition(sensors)
    coords = torch.randn(4, 2, dtype=torch.float64)
    state = field(coords)
    got = tops.laplacian(state, "u")

    def trunk_laplacian(y: Tensor) -> Tensor:
        # ``t`` is the (non-spatial) time axis in this fixture's CoordinateSpec,
        # so the PINN Laplacian sums only over the spatial axis ``x`` (index 0).
        hess = torch.func.hessian(lambda yi: op.core.trunk.value(yi[None, :])[0])(y)
        return hess[:, 0, 0]

    trunk_lap = torch.stack([trunk_laplacian(coords[i]) for i in range(coords.shape[0])])
    want = torch.einsum("bp,bcp->bc", trunk_lap, field.coeffs)[:, 0]
    assert torch.allclose(got, want, atol=1e-9, rtol=0.0)
    # Regression guard: the buggy fast lane would instead equal a *raw* trunk
    # basis column's Laplacian, which disagrees with the correct contraction
    # whenever the branch coefficients are not a one-hot selector (true here).
    raw_trunk_col0 = torch.stack(
        [trunk_laplacian(coords[i])[0] for i in range(coords.shape[0])]
    )
    assert not torch.allclose(got, raw_trunk_col0, atol=1e-6, rtol=0.0)


def test_polylaplacian_matches_manual_branch_contraction(specs) -> None:
    """``Delta^2`` (biharmonic) on the DeepONet field, same oracle as above."""
    op = _operator(specs, trunk_width=5, jet_order=4)
    sensors = torch.randn(3, 16, dtype=torch.float64)
    field = op.condition(sensors)
    coords = torch.randn(3, 2, dtype=torch.float64)
    state = field(coords)
    got = tops.biharmonic(state, "u")

    def trunk_bilaplacian(y: Tensor) -> Tensor:
        # ``t`` is the (non-spatial) time axis, so ``Delta^2`` is the pure
        # fourth ``x``-derivative (axis 0) of each trunk basis function.
        def lap(yi: Tensor) -> Tensor:
            hess = torch.func.hessian(lambda z: op.core.trunk.value(z[None, :])[0])(yi)
            return hess[:, 0, 0]

        out = []
        for k in range(op.core.trunk.out_dim):
            hess_k = torch.func.hessian(lambda z: lap(z)[k])(y)
            out.append(hess_k[0, 0])
        return torch.stack(out)

    trunk_bilap = torch.stack(
        [trunk_bilaplacian(coords[i]) for i in range(coords.shape[0])]
    )
    want = torch.einsum("bp,bcp->bc", trunk_bilap, field.coeffs)[:, 0]
    assert torch.allclose(got, want, atol=1e-6, rtol=0.0)


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
    cached = state.extra[TRUNK_JET_CACHE_KEY]
    assert sorted(cached) == [2]
    # Compact cache: Q rows, not F*Q.
    assert cached[2].shape[0] == 8
    tops.laplacian(state, "u")  # fast lane uses trunk coords[:Q], not the jet cache
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
