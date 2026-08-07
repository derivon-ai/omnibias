# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""4th-order KS operator: closed-form u_xxxx, one-jet residual, torch/jax parity."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import torch
from omnibias.jax.activations import get_activation as jax_get_activation
from omnibias.jax.architectures.pinn import JetMLP
from omnibias.pinn import ComponentSpec, CoordinateSpec
from omnibias.pinn.jax import ops as jops
from omnibias.pinn.operator._core.spec import OperatorSpec
from omnibias.pinn.operator.jax import (
    ks_residual_loss as jax_ks_residual_loss,
)
from omnibias.pinn.operator.jax import (
    ks_residual_loss_fd as jax_ks_residual_loss_fd,
)
from omnibias.pinn.operator.jax import (
    make_deeponet,
    make_ks_slab,
)
from omnibias.pinn.operator.jax.deeponet import (
    TRUNK_JET_CACHE_KEY as JAX_CACHE,
)
from omnibias.pinn.operator.jax.deeponet import (
    DeepONetOperator as JaxOp,
)
from omnibias.pinn.operator.jax.deeponet import (
    _BranchNet,
)
from omnibias.pinn.operator.torch import (
    build_deeponet,
    ks_residual_loss,
    ks_residual_loss_fd,
)
from omnibias.pinn.operator.torch import (
    make_ks_slab as torch_make_ks_slab,
)
from omnibias.pinn.operator.torch.deeponet import TRUNK_JET_CACHE_KEY
from omnibias.pinn.torch import ops as tops
from omnibias.pinn.torch.equations.kuramoto_sivashinsky import KuramotoSivashinsky
from omnibias.torch.architectures.pinn import JetMLP as TorchJetMLP

jax.config.update("jax_enable_x64", True)

DTYPE = torch.float64
RTOL = 1e-11
CS = CoordinateSpec(("x", "t"), time_axis="t")
COMPS = ComponentSpec(("u",))


def _torch_op(*, jet_order: int = 4, seed: int = 0):
    torch.manual_seed(seed)
    return build_deeponet(
        coordinate_spec=CS,
        components=COMPS,
        n_sensors=8,
        trunk_width=6,
        trunk_hidden=10,
        trunk_depth=2,
        branch_hidden=10,
        branch_depth=2,
        jet_order=jet_order,
        dtype=DTYPE,
    )


def test_shipped_ks_equation_runs_on_deeponet() -> None:
    op = _torch_op()
    field = op.condition(torch.randn(1, 8, dtype=DTYPE))
    state = field(torch.randn(5, 2, dtype=DTYPE))
    out = KuramotoSivashinsky()(state)
    assert out.residual.shape == (5,)
    assert torch.isfinite(out.residual).all()


def test_u_xxxx_matches_nested_autograd() -> None:
    op = _torch_op()
    field = op.condition(torch.randn(1, 8, dtype=DTYPE))
    coords = torch.randn(6, 2, dtype=DTYPE, requires_grad=True)
    state = field(coords.detach())
    closed = tops.derivative(state, "u", axis=0, order=4)
    u = field.forward_values(coords)[:, 0]
    v = u
    for _ in range(4):
        (g,) = torch.autograd.grad(v.sum(), coords, create_graph=True)
        v = g[:, 0]
    assert torch.allclose(closed, v.detach(), atol=1e-10, rtol=0.0)


def test_ks_residual_costs_exactly_one_order4_trunk_jet() -> None:
    op = _torch_op(jet_order=4)
    field = op.condition(torch.randn(1, 8, dtype=DTYPE))
    state = field(torch.randn(7, 2, dtype=DTYPE))
    tops.value(state, "u")
    assert TRUNK_JET_CACHE_KEY not in state.extra or not state.extra[TRUNK_JET_CACHE_KEY]
    _ = KuramotoSivashinsky()(state)
    cached = state.extra[TRUNK_JET_CACHE_KEY]
    assert sorted(cached) == [4]


def test_ks_residual_loss_finite() -> None:
    op = _torch_op()
    sensors = torch.randn(2, 8, dtype=DTYPE)
    coords = torch.randn(12, 2, dtype=DTYPE)
    loss = ks_residual_loss(op, sensors, coords)
    assert torch.isfinite(loss)


def test_ks_residual_loss_fd_finite() -> None:
    op = _torch_op(jet_order=2)
    sensors = torch.randn(2, 8, dtype=DTYPE)
    coords = torch.randn(12, 2, dtype=DTYPE)
    loss = ks_residual_loss_fd(op, sensors, coords, h=1e-2)
    assert torch.isfinite(loss)


def test_ks_residual_loss_fd_parity_torch_jax() -> None:
    top = _torch_op(seed=5, jet_order=2)
    jop = JaxOp(
        spec=OperatorSpec(CS, COMPS, n_sensors=8, trunk_width=6),
        trunk=_copy_trunk(top.core.trunk),
        branch=_copy_branch(top.branch),
        shared_bias=None,
        jet_order=2,
    )
    rng = np.random.default_rng(6)
    sensors_np = rng.standard_normal((2, 8))
    coords_np = rng.standard_normal((6, 2))
    h = 1e-2
    t_loss = ks_residual_loss_fd(
        top,
        torch.tensor(sensors_np, dtype=DTYPE),
        torch.tensor(coords_np, dtype=DTYPE),
        h=h,
    )
    j_loss = jax_ks_residual_loss_fd(
        jop, jnp.asarray(sensors_np), jnp.asarray(coords_np), h=h
    )
    # 5-point / h^4 amplifies float64 round-off vs the closed-form path.
    np.testing.assert_allclose(
        float(t_loss.detach()), float(j_loss), rtol=1e-7, atol=1e-12
    )


def test_ks_fd_vs_closed_form_ushape_across_h() -> None:
    """FD residual gap vs closed form falls then rises across h (U-floor)."""
    op = _torch_op(seed=7, jet_order=4)
    sensors = torch.randn(1, 8, dtype=DTYPE)
    coords = torch.randn(8, 2, dtype=DTYPE)
    closed = float(ks_residual_loss(op, sensors, coords).detach())
    hs = (1e-1, 1e-2, 1e-3, 1e-4)
    gaps = []
    for h in hs:
        fd = float(ks_residual_loss_fd(op, sensors, coords, h=h).detach())
        gaps.append(abs(fd - closed))
    # Truncation-dominated: finer h reduces the gap at least once.
    assert gaps[1] < gaps[0], (hs, gaps)
    # Round-off upturn: finest h is worse than the floor.
    floor = min(gaps)
    assert gaps[-1] > floor, (hs, gaps)
    assert floor > 0.0


def test_make_ks_slab_finite_torch() -> None:
    slab = torch_make_ks_slab(
        n_samples=2,
        n_grid=64,
        n_sensors=16,
        n_modes=2,
        amplitude=0.3,
        t_final=0.5,
        n_times=5,
        seed=0,
    )
    assert slab.sensors.shape == (2, 16)
    assert torch.isfinite(slab.values).all()


def _copy_trunk(torch_trunk: TorchJetMLP) -> JetMLP:
    weights = []
    biases = []
    for lin in torch_trunk.linears:
        weights.append(jnp.asarray(lin.weight.detach().cpu().numpy()))
        biases.append(jnp.asarray(lin.bias.detach().cpu().numpy()))
    return JetMLP(
        weights=tuple(weights),
        biases=tuple(biases),
        spec=jax_get_activation(torch_trunk.spec.name),
        in_dim=torch_trunk.in_dim,
        out_dim=torch_trunk.out_dim,
    )


def _copy_branch(torch_branch) -> _BranchNet:
    weights = []
    biases = []
    for lin in torch_branch.linears:
        weights.append(jnp.asarray(lin.weight.detach().cpu().numpy()))
        biases.append(jnp.asarray(lin.bias.detach().cpu().numpy()))
    return _BranchNet(
        weights=tuple(weights),
        biases=tuple(biases),
        spec=jax_get_activation(torch_branch.spec.name),
        n_sensors=torch_branch.n_sensors,
        n_components=torch_branch.n_components,
        trunk_width=torch_branch.trunk_width,
        per_sample_bias=torch_branch.per_sample_bias,
    )


def test_ks_parity_torch_jax() -> None:
    top = _torch_op(seed=3)
    jop = JaxOp(
        spec=OperatorSpec(CS, COMPS, n_sensors=8, trunk_width=6),
        trunk=_copy_trunk(top.core.trunk),
        branch=_copy_branch(top.branch),
        shared_bias=None,
        jet_order=4,
    )
    rng = np.random.default_rng(4)
    sensors_np = rng.standard_normal((2, 8))
    coords_np = rng.standard_normal((4, 2))
    t_state = top.condition(torch.tensor(sensors_np, dtype=DTYPE)).on_grid(
        torch.tensor(coords_np, dtype=DTYPE)
    )
    j_state = jop.condition(jnp.asarray(sensors_np)).on_grid(jnp.asarray(coords_np))
    np.testing.assert_allclose(
        tops.derivative(t_state, "u", axis=0, order=4).detach().numpy(),
        np.asarray(jops.derivative(j_state, "u", axis=0, order=4)),
        rtol=RTOL,
        atol=0.0,
    )
    t_loss = ks_residual_loss(
        top,
        torch.tensor(sensors_np, dtype=DTYPE),
        torch.tensor(coords_np, dtype=DTYPE),
    )
    j_loss = jax_ks_residual_loss(
        jop, jnp.asarray(sensors_np), jnp.asarray(coords_np)
    )
    np.testing.assert_allclose(
        float(t_loss.detach()), float(j_loss), rtol=RTOL, atol=0.0
    )


def test_jax_ks_one_jet_cache() -> None:
    op = make_deeponet(
        coordinate_spec=CS,
        components=COMPS,
        n_sensors=8,
        trunk_width=6,
        trunk_hidden=10,
        trunk_depth=2,
        branch_hidden=10,
        branch_depth=2,
        jet_order=4,
        seed=0,
    )
    field = op.condition(jax.random.normal(jax.random.PRNGKey(0), (1, 8)))
    state = field(jax.random.normal(jax.random.PRNGKey(1), (5, 2)))
    from omnibias.pinn.jax.equations.kuramoto_sivashinsky import KuramotoSivashinsky

    _ = KuramotoSivashinsky()(state)
    assert sorted(state.extra[JAX_CACHE]) == [4]


def test_make_ks_slab_parity() -> None:
    t_slab = torch_make_ks_slab(
        n_samples=2,
        n_grid=64,
        n_sensors=16,
        n_modes=2,
        amplitude=0.3,
        t_final=0.5,
        n_times=5,
        seed=2,
    )
    j_slab = make_ks_slab(
        n_samples=2,
        n_grid=64,
        n_sensors=16,
        n_modes=2,
        amplitude=0.3,
        t_final=0.5,
        n_times=5,
        seed=2,
    )
    np.testing.assert_allclose(
        t_slab.values.detach().numpy(),
        np.asarray(j_slab.values),
        rtol=RTOL,
        atol=1e-12,
    )
