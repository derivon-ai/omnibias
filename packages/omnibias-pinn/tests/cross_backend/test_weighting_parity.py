# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Cross-backend parity for the adaptive-weighting primitives.

The state machine (:mod:`omnibias.pinn._core.weighting`) is shared pure Python,
so what needs proving is that the two *measurements* agree and that the
resulting weights are therefore identical -- and that the tensor primitive,
:func:`self_adaptive_loss`, matches to ``rtol=atol=1e-12`` like every other
loss helper.

The marching schedule is pure numpy and shared, so both backends re-export the
same objects; that identity is asserted here rather than re-tested.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

jax.config.update("jax_enable_x64", True)

from omnibias.pinn.jax import losses as jax_losses  # noqa: E402
from omnibias.pinn.torch import losses as torch_losses  # noqa: E402

MASKS = ["sigmoid", "softplus", "identity", "square", "tanh"]


def _pair(shape, seed):
    a = np.random.default_rng(seed).standard_normal(shape).astype(np.float64)
    return torch.from_numpy(a), jnp.asarray(a)


# ---------------- the measurement -----------------------------------


def _torch_terms(w: torch.Tensor, b: torch.Tensor, x: torch.Tensor):
    return {
        "pde": torch.mean(torch.tanh(w[None, :] * x[:, None] + b).sum(1) ** 2),
        "bc": torch.mean((torch.tanh(b) - 1.0) ** 2),
    }


def _jax_terms(x: jnp.ndarray):
    return {
        "pde": lambda p: jnp.mean(
            jnp.sum(jnp.tanh(p["w"][None, :] * x[:, None] + p["b"]), 1) ** 2
        ),
        "bc": lambda p: jnp.mean((jnp.tanh(p["b"]) - 1.0) ** 2),
    }


def test_grad_stats_parity():
    w_np = np.linspace(0.1, 0.8, 8)
    b_np = np.linspace(-0.3, 0.4, 8)
    x_np = np.linspace(0.0, 1.0, 16)

    wt = torch.tensor(w_np, dtype=torch.float64, requires_grad=True)
    bt = torch.tensor(b_np, dtype=torch.float64, requires_grad=True)
    st = torch_losses.grad_stats(
        _torch_terms(wt, bt, torch.tensor(x_np, dtype=torch.float64)), [wt, bt]
    )
    sj = jax_losses.grad_stats(
        _jax_terms(jnp.asarray(x_np)),
        {"w": jnp.asarray(w_np), "b": jnp.asarray(b_np)},
    )
    for key in ("pde", "bc"):
        assert st[key].max_abs == pytest.approx(sj[key].max_abs, rel=1e-12)
        assert st[key].mean_abs == pytest.approx(sj[key].mean_abs, rel=1e-12)


def test_ntk_trace_stats_parity():
    w_np = np.linspace(0.1, 0.8, 8)
    b_np = np.linspace(-0.3, 0.4, 8)
    x_np = np.linspace(0.0, 1.0, 16)

    wt = torch.tensor(w_np, dtype=torch.float64, requires_grad=True)
    bt = torch.tensor(b_np, dtype=torch.float64, requires_grad=True)
    tt = torch_losses.ntk_trace_stats(
        _torch_terms(wt, bt, torch.tensor(x_np, dtype=torch.float64)), [wt, bt]
    )
    tj = jax_losses.ntk_trace_stats(
        _jax_terms(jnp.asarray(x_np)),
        {"w": jnp.asarray(w_np), "b": jnp.asarray(b_np)},
    )
    for key in ("pde", "bc"):
        assert tt[key] == pytest.approx(tj[key], rel=1e-12)


def test_weights_agree_after_a_full_measure_and_update_cycle():
    """The end-to-end claim: same problem, same weights, either backend."""
    w_np = np.linspace(0.1, 0.8, 8)
    b_np = np.linspace(-0.3, 0.4, 8)
    x_np = np.linspace(0.0, 1.0, 16)

    wt = torch.tensor(w_np, dtype=torch.float64, requires_grad=True)
    bt = torch.tensor(b_np, dtype=torch.float64, requires_grad=True)
    xt = torch.tensor(x_np, dtype=torch.float64)
    jparams = {"w": jnp.asarray(w_np), "b": jnp.asarray(b_np)}
    jfns = _jax_terms(jnp.asarray(x_np))

    wt_weighter = torch_losses.GradNormWeighter(
        ["pde", "bc"], reference="pde", alpha=0.7, every=2
    )
    jx_weighter = jax_losses.GradNormWeighter(
        ["pde", "bc"], reference="pde", alpha=0.7, every=2
    )
    for _ in range(6):
        wt_weighter.update(torch_losses.grad_stats(_torch_terms(wt, bt, xt), [wt, bt]))
        jx_weighter.update(jax_losses.grad_stats(jfns, jparams))
    for key in ("pde", "bc"):
        assert wt_weighter[key] == pytest.approx(jx_weighter[key], rel=1e-12)


def test_ntk_weighter_parity():
    shared = {"a": 0.25, "b": 9.0, "c": 2.0}
    tw = torch_losses.NTKWeighter(list(shared), alpha=0.3)
    jw = jax_losses.NTKWeighter(list(shared), alpha=0.3)
    for _ in range(4):
        tw.update(shared)
        jw.update(shared)
    assert tw.weights == jw.weights  # shared state machine: bit-identical


# ---------------- the tensor primitive ------------------------------


@pytest.mark.parametrize("mask", MASKS)
@pytest.mark.parametrize("ascent", [True, False])
def test_self_adaptive_loss_parity(mask, ascent):
    rt, rj = _pair((32,), seed=5)
    lt, lj = _pair((32,), seed=6)
    got_t = torch_losses.self_adaptive_loss(rt, lt, mask=mask, ascent=ascent)
    got_j = jax_losses.self_adaptive_loss(rj, lj, mask=mask, ascent=ascent)
    assert float(got_t) == pytest.approx(float(got_j), rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("mask", MASKS)
def test_self_adaptive_gradient_parity(mask):
    rt, rj = _pair((16,), seed=7)
    lt, lj = _pair((16,), seed=8)
    lt = lt.clone().requires_grad_(True)
    (gt,) = torch.autograd.grad(
        torch_losses.self_adaptive_loss(rt, lt, mask=mask), lt
    )
    gj = jax.grad(lambda v: jax_losses.self_adaptive_loss(rj, v, mask=mask))(lj)
    assert np.allclose(gt.numpy(), np.asarray(gj), rtol=1e-12, atol=1e-12)


def test_self_adaptive_weights_object_parity():
    rt, rj = _pair((8,), seed=9)
    saw_t = torch_losses.SelfAdaptiveWeights(8, init=0.3, dtype=torch.float64)
    saw_j = jax_losses.make_self_adaptive_weights(8, init=0.3)
    assert np.allclose(
        saw_t.attention().numpy(), np.asarray(saw_j.attention()), rtol=1e-12
    )
    assert float(saw_t(rt).detach()) == pytest.approx(
        float(saw_j.loss(rj)), rel=1e-12
    )


def test_reverse_gradient_parity():
    xt, xj = _pair((12,), seed=10)
    assert np.allclose(
        torch_losses.reverse_gradient(xt).numpy(),
        np.asarray(jax_losses.reverse_gradient(xj)),
        rtol=0.0,
        atol=0.0,
    )


# ---------------- the shared marching surface -----------------------


def test_both_backends_reexport_the_same_marching_objects():
    assert torch_losses.TimeWindowSchedule is jax_losses.TimeWindowSchedule
    assert torch_losses.TimeMarcher is jax_losses.TimeMarcher
    assert torch_losses.window_points is jax_losses.window_points
    assert torch_losses.slice_points is jax_losses.slice_points


def test_both_backends_reexport_the_same_weighters():
    assert torch_losses.LossWeighter is jax_losses.LossWeighter
    assert torch_losses.GradNormWeighter is jax_losses.GradNormWeighter
    assert torch_losses.NTKWeighter is jax_losses.NTKWeighter
    assert torch_losses.ConstantWeighter is jax_losses.ConstantWeighter
    assert torch_losses.GradStats is jax_losses.GradStats
