# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Unit tests for the torch adaptive-weighting primitives.

Covers:

* :func:`grad_stats`: agreement with a hand-computed reference, the
  unused-parameter-counts-as-zero rule, and scalar / non-empty validation.
* :func:`ntk_trace_stats`: agreement with ``estimate_ntk_trace`` per term.
* The measurement + state-machine loop: a :class:`GradNormWeighter` driven by
  real gradients raises the weight of the weaker term and equalises the two
  terms' weighted gradient scales.
* :func:`reverse_gradient`: exact identity forward, exactly negated backward.
* :func:`self_adaptive_loss` / :class:`SelfAdaptiveWeights`: the masked-mean
  value, ascent on the weights and descent on the network from one backward,
  mask selection, and the qualitative property the method exists for -- badly
  fitted points grow their own attention.
"""

from __future__ import annotations

import pytest
import torch
from omnibias.pinn.torch.losses import (
    GradNormWeighter,
    SelfAdaptiveWeights,
    estimate_ntk_trace,
    grad_stats,
    ntk_trace_stats,
    reverse_gradient,
    self_adaptive_loss,
)

DTYPE = torch.float64


@pytest.fixture
def net() -> torch.nn.Module:
    torch.manual_seed(0)
    return torch.nn.Sequential(
        torch.nn.Linear(1, 8, dtype=DTYPE),
        torch.nn.Tanh(),
        torch.nn.Linear(8, 1, dtype=DTYPE),
    )


@pytest.fixture
def x() -> torch.Tensor:
    return torch.linspace(0.0, 1.0, 16, dtype=DTYPE).reshape(-1, 1)


# ---------------- grad_stats ----------------------------------------


def test_grad_stats_match_a_hand_computed_reference(net, x):
    loss = (net(x) ** 2).mean()
    got = grad_stats({"pde": loss}, net.parameters())

    params = list(net.parameters())
    grads = torch.autograd.grad(loss, params)
    flat = torch.cat([g.reshape(-1) for g in grads]).abs()
    assert got["pde"].max_abs == pytest.approx(float(flat.max()), rel=1e-14)
    assert got["pde"].mean_abs == pytest.approx(float(flat.mean()), rel=1e-14)


def test_unused_parameters_count_as_zeros_in_the_mean(net, x):
    """mean_theta is over the whole network, not over the terms it touches."""
    extra = torch.nn.Parameter(torch.zeros(1000, dtype=DTYPE))
    loss = (net(x) ** 2).mean()
    params = [*net.parameters(), extra]
    narrow = grad_stats({"a": loss}, net.parameters())["a"]
    wide = grad_stats({"a": loss}, params)["a"]
    assert wide.max_abs == pytest.approx(narrow.max_abs)
    assert wide.mean_abs < narrow.mean_abs


def test_grad_stats_are_non_negative_and_ordered(net, x):
    losses = {"pde": (net(x) ** 2).mean(), "bc": (net(x[:2]) - 1.0).pow(2).mean()}
    for stat in grad_stats(losses, net.parameters()).values():
        assert 0.0 <= stat.mean_abs <= stat.max_abs


def test_grad_stats_validate(net, x):
    with pytest.raises(ValueError, match="empty"):
        grad_stats({}, net.parameters())
    with pytest.raises(ValueError, match="requires_grad"):
        grad_stats({"a": (net(x) ** 2).mean()}, [])
    with pytest.raises(ValueError, match="scalar"):
        grad_stats({"a": net(x)}, net.parameters())


# ---------------- ntk_trace_stats -----------------------------------


def test_ntk_trace_stats_match_estimate_ntk_trace(net, x):
    losses = {"pde": (net(x) ** 2).mean(), "bc": (net(x[:2]) - 1.0).pow(2).mean()}
    got = ntk_trace_stats(losses, net.parameters())
    for name, loss in losses.items():
        want = float(estimate_ntk_trace(loss, list(net.parameters())))
        assert got[name] == pytest.approx(want, rel=1e-12)


def test_ntk_trace_stats_reject_empty(net):
    with pytest.raises(ValueError, match="empty"):
        ntk_trace_stats({}, net.parameters())


# ---------------- measurement + state machine -----------------------


def test_gradnorm_loop_equalises_the_weighted_gradient_scales(net, x):
    """The whole point: after weighting, the terms are on one scale."""
    losses = {
        "pde": (net(x) ** 2).mean(),
        "bc": 1e-4 * (net(x[:2]) - 1.0).pow(2).mean(),
    }
    stats = grad_stats(losses, net.parameters())
    assert stats["bc"].mean_abs < stats["pde"].mean_abs  # the pathology

    w = GradNormWeighter(["pde", "bc"], reference="pde", alpha=0.0)
    weights = w.update(stats)
    assert weights["bc"] > 1.0
    scaled_bc = weights["bc"] * stats["bc"].mean_abs
    assert scaled_bc == pytest.approx(stats["pde"].max_abs, rel=1e-6)


def test_weighter_combine_returns_a_differentiable_tensor(net, x):
    losses = {"pde": (net(x) ** 2).mean(), "bc": (net(x[:2]) - 1.0).pow(2).mean()}
    w = GradNormWeighter(["pde", "bc"], reference="pde")
    w.update(grad_stats(losses, net.parameters()))
    total = w.combine(losses)
    total.backward()
    assert all(p.grad is not None for p in net.parameters())


# ---------------- reverse_gradient ----------------------------------


def test_reverse_gradient_is_exactly_the_identity_forward():
    x = torch.tensor([-3.25, 0.0, 1e-17, 7.5], dtype=DTYPE)
    assert torch.equal(reverse_gradient(x), x)


def test_reverse_gradient_negates_the_gradient_exactly():
    x = torch.tensor([0.5, -2.0], dtype=DTYPE, requires_grad=True)
    reverse_gradient(x).pow(2).sum().backward()
    assert torch.equal(x.grad, -2.0 * x.detach())


# ---------------- self-adaptive weights -----------------------------


def test_self_adaptive_loss_is_the_masked_mean():
    r = torch.tensor([1.0, 2.0], dtype=DTYPE)
    lam = torch.tensor([0.0, 0.0], dtype=DTYPE)
    # sigmoid(0) = 1/2
    got = self_adaptive_loss(r, lam, mask="sigmoid", ascent=False)
    assert got == pytest.approx(0.5 * (1.0 + 4.0) / 2)


@pytest.mark.parametrize(
    ("mask", "expected"), [("identity", 2.0), ("square", 4.0), ("relu", 2.0)]
)
def test_masks_apply_the_named_map(mask, expected):
    r = torch.ones(1, dtype=DTYPE)
    lam = torch.full((1,), 2.0, dtype=DTYPE)
    got = self_adaptive_loss(r, lam, mask=mask, ascent=False)
    assert got == pytest.approx(expected)


def test_unknown_mask_is_rejected():
    r = torch.ones(2, dtype=DTYPE)
    with pytest.raises(ValueError, match="unknown mask"):
        self_adaptive_loss(r, torch.zeros(2, dtype=DTYPE), mask="not_an_activation")


def test_non_broadcastable_lambdas_rejected():
    with pytest.raises(ValueError, match="broadcast"):
        self_adaptive_loss(torch.ones(3, dtype=DTYPE), torch.ones(4, dtype=DTYPE))


def test_ascent_flips_only_the_weight_gradient(net, x):
    lam = torch.zeros(16, dtype=DTYPE, requires_grad=True)
    r = net(x)[:, 0]

    up = self_adaptive_loss(r, lam, ascent=True)
    (g_up,) = torch.autograd.grad(up, lam, retain_graph=True)
    down = self_adaptive_loss(r, lam, ascent=False)
    (g_down,) = torch.autograd.grad(down, lam, retain_graph=True)
    assert torch.equal(g_up, -g_down)
    assert float(up.detach()) == pytest.approx(float(down.detach()))


def test_one_backward_descends_theta_and_ascends_lambda(net, x):
    """The minimax, driven by a single ordinary optimiser."""
    saw = SelfAdaptiveWeights(16, dtype=DTYPE)
    opt = torch.optim.SGD([*net.parameters(), *saw.parameters()], lr=0.1)
    target = torch.ones(16, dtype=DTYPE)

    with torch.no_grad():
        before_loss = float(((net(x)[:, 0] - target) ** 2).mean())
    before_raw = saw.raw.detach().clone()
    for _ in range(20):
        opt.zero_grad()
        saw(net(x)[:, 0] - target).backward()
        opt.step()
    with torch.no_grad():
        after_loss = float(((net(x)[:, 0] - target) ** 2).mean())

    assert after_loss < before_loss  # theta descended
    assert float(saw.raw.detach().mean()) > float(before_raw.mean())  # lambda ascended


def test_attention_concentrates_on_the_badly_fitted_points():
    """The reason for pointwise weights: stiff regions weight themselves up."""
    saw = SelfAdaptiveWeights(4, dtype=DTYPE)
    opt = torch.optim.SGD(saw.parameters(), lr=1.0)
    residual = torch.tensor([0.01, 0.01, 1.0, 0.01], dtype=DTYPE)
    for _ in range(50):
        opt.zero_grad()
        saw(residual).backward()
        opt.step()
    attention = saw.attention()
    assert int(attention.argmax()) == 2
    assert float(attention[2]) > float(attention[0])


def test_self_adaptive_weights_start_uniform_and_validate():
    saw = SelfAdaptiveWeights(5, dtype=DTYPE)
    assert torch.allclose(saw.attention(), torch.full((5,), 0.5, dtype=DTYPE))
    with pytest.raises(ValueError, match="n_points"):
        SelfAdaptiveWeights(0)
    with pytest.raises(ValueError, match="weights"):
        saw(torch.ones(4, dtype=DTYPE))


def test_self_adaptive_weights_handle_vector_residuals():
    saw = SelfAdaptiveWeights(3, dtype=DTYPE)
    got = saw(torch.ones((3, 2), dtype=DTYPE))
    assert got.ndim == 0
    assert float(got.detach()) == pytest.approx(0.5)


def test_self_adaptive_weights_default_to_the_framework_dtype():
    saw = SelfAdaptiveWeights(3)
    assert saw.raw.dtype == torch.get_default_dtype()
