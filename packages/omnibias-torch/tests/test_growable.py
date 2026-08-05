# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Tests for :class:`GrowableOperatorMultiBiasUnit` and
:class:`KGrowthScheduler`."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from omnibias.torch import GrowableOperatorMultiBiasUnit
from omnibias.torch.activations import list_activations
from omnibias.torch.growable import _SATURATE_FRIENDLY
from omnibias.torch.training import KGrowthScheduler

# ----- construction & forward -----------------------------------------------


def test_init_K_equal_to_one_matches_base_activation() -> None:
    """A freshly-instantiated K=1 unit equals ``sigma(z + init_bias)``."""
    z = torch.linspace(-2, 2, 16).unsqueeze(-1).expand(16, 4).contiguous()
    g = GrowableOperatorMultiBiasUnit(num_channels=4, init_K=1, K_max=8, base="sigmoid")
    assert g.active_K == 1
    assert torch.allclose(g(z), torch.sigmoid(z))


def test_default_init_K_is_one() -> None:
    g = GrowableOperatorMultiBiasUnit(num_channels=2, K_max=4, base="tanh")
    assert g.active_K == 1


def test_init_K_above_K_max_raises() -> None:
    with pytest.raises(ValueError):
        GrowableOperatorMultiBiasUnit(num_channels=2, init_K=5, K_max=4)


def test_invalid_K_max_raises() -> None:
    with pytest.raises(ValueError):
        GrowableOperatorMultiBiasUnit(num_channels=2, init_K=1, K_max=0)


# ----- pair-growth no-degradation invariant ---------------------------------


@pytest.mark.parametrize("name", list_activations())
def test_pair_grow_preserves_output_for_every_activation(name: str) -> None:
    """For *every* registered activation, ``grow(strategy='pair')`` leaves
    the unit's literal forward output bit-identically (or to float
    epsilon) unchanged at the moment of growth."""
    z = torch.randn(8, 3)
    g = GrowableOperatorMultiBiasUnit(num_channels=3, init_K=1, K_max=8, base=name)
    y_pre = g(z).detach().clone()
    g.grow(strategy="pair")
    assert g.active_K == 3
    y_post = g(z)
    assert torch.allclose(y_pre, y_post, atol=1e-6, rtol=1e-6), (
        f"{name}: max diff after pair grow = {(y_post - y_pre).abs().max():.2e}"
    )


def test_pair_grow_can_be_repeated() -> None:
    z = torch.randn(4, 2)
    g = GrowableOperatorMultiBiasUnit(num_channels=2, init_K=1, K_max=7, base="sigmoid")
    y0 = g(z).detach().clone()
    g.grow("pair")
    g.grow("pair")
    g.grow("pair")
    assert g.active_K == 7
    assert torch.allclose(y0, g(z), atol=1e-6)


def test_pair_grow_exceeding_K_max_raises() -> None:
    g = GrowableOperatorMultiBiasUnit(num_channels=2, init_K=1, K_max=2, base="sigmoid")
    with pytest.raises(RuntimeError):
        g.grow("pair")  # 1 + 2 > 2


# ----- saturate-growth invariant -------------------------------------------


@pytest.mark.parametrize("name", sorted(_SATURATE_FRIENDLY))
def test_saturate_grow_preserves_output_for_friendly_activations(name: str) -> None:
    """For activations that vanish at ``-infty``, ``grow('saturate')``
    leaves the output unchanged to float-epsilon tolerance."""
    z = torch.randn(8, 3)
    g = GrowableOperatorMultiBiasUnit(
        num_channels=3, init_K=1, K_max=8, base=name, saturate_big=30.0
    )
    y_pre = g(z).detach().clone()
    g.grow(strategy="saturate")
    assert g.active_K == 2
    y_post = g(z)
    # For exp the saturate-add contributes exp(-30) ~ 1e-13; for sigmoid /
    # softplus / gaussian / relu / huber it is 0 to machine epsilon.
    assert torch.allclose(y_pre, y_post, atol=1e-6, rtol=1e-6), (
        f"{name}: max diff after saturate grow = {(y_post - y_pre).abs().max():.2e}"
    )


def test_saturate_grow_rejected_for_unfriendly_activations() -> None:
    for name in ("tanh", "arctan", "log1pu2", "silu", "gelu"):
        g = GrowableOperatorMultiBiasUnit(num_channels=2, init_K=1, K_max=4, base=name)
        with pytest.raises(ValueError):
            g.grow(strategy="saturate")


def test_saturate_grow_exceeding_K_max_raises() -> None:
    g = GrowableOperatorMultiBiasUnit(num_channels=2, init_K=1, K_max=1, base="sigmoid")
    with pytest.raises(RuntimeError):
        g.grow("saturate")


# ----- frozen-sign growth is refused, not silently dead ---------------------


def test_grow_pair_rejected_when_signs_frozen() -> None:
    """``grow('pair')`` needs trainable signs; a ``learnable_signs=False`` unit
    must refuse rather than silently inject a frozen ``(+eta, -eta)`` column
    that can never train."""
    g = GrowableOperatorMultiBiasUnit(
        num_channels=2, init_K=1, K_max=4, base="sigmoid", learnable_signs=False
    )
    with pytest.raises(ValueError, match="learnable_signs"):
        g.grow("pair")
    assert g.active_K == 1  # growth aborted, state unchanged


def test_grow_saturate_rejected_when_signs_frozen() -> None:
    """``grow('saturate')`` with frozen signs would leave a permanently-zero
    (dead) column; it must raise. A saturate-friendly base is used so the only
    failure cause is the frozen signs."""
    g = GrowableOperatorMultiBiasUnit(
        num_channels=2, init_K=1, K_max=4, base="sigmoid", learnable_signs=False
    )
    with pytest.raises(ValueError, match="learnable_signs"):
        g.grow("saturate")
    assert g.active_K == 1  # growth aborted, state unchanged


# ----- gradient flow to newly-activated columns ----------------------------


def test_pair_grow_unlocks_gradient_on_new_columns() -> None:
    """After ``grow('pair')`` the new bias and sign columns must receive
    non-zero gradient on a non-trivial loss."""
    z = torch.randn(8, 3)
    g = GrowableOperatorMultiBiasUnit(num_channels=3, init_K=1, K_max=5, base="sigmoid")
    g.grow("pair")
    out = g(z)
    target = torch.zeros_like(out)
    loss = (out - target).pow(2).sum()
    loss.backward()
    # Newly-active columns are 1 and 2.
    assert g.biases.grad[:, 1].abs().sum() > 0
    assert g.biases.grad[:, 2].abs().sum() > 0
    assert g.signs.grad[:, 1].abs().sum() > 0
    assert g.signs.grad[:, 2].abs().sum() > 0
    # Reserve columns (3, 4) must stay at zero gradient.
    assert g.biases.grad[:, 3].abs().sum() == 0
    assert g.biases.grad[:, 4].abs().sum() == 0


def test_inactive_columns_stay_at_init_after_optim_step() -> None:
    """Reserve columns receive no gradient -> Adam keeps them frozen."""
    g = GrowableOperatorMultiBiasUnit(
        num_channels=4, init_K=1, K_max=6, base="sigmoid", init_bias=0.5
    )
    optim = torch.optim.Adam(g.parameters(), lr=0.1)
    z = torch.randn(8, 4)
    initial_reserve = g.biases.detach()[:, 1:].clone()
    for _ in range(20):
        optim.zero_grad()
        loss = (g(z) - 0.0).pow(2).mean()
        loss.backward()
        optim.step()
    final_reserve = g.biases.detach()[:, 1:]
    assert torch.equal(initial_reserve, final_reserve), (
        "Reserve bias columns should not move while inactive."
    )


# ----- KGrowthScheduler ---------------------------------------------------


def _toy_growable_model(K_max: int = 6) -> nn.Module:
    class _M(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 8)
            self.act = GrowableOperatorMultiBiasUnit(
                num_channels=8, init_K=1, K_max=K_max, base="sigmoid"
            )
            self.head = nn.Linear(8, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.head(self.act(self.lin(x)))

    return _M()


def test_scheduler_rejects_models_without_growable_units() -> None:
    plain = nn.Linear(2, 2)
    with pytest.raises(ValueError):
        KGrowthScheduler(plain)


def test_scheduler_grows_after_patience_window() -> None:
    m = _toy_growable_model(K_max=6)
    sched = KGrowthScheduler(
        m, patience=3, max_K=6, strategy="pair", lr_boost_epochs=2, cooldown_epochs=0
    )
    # First step establishes baseline; then 3 plateau steps -> grow on the
    # third plateau step (4th call total).
    sched.step(1.0, epoch=0)  # baseline
    for ep in range(1, 4):
        mult = sched.step(1.0, epoch=ep)
        if ep < 3:
            assert mult == 1.0, f"epoch {ep}: pre-grow mult should be 1.0"
            assert m.act.active_K == 1
    assert m.act.active_K == 3, "pair grow should add 2 columns at patience=3"
    assert sched.total_growth_events() == 1
    assert sched.current_lr_multiplier == 10.0


def test_scheduler_resets_patience_on_improvement() -> None:
    m = _toy_growable_model(K_max=6)
    sched = KGrowthScheduler(m, patience=3, max_K=6, strategy="pair")
    sched.step(1.0, epoch=0)  # baseline
    sched.step(1.0, epoch=1)  # plateau 1
    sched.step(1.0, epoch=2)  # plateau 2
    sched.step(0.5, epoch=3)  # improvement -> reset
    sched.step(0.5, epoch=4)
    sched.step(0.5, epoch=5)
    sched.step(0.5, epoch=6)  # 3 consecutive plateau steps at 0.5 -> grow
    assert m.act.active_K == 3, "should grow only after the second plateau"
    assert sched.total_growth_events() == 1


def test_scheduler_respects_max_K_cap() -> None:
    m = _toy_growable_model(K_max=6)
    sched = KGrowthScheduler(
        m, patience=1, max_K=4, strategy="pair", lr_boost_epochs=0, cooldown_epochs=0
    )
    for ep in range(20):
        sched.step(1.0, epoch=ep)
    assert m.act.active_K <= 4, f"max_K=4 cap violated; active_K={m.act.active_K}"


def test_scheduler_lr_boost_decays() -> None:
    m = _toy_growable_model(K_max=6)
    sched = KGrowthScheduler(
        m,
        patience=1,
        max_K=6,
        strategy="pair",
        lr_boost_factor=10.0,
        lr_boost_epochs=2,
        cooldown_epochs=10,  # large to prevent re-trigger
    )
    # First call establishes baseline; growth fires on the second call.
    sched.step(1.0, epoch=0)
    assert sched.current_lr_multiplier == 1.0
    sched.step(1.0, epoch=1)  # patience=1 met -> grow + boost on
    assert sched.current_lr_multiplier == 10.0
    sched.step(1.0, epoch=2)  # decrement (boost_remaining: 2 -> 1)
    assert sched.current_lr_multiplier == 10.0
    sched.step(1.0, epoch=3)  # decrement (1 -> 0)
    assert sched.current_lr_multiplier == 1.0


def test_scheduler_cooldown_blocks_immediate_regrowth() -> None:
    m = _toy_growable_model(K_max=8)
    sched = KGrowthScheduler(
        m, patience=1, max_K=8, strategy="pair", cooldown_epochs=5, lr_boost_epochs=0
    )
    sched.step(1.0, epoch=0)  # baseline
    sched.step(1.0, epoch=1)  # patience=1 met -> grow (active_K=1 -> 3)
    K_after_first_grow = m.act.active_K
    assert K_after_first_grow == 3
    for ep in range(2, 7):
        sched.step(1.0, epoch=ep)  # cooldown should block growth
    assert m.act.active_K == K_after_first_grow, "cooldown failed"
    sched.step(1.0, epoch=7)  # cooldown elapsed -> grow again
    assert m.act.active_K > K_after_first_grow


# ----- anchor_value -------------------------------------------------------


def test_grow_pair_with_scalar_anchor_value() -> None:
    """grow(anchor_value=...) uses the user-supplied location, not bias[0]."""
    g = GrowableOperatorMultiBiasUnit(num_channels=3, init_K=1, K_max=5, base="sigmoid")
    g.grow(strategy="pair", anchor_value=0.7)
    assert g.active_K == 3
    assert torch.allclose(g.biases[:, 1], torch.tensor(0.7))
    assert torch.allclose(g.biases[:, 2], torch.tensor(0.7))
    # Sign sums to zero on the new pair => Lemma-1 invariance preserved.
    assert torch.allclose(g.signs[:, 1] + g.signs[:, 2], torch.tensor(0.0))


def test_grow_pair_with_tensor_anchor_value() -> None:
    """A length-num_channels anchor tensor places per-channel anchors."""
    g = GrowableOperatorMultiBiasUnit(num_channels=4, init_K=1, K_max=5, base="sigmoid")
    anchors = torch.tensor([-1.0, 0.0, 0.5, 2.0])
    g.grow(strategy="pair", anchor_value=anchors)
    assert g.active_K == 3
    assert torch.allclose(g.biases[:, 1], anchors)
    assert torch.allclose(g.biases[:, 2], anchors)


def test_grow_pair_with_anchor_preserves_bit_identical_output() -> None:
    """Lemma-1: grow with any anchor_value leaves the output unchanged."""
    g = GrowableOperatorMultiBiasUnit(num_channels=3, init_K=1, K_max=5, base="tanh")
    z = torch.linspace(-2, 2, 16).unsqueeze(-1).expand(16, 3).contiguous()
    out_before = g(z).clone()
    g.grow(strategy="pair", anchor_value=1.5)
    out_after = g(z)
    assert torch.allclose(out_before, out_after, atol=1e-6)


def test_grow_pair_anchor_tensor_shape_validation() -> None:
    """Mismatched anchor tensor shape must raise ValueError."""
    g = GrowableOperatorMultiBiasUnit(num_channels=3, init_K=1, K_max=5, base="sigmoid")
    with pytest.raises(ValueError, match="shape"):
        g.grow(strategy="pair", anchor_value=torch.tensor([0.0, 1.0]))


def test_scheduler_anchor_provider_invoked_on_growth() -> None:
    """The scheduler calls anchor_provider for each growable unit at growth."""
    m = _toy_growable_model(K_max=6)
    calls: list[tuple[str, int]] = []

    def provider(name, unit):
        calls.append((name, unit.active_K))
        return 0.42

    sched = KGrowthScheduler(
        m,
        patience=1,
        max_K=6,
        strategy="pair",
        cooldown_epochs=0,
        anchor_provider=provider,
    )
    sched.step(1.0, epoch=0)
    sched.step(1.0, epoch=1)  # triggers growth
    assert calls, "anchor_provider was never invoked"
    assert calls[0][0] == "act"
    # The anchor we returned should have been written to the new pair columns.
    assert torch.allclose(m.act.biases[:, 1], torch.tensor(0.42))
    assert torch.allclose(m.act.biases[:, 2], torch.tensor(0.42))


def test_scheduler_anchor_provider_failure_falls_back() -> None:
    """A raising anchor_provider must not abort training; falls back to default."""
    m = _toy_growable_model(K_max=6)

    def bad_provider(_n, _u):
        raise RuntimeError("synthetic failure")

    sched = KGrowthScheduler(
        m,
        patience=1,
        max_K=6,
        strategy="pair",
        cooldown_epochs=0,
        anchor_provider=bad_provider,
    )
    sched.step(1.0, epoch=0)
    sched.step(1.0, epoch=1)  # would trigger growth; anchor_provider raises
    # Growth still happened with default anchor (= bias[:, 0]).
    assert m.act.active_K == 3
