# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Plateau-triggered K-growth scheduler.

Watches a scalar metric (typically validation loss); when it has not improved
by ``min_delta`` for ``patience`` consecutive epochs, the scheduler calls
``grow(strategy)`` on every :class:`GrowableOperatorMultiBiasUnit` in the
model. After growth, an LR-boost window is opened so freshly-activated bias
columns can warm up faster than the rest of the network.

Usage::

    sched = KGrowthScheduler(
        model,
        patience=10,
        min_delta=1e-3,
        max_K=8,
        strategy="pair",
        lr_boost_factor=10.0,
        lr_boost_epochs=5,
    )
    base_lr = optimizer.param_groups[0]["lr"]

    for epoch in range(num_epochs):
        train_one_epoch(...)
        val_loss = evaluate(...)
        boost = sched.step(val_loss, epoch=epoch)
        for pg in optimizer.param_groups:
            pg["lr"] = base_lr * boost
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

from omnibias.torch.growable import GrowableOperatorMultiBiasUnit, GrowStrategy

import torch
import torch.nn as nn

_LOGGER = logging.getLogger(__name__)

# Type alias: an anchor provider returns the bias value(s) at which the new
# pair should be anchored. Receives the unit's name + the unit itself, so it
# can inspect existing biases to decide where to place the new pair.
AnchorProvider = Callable[[str, GrowableOperatorMultiBiasUnit], "float | torch.Tensor | None"]


@dataclass
class GrowthEvent:
    """One growth event recorded by the scheduler."""

    epoch: int
    module_name: str
    new_active_K: int
    strategy: str
    anchor: float | None = None


@dataclass
class KGrowthScheduler:
    """Plateau-triggered K-growth controller.

    Parameters
    ----------
    model : nn.Module
        Any module containing :class:`GrowableOperatorMultiBiasUnit` children.
    patience : int, default 10
        Epochs of no-improvement before growth is triggered.
    min_delta : float, default 1e-3
        Minimum *relative* improvement counted as progress
        (loss must drop below ``best * (1 - min_delta)`` to reset patience).
    max_K : int, default 8
        Cap on each unit's ``active_K``. The scheduler stops growing units
        that have reached this cap.
    strategy : ``"pair"`` (default) or ``"saturate"``
    lr_boost_factor : float, default 10.0
        Multiplicative LR boost applied for ``lr_boost_epochs`` after each
        growth event. The training loop is responsible for reading the
        scheduler's ``current_lr_multiplier`` and applying it to the
        optimizer's ``lr``.
    lr_boost_epochs : int, default 5
    cooldown_epochs : int, default 5
        Minimum epochs between successive growth events; gives the new
        biases time to settle before the next plateau trigger fires.
    """

    model: nn.Module
    patience: int = 10
    min_delta: float = 1e-3
    max_K: int = 8
    strategy: GrowStrategy = "pair"
    lr_boost_factor: float = 10.0
    lr_boost_epochs: int = 5
    cooldown_epochs: int = 5
    anchor_provider: AnchorProvider | None = None
    """Optional callable invoked at every growth event to decide where the
    new pair lands. Signature: ``(module_name, growable_unit) -> float |
    Tensor | None``. Returning ``None`` falls back to anchoring at the
    unit's first existing bias (the default ``grow()`` behaviour).
    Recommended pattern: sample a value from the empirical distribution of
    pre-activations seen by ``growable_unit`` so the new pair lands inside
    the activation's informative range (avoiding the "dead-tail" failure
    mode where new biases drift to where ``sigma'`` vanishes)."""

    # Internal state.
    best_loss: float = float("inf")
    epochs_since_improve: int = 0
    cooldown_remaining: int = 0
    boost_remaining: int = 0
    growth_history: list[GrowthEvent] = field(default_factory=list)
    _initialized: bool = False

    def __post_init__(self) -> None:
        # Validate that the model has at least one growable unit.
        units = self._growable_units()
        if not units:
            raise ValueError(
                "KGrowthScheduler: the model contains no GrowableOperatorMultiBiasUnit "
                "children; nothing to grow."
            )

    # ----- public API ------------------------------------------------------

    @property
    def current_lr_multiplier(self) -> float:
        """The multiplier the training loop should apply to the base LR."""
        return self.lr_boost_factor if self.boost_remaining > 0 else 1.0

    def step(self, metric: float, *, epoch: int) -> float:
        """Update internal state with the latest metric value (lower-is-better).

        Returns the LR multiplier the training loop should apply for the
        *next* epoch. The multiplier equals ``lr_boost_factor`` for
        ``lr_boost_epochs`` epochs after a growth event, and ``1.0``
        otherwise.

        Patience semantics: the very first call establishes the baseline
        (counts neither as improvement nor as plateau). From the second
        call onwards, ``epochs_since_improve`` is incremented unless the
        metric drops below ``best_loss * (1 - min_delta)``.
        """
        # Decrement boost first; the multiplier we return at the end of
        # this call applies to the *next* training step.
        if self.boost_remaining > 0:
            self.boost_remaining -= 1

        if not self._initialized:
            self.best_loss = metric
            self.epochs_since_improve = 0
            self._initialized = True
        else:
            improved = metric < self.best_loss * (1.0 - self.min_delta)
            if improved:
                self.best_loss = metric
                self.epochs_since_improve = 0
            else:
                self.epochs_since_improve += 1

        # Cooldown blocks growth while still positive; decrement happens
        # *after* the readiness check so a fresh grow with cooldown=N
        # blocks exactly the next N step() calls.
        ready = (
            self.epochs_since_improve >= self.patience
            and self.cooldown_remaining == 0
            and self._has_pair_capacity_or_saturate_capacity()
        )
        grew_this_step = False
        if ready:
            grew_this_step = self._grow_all_eligible(epoch=epoch)
            if grew_this_step:
                self.epochs_since_improve = 0
                self.boost_remaining = self.lr_boost_epochs
                self.cooldown_remaining = self.cooldown_epochs

        if not grew_this_step and self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1

        return self.current_lr_multiplier

    def reset_best(self) -> None:
        """Forget the best-so-far loss; useful after dataset switch."""
        self.best_loss = float("inf")
        self.epochs_since_improve = 0
        self._initialized = False

    # ----- introspection ---------------------------------------------------

    def active_K_summary(self) -> dict[str, int]:
        """Return ``{module_name: active_K}`` for all growable units."""
        return {name: mod.active_K for name, mod in self._growable_units()}

    def total_growth_events(self) -> int:
        return len(self.growth_history)

    # ----- internals -------------------------------------------------------

    def _growable_units(self) -> list[tuple[str, GrowableOperatorMultiBiasUnit]]:
        return [
            (name, mod)
            for name, mod in self.model.named_modules()
            if isinstance(mod, GrowableOperatorMultiBiasUnit)
        ]

    def _strategy_increment(self) -> int:
        return 2 if self.strategy == "pair" else 1

    def _has_pair_capacity_or_saturate_capacity(self) -> bool:
        """At least one unit can still grow under this scheduler's max_K
        (and per-unit K_max) without overshooting."""
        inc = self._strategy_increment()
        for _, mod in self._growable_units():
            if mod.active_K + inc <= min(self.max_K, mod.K_max):
                return True
        return False

    def _grow_all_eligible(self, *, epoch: int) -> bool:
        inc = self._strategy_increment()
        grew_any = False
        for name, mod in self._growable_units():
            if mod.active_K + inc > min(self.max_K, mod.K_max):
                continue
            anchor: float | torch.Tensor | None = None
            if self.anchor_provider is not None and self.strategy == "pair":
                try:
                    anchor = self.anchor_provider(name, mod)
                except Exception as exc:  # noqa: BLE001
                    # An anchor provider failure should not abort training;
                    # log and fall back to the default anchor (anchor=None).
                    _LOGGER.warning(
                        "anchor_provider failed for %r: %r; falling back to "
                        "default anchor.",
                        name,
                        exc,
                    )
                    anchor = None
            try:
                if anchor is not None:
                    mod.grow(strategy=self.strategy, anchor_value=anchor)
                else:
                    mod.grow(strategy=self.strategy)
            except (RuntimeError, ValueError):
                # Hit per-unit K_max during grow, or strategy not supported
                # for this activation. Skip the unit; do not abort.
                continue
            anchor_scalar: float | None = None
            if isinstance(anchor, torch.Tensor):
                anchor_scalar = float(anchor.float().mean().item())
            elif anchor is not None:
                anchor_scalar = float(anchor)
            self.growth_history.append(
                GrowthEvent(
                    epoch=epoch,
                    module_name=name,
                    new_active_K=mod.active_K,
                    strategy=self.strategy,
                    anchor=anchor_scalar,
                )
            )
            grew_any = True
        return grew_any


__all__ = ["AnchorProvider", "GrowthEvent", "KGrowthScheduler"]
