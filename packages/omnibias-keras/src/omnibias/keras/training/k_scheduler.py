# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Plateau-triggered K-growth scheduler (Keras backend).

Watches a scalar metric; when it has not improved by ``min_delta`` for
``patience`` consecutive epochs, calls ``grow(strategy)`` on every
:class:`GrowableOperatorMultiBiasUnit` reachable from the model. After
growth, an LR-boost window is opened. Mirrors
:class:`omnibias.torch.training.k_scheduler.KGrowthScheduler`; the
training loop is responsible for reading ``current_lr_multiplier``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from omnibias.keras.growable import GrowableOperatorMultiBiasUnit, GrowStrategy

from keras import layers

_LOGGER = logging.getLogger(__name__)

AnchorProvider = Callable[
    [str, GrowableOperatorMultiBiasUnit], "float | np.ndarray | None"
]


def _iter_sublayers(
    layer: layers.Layer, prefix: str = "", _seen: set[int] | None = None
) -> Iterator[tuple[str, layers.Layer]]:
    """Recursively yield ``(name, sublayer)`` for all tracked sub-layers."""
    if _seen is None:
        _seen = set()
    for attr_name, attr in list(vars(layer).items()):
        candidates: list[tuple[str, Any]] = []
        if isinstance(attr, layers.Layer):
            candidates.append((f"{prefix}{attr_name}", attr))
        elif isinstance(attr, list | tuple):
            for i, item in enumerate(attr):
                if isinstance(item, layers.Layer):
                    candidates.append((f"{prefix}{attr_name}.{i}", item))
        for name, sub in candidates:
            if id(sub) in _seen:
                continue
            _seen.add(id(sub))
            yield name, sub
            yield from _iter_sublayers(sub, prefix=f"{name}.", _seen=_seen)


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
    """Plateau-triggered K-growth controller (Keras backend).

    Parameters mirror the torch scheduler: ``patience``, ``min_delta``,
    ``max_K``, ``strategy``, ``lr_boost_factor``, ``lr_boost_epochs``,
    ``cooldown_epochs``, and an optional ``anchor_provider``.
    """

    model: layers.Layer
    patience: int = 10
    min_delta: float = 1e-3
    max_K: int = 8
    strategy: GrowStrategy = "pair"
    lr_boost_factor: float = 10.0
    lr_boost_epochs: int = 5
    cooldown_epochs: int = 5
    anchor_provider: AnchorProvider | None = None

    best_loss: float = float("inf")
    epochs_since_improve: int = 0
    cooldown_remaining: int = 0
    boost_remaining: int = 0
    growth_history: list[GrowthEvent] = field(default_factory=list)
    _initialized: bool = False

    def __post_init__(self) -> None:
        if not self._growable_units():
            raise ValueError(
                "KGrowthScheduler: the model contains no "
                "GrowableOperatorMultiBiasUnit children; nothing to grow."
            )

    @property
    def current_lr_multiplier(self) -> float:
        return self.lr_boost_factor if self.boost_remaining > 0 else 1.0

    def step(self, metric: float, *, epoch: int) -> float:
        """Update with the latest metric; return the LR multiplier to apply next."""
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

        ready = (
            self.epochs_since_improve >= self.patience
            and self.cooldown_remaining == 0
            and self._has_capacity()
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
        self.best_loss = float("inf")
        self.epochs_since_improve = 0
        self._initialized = False

    def active_K_summary(self) -> dict[str, int]:
        return {name: mod.active_K for name, mod in self._growable_units()}

    def total_growth_events(self) -> int:
        return len(self.growth_history)

    # ----- internals -------------------------------------------------------

    def _growable_units(self) -> list[tuple[str, GrowableOperatorMultiBiasUnit]]:
        return [
            (name, mod)
            for name, mod in _iter_sublayers(self.model)
            if isinstance(mod, GrowableOperatorMultiBiasUnit)
        ]

    def _strategy_increment(self) -> int:
        return 2 if self.strategy == "pair" else 1

    def _has_capacity(self) -> bool:
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
            anchor: float | np.ndarray | None = None
            if self.anchor_provider is not None and self.strategy == "pair":
                try:
                    anchor = self.anchor_provider(name, mod)
                except Exception as exc:  # noqa: BLE001
                    # An anchor provider failure must not abort training; log and
                    # fall back to the default anchor (anchor=None).
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
                continue
            anchor_scalar: float | None = None
            if isinstance(anchor, np.ndarray):
                anchor_scalar = float(anchor.mean())
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
