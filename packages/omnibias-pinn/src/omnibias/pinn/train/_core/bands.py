# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Spectral band scheduler: grow Fourier / Mscale bands from residual spectra.

Closes the chicken-and-egg loop where
:func:`~omnibias.pinn._core.multiscale.suggest_frequency_bands` needs a
solution sample you do not yet have -- during training, measure the
*residual*'s power spectrum instead and grow the band set.

:class:`SpectralBandScheduler` is a *controller*: it observes residuals at
deterministic update points, merges suggested bands, and can apply them to
compatible torch / JAX fields via :meth:`apply_to`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from omnibias.pinn._core.multiscale import geometric_bands, suggest_frequency_bands


def _is_mscale_field(field: Any) -> bool:
    name = type(field).__name__
    return name == "MscaleVectorField"


def _is_fourier_field(field: Any) -> bool:
    name = type(field).__name__
    return name == "FourierFeatureVectorField"


def _apply_bands_torch(field: Any, bands: tuple[float, ...]) -> bool:
    """Hot-update band scales on a compatible torch field."""
    if _is_mscale_field(field):
        net = field.net
        n = len(net.scales)
        scales = tuple(float(s) for s in bands[:n])
        if tuple(net.scales) == scales:
            return False
        net.scales = scales
        return True
    if _is_fourier_field(field):
        net = field.net
        scales = tuple(float(s) for s in bands)
        if hasattr(net, "frequency_scale"):
            current = net.frequency_scale
            if isinstance(current, tuple) and current == scales:
                return False
            net.frequency_scale = scales
            return True
    return False


def _apply_bands_jax(field: Any, bands: tuple[float, ...]) -> Any:
    """Return a JAX field with updated band scales, or ``None`` if incompatible."""
    import dataclasses

    scales = tuple(float(s) for s in bands)
    if _is_mscale_field(field):
        if tuple(field.net.scales) == scales:
            return None
        new_net = dataclasses.replace(field.net, scales=scales)
        return dataclasses.replace(field, net=new_net)
    if _is_fourier_field(field):
        if tuple(field.net.frequency_scale) == scales:
            return None
        new_net = dataclasses.replace(field.net, frequency_scale=scales)
        return dataclasses.replace(field, net=new_net)
    return None


@dataclass
class SpectralBandScheduler:
    """Grow and apply frequency-band tuples from residual power spectra.

    Parameters
    ----------
    n_bands_max
        Hard cap on the number of bands.
    n_bands_init
        Initial band count (geometric ladder until the first measurement).
    L
        Domain period(s) forwarded to
        :func:`~omnibias.pinn._core.multiscale.suggest_frequency_bands`.
    every
        Suggest a refresh every ``every`` calls to :meth:`step` / :meth:`observe`.
    min_scale
        Floor on each suggested scale.
    update_steps
        Optional explicit global-step schedule; when set, overrides ``every``.
    """

    n_bands_max: int = 4
    n_bands_init: int = 2
    L: float | tuple[float, ...] = 1.0
    every: int = 1
    min_scale: float = 1e-3
    update_steps: tuple[int, ...] | None = None
    bands: tuple[float, ...] = field(init=False)
    _calls: int = field(default=0, init=False, repr=False)
    _updates: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.n_bands_max < 1:
            raise ValueError(f"n_bands_max must be >= 1, got {self.n_bands_max}")
        if self.n_bands_init < 1:
            raise ValueError(f"n_bands_init must be >= 1, got {self.n_bands_init}")
        if self.n_bands_init > self.n_bands_max:
            raise ValueError("n_bands_init must be <= n_bands_max")
        if self.every < 1:
            raise ValueError(f"every must be >= 1, got {self.every}")
        self.bands = geometric_bands(self.n_bands_init)

    def _should_update(self, step: int | None) -> bool:
        if self.update_steps is not None:
            if step is None:
                return False
            return int(step) in self.update_steps
        self._calls += 1
        return self._calls % self.every == 0

    def observe(
        self, residual_grid: np.ndarray, *, step: int | None = None
    ) -> tuple[float, ...]:
        """Update bands from a residual sample on a uniform grid."""
        if not self._should_update(step):
            return self.bands
        suggested = suggest_frequency_bands(
            np.asarray(residual_grid, dtype=float),
            L=self.L,
            n_bands=self.n_bands_max,
            min_scale=self.min_scale,
        )
        merged: list[float] = list(self.bands)
        for s in suggested:
            if all(abs(s - m) / max(m, 1e-12) > 0.1 for m in merged):
                merged.append(float(s))
            if len(merged) >= self.n_bands_max:
                break
        merged = sorted(merged)[: self.n_bands_max]
        if tuple(merged) != self.bands:
            self.bands = tuple(merged)
            self._updates += 1
        return self.bands

    def step(
        self,
        residual_grid: np.ndarray,
        field: Any | None = None,
        *,
        step: int | None = None,
        backend: str = "torch",
    ) -> tuple[float, ...]:
        """Observe residual bands and optionally apply them to ``field``."""
        old = self.bands
        bands = self.observe(residual_grid, step=step)
        if field is not None and bands != old:
            if backend == "jax":
                updated = _apply_bands_jax(field, bands)
                if updated is not None:
                    return bands
            else:
                self.apply_to(field, bands=bands)
        return bands

    def apply_to(self, field: Any, *, bands: tuple[float, ...] | None = None) -> bool:
        """Apply the current (or explicit) band tuple to a compatible field."""
        target = self.bands if bands is None else bands
        return _apply_bands_torch(field, target)

    def state_dict(self) -> dict[str, Any]:
        """Serialisable scheduler state for resume."""
        return {
            "bands": list(self.bands),
            "calls": self._calls,
            "updates": self._updates,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore scheduler state from :meth:`state_dict`."""
        self.bands = tuple(float(s) for s in state["bands"])
        self._calls = int(state.get("calls", 0))
        self._updates = int(state.get("updates", 0))


__all__ = ["SpectralBandScheduler"]
