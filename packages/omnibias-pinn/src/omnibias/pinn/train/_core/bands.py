# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Spectral band scheduler: grow Fourier / Mscale bands from residual spectra.

Closes the chicken-and-egg loop where
:func:`~omnibias.pinn._core.multiscale.suggest_frequency_bands` needs a
solution sample you do not yet have -- during training, measure the
*residual*'s power spectrum instead and grow the band set.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from omnibias.pinn._core.multiscale import geometric_bands, suggest_frequency_bands


@dataclass
class SpectralBandScheduler:
    """Grow a frequency-band tuple from residual power spectra.

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
        Suggest a refresh every ``every`` calls to :meth:`observe`.
    min_scale
        Floor on each suggested scale.
    """

    n_bands_max: int = 4
    n_bands_init: int = 2
    L: float | tuple[float, ...] = 1.0
    every: int = 1
    min_scale: float = 1e-3
    bands: tuple[float, ...] = field(init=False)
    _calls: int = field(default=0, init=False, repr=False)

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

    def observe(self, residual_grid: np.ndarray) -> tuple[float, ...]:
        """Update bands from a residual sample on a uniform grid.

        ``residual_grid`` has shape ``(T, *spatial)`` (use ``T=1`` for
        steady). Returns the (possibly grown) band tuple.
        """
        self._calls += 1
        if self._calls % self.every != 0:
            return self.bands
        # Request up to n_bands_max; suggest_frequency_bands may return fewer.
        suggested = suggest_frequency_bands(
            np.asarray(residual_grid, dtype=float),
            L=self.L,
            n_bands=self.n_bands_max,
            min_scale=self.min_scale,
        )
        # Grow monotonically: keep existing bands, append new higher ones.
        merged: list[float] = list(self.bands)
        for s in suggested:
            if all(abs(s - m) / max(m, 1e-12) > 0.1 for m in merged):
                merged.append(float(s))
            if len(merged) >= self.n_bands_max:
                break
        merged = sorted(merged)[: self.n_bands_max]
        self.bands = tuple(merged)
        return self.bands


__all__ = ["SpectralBandScheduler"]
