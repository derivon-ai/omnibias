# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Sensor grids and deterministic truncated-Fourier initial-condition ensembles.

No Gaussian-random-field helper ships in the repo; operator-learning data is
built from a seeded truncated Fourier series on a periodic interval, which is
exact under the spectral method-of-lines reference already in
``omnibias.pinn.solver``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SensorGrid:
    """Fixed sensor locations at which an input function is observed.

    Parameters
    ----------
    points
        Sensor coordinates of shape ``(m,)`` for a 1-D spatial axis, or
        ``(m, d)`` for multi-D. Stored as float64.
    length
        Period of the underlying spatial domain (used by the Fourier IC
        sampler). Default ``2 pi``.
    """

    points: np.ndarray
    length: float = 2.0 * np.pi

    def __post_init__(self) -> None:
        pts = np.asarray(self.points, dtype=np.float64)
        if pts.ndim == 0:
            raise ValueError("points must be at least 1-D")
        if pts.size < 1:
            raise ValueError("SensorGrid needs at least one sensor")
        object.__setattr__(self, "points", pts)
        if self.length <= 0.0:
            raise ValueError(f"length must be > 0, got {self.length}")

    @property
    def n_sensors(self) -> int:
        return int(self.points.shape[0])

    @classmethod
    def uniform_1d(
        cls, n_sensors: int, length: float = 2.0 * np.pi
    ) -> SensorGrid:
        """``n_sensors`` equispaced points on ``[0, length)``."""
        if n_sensors < 1:
            raise ValueError(f"n_sensors must be >= 1, got {n_sensors}")
        pts = (length / n_sensors) * np.arange(n_sensors, dtype=np.float64)
        return cls(points=pts, length=float(length))


def sample_fourier_ics(
    n_samples: int,
    grid: SensorGrid,
    *,
    n_modes: int = 4,
    amplitude: float = 1.0,
    seed: int = 0,
) -> np.ndarray:
    """Deterministic truncated-Fourier initial-condition ensemble.

    Each sample is

        u(x) = sum_{k=1}^{n_modes} (a_k cos(k x') + b_k sin(k x'))

    with ``x' = 2 pi x / length`` and ``(a_k, b_k)`` drawn from a seeded
    ``N(0, amplitude^2 / n_modes)``. Returns shape ``(n_samples, m)`` evaluated
    on ``grid.points`` (1-D sensors only).

    Parameters
    ----------
    n_samples
        Number of independent initial conditions.
    grid
        1-D :class:`SensorGrid`.
    n_modes
        Highest Fourier mode retained (``>= 1``).
    amplitude
        Overall scale of the coefficient draw.
    seed
        RNG seed; the ensemble is fully determined by this seed.
    """
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")
    if n_modes < 1:
        raise ValueError(f"n_modes must be >= 1, got {n_modes}")
    pts = np.asarray(grid.points, dtype=np.float64)
    if pts.ndim != 1:
        raise ValueError(
            f"sample_fourier_ics requires a 1-D SensorGrid; got points shape "
            f"{pts.shape}"
        )
    rng = np.random.default_rng(int(seed))
    scale = float(amplitude) / np.sqrt(float(n_modes))
    a = rng.normal(0.0, scale, size=(n_samples, n_modes))
    b = rng.normal(0.0, scale, size=(n_samples, n_modes))
    x = (2.0 * np.pi / float(grid.length)) * pts  # (m,)
    out = np.zeros((n_samples, pts.shape[0]), dtype=np.float64)
    for k in range(1, n_modes + 1):
        out += a[:, k - 1 : k] * np.cos(k * x)[None, :]
        out += b[:, k - 1 : k] * np.sin(k * x)[None, :]
    return out


__all__ = ["SensorGrid", "sample_fourier_ics"]
