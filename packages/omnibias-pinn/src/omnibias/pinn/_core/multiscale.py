# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Choosing frequency bands from a measured spectrum, instead of guessing them.

The multi-scale PINN constructs -- the Fourier-feature encoding
(:class:`omnibias.pinn.torch.fields.FourierFeatureVectorField`) and the MscaleDNN
band mixture (:class:`omnibias.pinn.torch.fields.MscaleVectorField`) -- both take
a tuple of band scales, and in the literature that tuple is a hyperparameter you
guess (usually the ladder ``1, 2, 4, 8, ...``). omnibias already ships
:func:`~omnibias.pinn._core.diagnostics.power_spectrum_per_d`, which measures where
a field actually keeps its energy, so the tuple can be *read off the data*.

These helpers close that loop. They are pure numpy, so they live in ``_core`` and
are re-exported by both backend diagnostics modules.

Units
-----
:func:`~omnibias.pinn._core.diagnostics.power_spectrum_per_d` bins by the *integer*
wavenumber index
``j``, i.e. the mode ``exp(2 pi i j x / L)`` on a domain of length ``L``. A
Fourier feature ``cos(B x)`` with ``B ~ N(0, (2 pi s)^2)`` has typical angular
frequency ``2 pi s``, so the band scale that resolves index ``j`` is ``s = j / L``
-- both are "cycles per unit length". That conversion is the only physics here;
everything else is bookkeeping over the spectrum.
"""

from __future__ import annotations

import math

import numpy as np
from omnibias.pinn._core.diagnostics import power_spectrum_per_d


def geometric_bands(
    n_bands: int, *, base: float = 2.0, start: float = 1.0
) -> tuple[float, ...]:
    """The classic MscaleDNN ladder ``(start, start*base, start*base^2, ...)``.

    The literature default, and the sensible fallback when there is no solution
    sample to measure yet. :func:`suggest_frequency_bands` replaces it once one
    exists.
    """
    if n_bands < 1:
        raise ValueError(f"n_bands must be >= 1, got {n_bands}")
    if base <= 1.0:
        raise ValueError(f"base must be > 1, got {base}")
    if start <= 0.0:
        raise ValueError(f"start must be > 0, got {start}")
    return tuple(start * math.pow(base, j) for j in range(n_bands))


def dominant_wavenumbers(
    power: np.ndarray, *, n_bands: int = 2, drop_dc: bool = True
) -> tuple[float, ...]:
    """Split a power spectrum into equal-energy bands and return their centroids.

    Equal-*energy* rather than equal-width splitting is what makes this useful for
    band selection: a spectrum with a strong low peak and a weak high tail gets one
    band on each, whereas equal-width bins would spend every band on the peak.

    Parameters
    ----------
    power:
        Power per integer wavenumber bin, as returned by
        :func:`~omnibias.pinn._core.diagnostics.power_spectrum_per_d`; ``power[j]`` is the
        energy at
        ``|k| = j``.
    n_bands:
        How many bands to return.
    drop_dc:
        Ignore bin 0 (the mean), which carries no oscillation and would otherwise
        pull the lowest band toward zero.

    Returns
    -------
    Ascending integer-wavenumber centroids, one per band. Bands whose segment
    holds no energy are skipped, so the result can be shorter than ``n_bands``;
    it is never empty (a flat-zero spectrum yields ``(0.0,)``).
    """
    if n_bands < 1:
        raise ValueError(f"n_bands must be >= 1, got {n_bands}")
    p = np.asarray(power, dtype=float)
    if p.ndim != 1:
        raise ValueError(f"power must be 1-D (per-wavenumber bins), got shape {p.shape}")
    if p.size == 0:
        raise ValueError("power must have at least one bin")
    if np.any(p < 0.0):
        raise ValueError("power bins must be non-negative")

    k = np.arange(p.size, dtype=float)
    if drop_dc and p.size > 1:
        p = p[1:]
        k = k[1:]
    total = float(p.sum())
    if total <= 0.0:
        return (0.0,)

    # Equal-energy quantile edges over the cumulative spectrum.
    cumulative = np.cumsum(p)
    centroids: list[float] = []
    lo = 0
    for m in range(1, n_bands + 1):
        target = total * m / n_bands
        hi = int(np.searchsorted(cumulative, target, side="left")) + 1
        hi = min(hi, p.size)
        if hi <= lo:
            continue
        seg_p = p[lo:hi]
        seg_energy = float(seg_p.sum())
        if seg_energy > 0.0:
            centroids.append(float((k[lo:hi] * seg_p).sum() / seg_energy))
        lo = hi
    if not centroids:
        return (0.0,)
    return tuple(centroids)


def suggest_frequency_bands(
    u_grid: np.ndarray,
    *,
    L: float | tuple[float, ...],
    n_bands: int = 2,
    min_scale: float = 1e-3,
) -> tuple[float, ...]:
    """Read Fourier / Mscale band scales off a sampled field's power spectrum.

    The feedback loop: sample the current solution (or a reference / coarse
    solution) on a uniform grid, measure where its energy lives, and hand the
    resulting scales to a :class:`FourierFeatureVectorField` ``frequency_scale``
    or an :class:`MscaleVectorField` ``scales``. A field that keeps energy at
    wavenumber ``j`` on a domain of length ``L`` needs a band near ``j / L``.

    Parameters
    ----------
    u_grid:
        Field samples of shape ``(T, *spatial)`` on a uniform grid; use ``T = 1``
        for a steady problem.
    L:
        Domain period(s). A tuple is reduced to its mean, because
        :func:`~omnibias.pinn._core.diagnostics.power_spectrum_per_d` bins isotropically
        over the index magnitude;
        for a strongly anisotropic domain, call this per axis instead.
    n_bands:
        How many bands to return.
    min_scale:
        Floor applied to each scale, since a band scale of exactly zero would give
        a constant feature.

    Returns
    -------
    Ascending, strictly increasing band scales in cycles per unit length.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.linspace(0.0, 1.0, 128, endpoint=False)
    >>> u = np.sin(2 * np.pi * 3 * x) + 0.5 * np.sin(2 * np.pi * 20 * x)
    >>> bands = suggest_frequency_bands(u[None, :], L=1.0, n_bands=2)
    >>> [round(b) for b in bands]
    [3, 20]
    """
    if n_bands < 1:
        raise ValueError(f"n_bands must be >= 1, got {n_bands}")
    if min_scale <= 0.0:
        raise ValueError(f"min_scale must be > 0, got {min_scale}")
    if isinstance(L, int | float):
        length = float(L)
    else:
        lengths = tuple(float(x) for x in L)
        if not lengths:
            raise ValueError("L must contain at least one period")
        length = sum(lengths) / len(lengths)
    if length <= 0.0:
        raise ValueError(f"L must be > 0, got {L!r}")

    power = power_spectrum_per_d(np.asarray(u_grid, dtype=float), L)
    centroids = dominant_wavenumbers(power, n_bands=n_bands)

    scales: list[float] = []
    for c in centroids:
        s = max(float(c) / length, min_scale)
        # Keep the tuple strictly increasing: coincident bands are redundant
        # (identical feature statistics) and only waste width.
        if scales and s <= scales[-1]:
            continue
        scales.append(s)
    return tuple(scales)


__all__ = [
    "dominant_wavenumbers",
    "geometric_bands",
    "suggest_frequency_bands",
]
