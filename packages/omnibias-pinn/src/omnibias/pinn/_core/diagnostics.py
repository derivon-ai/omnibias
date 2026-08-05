# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Backend-agnostic diagnostics that consume numpy arrays.

These trajectory-level metrics are pure numpy so they live in
``_core`` and are re-exported by both ``omnibias.pinn.torch.diagnostics``
and ``omnibias.pinn.jax.diagnostics``.

Generalised from an internal Kuramoto-Sivashinsky benchmark reference
to arbitrary spatial dimension :math:`D` (the original helpers were
1D-only).
"""

from __future__ import annotations

import numpy as np

# ---------------- per-time errors -----------------------------------


def relative_l2_per_time(u_pred: np.ndarray, u_ref: np.ndarray) -> np.ndarray:
    """Per-snapshot relative L^2 error.

    Parameters
    ----------
    u_pred, u_ref
        Numpy arrays of shape ``(T, *spatial)``. The relative L^2 is
        computed over the spatial axes; the leading axis is treated as
        time.

    Returns
    -------
    rel_l2 : ndarray of shape ``(T,)``.
    """
    if u_pred.shape != u_ref.shape:
        raise ValueError(
            f"shape mismatch: u_pred {u_pred.shape}, u_ref {u_ref.shape}"
        )
    if u_pred.ndim < 2:
        raise ValueError(
            f"need at least (T, *spatial); got shape {u_pred.shape}"
        )
    spatial = tuple(range(1, u_pred.ndim))
    num = np.sqrt(np.mean((u_pred - u_ref) ** 2, axis=spatial))
    den = np.sqrt(np.mean(u_ref ** 2, axis=spatial)) + 1e-30
    return num / den


def forecast_horizon(
    times: np.ndarray, rel_l2_t: np.ndarray, *, threshold: float = 0.5,
) -> float:
    """First time at which ``rel_l2_t`` exceeds ``threshold``.

    Returns ``times[-1]`` if the threshold is never crossed (i.e. the
    predictor tracks the truth for the full horizon).
    """
    rel_l2_t = np.asarray(rel_l2_t)
    times = np.asarray(times)
    if rel_l2_t.shape != times.shape:
        raise ValueError(
            f"shape mismatch: rel_l2_t {rel_l2_t.shape}, times {times.shape}"
        )
    crossings = np.where(rel_l2_t > threshold)[0]
    if crossings.size == 0:
        return float(times[-1])
    return float(times[crossings[0]])


# ---------------- spectral fidelity --------------------------------


def power_spectrum_per_d(
    u_grid: np.ndarray, L: float | tuple[float, ...],
) -> np.ndarray:
    """Time-averaged isotropic power spectrum.

    Parameters
    ----------
    u_grid
        Array of shape ``(T, *spatial)``. Spatial axes uniformly sampled.
    L
        Period(s).

    Returns
    -------
    P : ndarray of shape ``(K + 1,)`` where ``K`` is the largest binned
        wavenumber index. Bin ``j`` corresponds to integer wavenumber
        :math:`|k| = j` in the index basis.
    """
    if u_grid.ndim < 2:
        raise ValueError(
            f"need (T, *spatial); got shape {u_grid.shape}"
        )
    spatial = u_grid.shape[1:]
    D = len(spatial)
    if isinstance(L, int | float):
        L_t = tuple(float(L) for _ in range(D))
    else:
        L_t = tuple(float(x) for x in L)
        if len(L_t) != D:
            raise ValueError(
                f"L tuple length {len(L_t)} does not match spatial dim {D}"
            )

    # Build integer wavenumber index grid.
    k_int = [np.fft.fftfreq(n, d=1.0 / n).astype(int) for n in spatial]
    grids = np.meshgrid(*k_int, indexing="ij")
    k_mag_int = np.sqrt(sum(g * g for g in grids))
    k_idx = np.round(k_mag_int).astype(int)

    u_hat = np.fft.fftn(u_grid, axes=tuple(range(1, D + 1)))
    n_spatial = int(np.prod(spatial))
    energy_modes = (np.abs(u_hat) ** 2).mean(axis=0) / (n_spatial ** 2)

    k_max = int(k_idx.max())
    P = np.zeros(k_max + 1)
    counts = np.zeros(k_max + 1)
    flat_idx = k_idx.ravel()
    valid = flat_idx >= 0
    flat_idx = flat_idx[valid]
    e_flat = energy_modes.ravel()[valid]
    np.add.at(P, flat_idx, e_flat)
    np.add.at(counts, flat_idx, 1.0)
    return P


def spectral_fidelity(
    u_pred: np.ndarray, u_ref: np.ndarray, *,
    L: float | tuple[float, ...],
    n_modes: int | None = None,
) -> float:
    """Relative L^2 distance between time-averaged power spectra.

    Used to validate that the predictor matches the reference *statistically*
    even when pointwise tracking has diverged due to chaos. Generalises
    :func:`research.experiments.kuramoto_sivashinsky.benchmark.power_spectrum_l2`
    to any spatial dimension.

    Parameters
    ----------
    u_pred, u_ref
        Arrays of shape ``(T, *spatial)``.
    L
        Period(s).
    n_modes
        Optional cap on the number of wavenumber bins compared.
        Default: use all bins.
    """
    P_pred = power_spectrum_per_d(u_pred, L)
    P_ref = power_spectrum_per_d(u_ref, L)
    n = min(len(P_pred), len(P_ref))
    if n_modes is not None:
        n = min(n, int(n_modes))
    P_pred = P_pred[:n]
    P_ref = P_ref[:n]
    num = np.sqrt(np.sum((P_pred - P_ref) ** 2))
    den = np.sqrt(np.sum(P_ref ** 2)) + 1e-30
    return float(num / den)


__all__ = [
    "forecast_horizon",
    "power_spectrum_per_d",
    "relative_l2_per_time",
    "spectral_fidelity",
]
