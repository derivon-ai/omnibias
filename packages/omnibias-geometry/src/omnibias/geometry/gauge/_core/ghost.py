# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Landau-gauge ghost 2-point from the adjoint covariant Laplacian.

Plane-wave Rayleigh quotient of the Faddeev–Popov / adjoint Laplacian on
Landau-fixed links. Lattice observable, not continuum Kugo–Ojima.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from omnibias.geometry.gauge._core.data_paths import LatticeLinkField
from omnibias.geometry.gauge._core.ensemble_language import (
    ENSEMBLE_GHOST_G,
    ENSEMBLE_INV_P2,
    ENSEMBLE_LOG_GHOST_G,
    ENSEMBLE_LOG_P2,
    ENSEMBLE_P2,
    EnsembleObservableTable,
    LatticeMetadata,
    refuse_single_config_as_ensemble,
)
from omnibias.geometry.gauge._core.landau_gluon import (
    DEFAULT_LANDAU_OMEGA,
    DEFAULT_LANDAU_STEPS,
    _lattice_hat_p,
    landau_gauge_fix,
)
from omnibias.geometry.gauge.lattice._core.stats import ensemble_mean_jackknife


def _rotate_adjoint(q: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """SO(3) image of a unit quaternion acting on an algebra 3-vector."""
    w = q[..., 0]
    u = q[..., 1:4]
    cross = np.cross(u, vec)
    return (
        vec * (2.0 * w * w - 1.0)[..., None]
        + 2.0 * u * np.sum(u * vec, axis=-1, keepdims=True)
        + 2.0 * w[..., None] * cross
    )


def _adjoint_laplacian(links: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Lattice adjoint Laplacian ``Σ_μ (2φ - U_μ φ(x+μ) - U_{x-μ}† φ(x-μ))``."""
    out = np.zeros_like(phi)
    for mu in range(4):
        fwd = np.roll(phi, -1, axis=mu)
        bwd = np.roll(phi, 1, axis=mu)
        u_fwd = links[mu]
        u_bwd = np.roll(links[mu], 1, axis=mu)
        u_bwd_inv = u_bwd.copy()
        u_bwd_inv[..., 1:4] = -u_bwd[..., 1:4]
        out = out + 2.0 * phi - _rotate_adjoint(u_fwd, fwd) - _rotate_adjoint(u_bwd_inv, bwd)
    return out


def _plane_wave(shape: tuple[int, int, int, int], wave: tuple[int, int, int, int]) -> np.ndarray:
    grids = np.meshgrid(
        *[np.arange(length, dtype=np.float64) for length in shape], indexing="ij"
    )
    phase = np.zeros(shape, dtype=np.float64)
    for mu, k_mu in enumerate(wave):
        phase = phase + 2.0 * np.pi * k_mu * grids[mu] / float(shape[mu])
    wave_fn = np.cos(phase)
    field = np.zeros((*shape, 3), dtype=np.float64)
    field[..., 0] = wave_fn
    return field


def ghost_propagator_p2(
    field: LatticeLinkField,
    *,
    already_fixed: bool = False,
    n_steps: int = DEFAULT_LANDAU_STEPS,
    omega: float = DEFAULT_LANDAU_OMEGA,
) -> tuple[EnsembleObservableTable, dict[str, Any]]:
    """Landau ghost dressing from the adjoint-Laplacian Rayleigh quotient."""
    if already_fixed:
        links = np.asarray(field.links, dtype=np.float64)
        residual = 0.0
    else:
        fixed = landau_gauge_fix(field, n_steps=n_steps, omega=omega)
        links = np.asarray(fixed["links"], dtype=np.float64)
        residual = float(fixed["residual"])
    shape = tuple(int(size) for size in links.shape[1:5])
    hat_p = _lattice_hat_p(shape)
    p2_grid = sum(component**2 for component in hat_p)
    p2_vals: list[float] = []
    ghost_vals: list[float] = []
    for wave in np.ndindex(*shape):
        p2 = float(p2_grid[wave])
        if p2 <= 1e-18:
            continue
        psi = _plane_wave(shape, wave)
        applied = _adjoint_laplacian(links, psi)
        denom = float(np.sum(psi * applied))
        numer = float(np.sum(psi * psi))
        ghost = numer / denom if abs(denom) > 1e-18 else float("nan")
        if np.isfinite(ghost) and ghost > 0.0:
            p2_vals.append(p2)
            ghost_vals.append(ghost)
    p2_arr = np.asarray(p2_vals, dtype=np.float64)
    ghost_arr = np.asarray(ghost_vals, dtype=np.float64)
    table = EnsembleObservableTable(
        values={
            ENSEMBLE_P2: p2_arr,
            ENSEMBLE_GHOST_G: ghost_arr,
            ENSEMBLE_LOG_GHOST_G: np.log(np.maximum(ghost_arr, 1e-30)),
            ENSEMBLE_LOG_P2: np.log(np.maximum(p2_arr, 1e-30)),
            ENSEMBLE_INV_P2: 1.0 / np.maximum(p2_arr, 1e-30),
        },
        source="landau_gluon",
        metadata=LatticeMetadata(
            lattice_shape=shape,
            scheme="landau",
            n_configs=1,
        ),
    )
    report = {
        "residual": residual,
        "yang_mills_claim": False,
        "continuum_claim": False,
        "ill_posed": False,
        "method": "plane_wave_rayleigh",
    }
    return table, report


def ghost_propagator_ensemble(
    fields: Sequence[LatticeLinkField],
    *,
    already_fixed: bool = False,
    n_steps: int = DEFAULT_LANDAU_STEPS,
    omega: float = DEFAULT_LANDAU_OMEGA,
) -> tuple[EnsembleObservableTable, dict[str, Any]]:
    """Jackknifed ghost dressing from two or more Landau-fixed configs."""
    if len(fields) < 2:
        refuse_single_config_as_ensemble(fields[0] if fields else None)
    tables = [
        ghost_propagator_p2(
            field, already_fixed=already_fixed, n_steps=n_steps, omega=omega
        )[0]
        for field in fields
    ]
    p2 = np.asarray(tables[0].values[ENSEMBLE_P2], dtype=np.float64)
    stacked = np.stack(
        [np.asarray(table.values[ENSEMBLE_GHOST_G], dtype=np.float64) for table in tables],
        axis=0,
    )
    means = np.zeros(p2.shape[0], dtype=np.float64)
    errs = np.zeros(p2.shape[0], dtype=np.float64)
    for col in range(p2.shape[0]):
        means[col], errs[col] = ensemble_mean_jackknife(stacked[:, col].tolist())
    table = EnsembleObservableTable(
        values={
            ENSEMBLE_P2: p2,
            ENSEMBLE_GHOST_G: means,
            ENSEMBLE_LOG_GHOST_G: np.log(np.maximum(means, 1e-30)),
            ENSEMBLE_LOG_P2: np.log(np.maximum(p2, 1e-30)),
            ENSEMBLE_INV_P2: 1.0 / np.maximum(p2, 1e-30),
        },
        source="landau_gluon",
        metadata=LatticeMetadata(scheme="landau", n_configs=len(fields)),
    )
    return table, {
        "n_configs": len(fields),
        "ghost_err": errs,
        "yang_mills_claim": False,
        "continuum_claim": False,
        "ill_posed": False,
    }


def planted_ghost_table(
    *,
    mass2: float = 0.5,
    z0: float = 1.0,
    n_rows: int = 24,
) -> EnsembleObservableTable:
    """Plant ``ghost_G = Z / (p² + M²)``. Not Kugo–Ojima."""
    p2 = np.linspace(0.08, 3.5, n_rows, dtype=np.float64)
    ghost = z0 / (p2 + mass2)
    return EnsembleObservableTable(
        values={
            ENSEMBLE_P2: p2,
            ENSEMBLE_GHOST_G: ghost,
            ENSEMBLE_LOG_GHOST_G: np.log(ghost),
            ENSEMBLE_LOG_P2: np.log(p2),
            ENSEMBLE_INV_P2: 1.0 / p2,
        },
        source="planted",
        metadata=LatticeMetadata(scheme="landau", already_inverse=False, n_configs=n_rows),
    )


__all__ = [
    "ghost_propagator_ensemble",
    "ghost_propagator_p2",
    "planted_ghost_table",
]
