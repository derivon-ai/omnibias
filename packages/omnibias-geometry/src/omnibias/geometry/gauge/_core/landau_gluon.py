# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Landau-gauge SU(2) gluon 2-point on lattice links (Path B, scope B).

The 2-point is a fixed-spacing lattice observable after a Landau-gauge
overrelaxation. It is not a ``GaugeCovariantJet`` and not a continuum
propagator. Spectral inversion of ``G(p^2)`` is a separate ill-posed step.

Honesty: ``yang_mills_claim`` / ``continuum_claim`` stay false. ``ill_posed``
is false for the 2-point itself.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from omnibias.geometry.gauge._core.data_paths import LatticeLinkField
from omnibias.geometry.gauge._core.ensemble_language import (
    ENSEMBLE_G_P2,
    ENSEMBLE_P2,
    EnsembleObservableTable,
)
from omnibias.geometry.gauge.lattice._core.kernels import (
    algebra_from_links,
    gauge_transform_links,
    landau_gauge_overrelax,
    normalize_quaternion,
    quat_conj,
    quat_mul,
    quat_power,
)

LANDAU_RESIDUAL_ATOL = 1e-3
LANDAU_ORBIT_ATOL = 1e-4
LANDAU_TRANSVERSE_ATOL = 5e-2
DEFAULT_LANDAU_STEPS = 48
DEFAULT_LANDAU_OMEGA = 1.7


def _copy_links(links: np.ndarray) -> np.ndarray:
    return np.array(links, dtype=np.float64, copy=True)


def _landau_functional(links: np.ndarray) -> float:
    return float(np.mean(links[..., 0]))


def _div_algebra(links: np.ndarray) -> np.ndarray:
    algebra = np.asarray(algebra_from_links(np, links), dtype=np.float64)
    divergence = np.zeros(algebra.shape[1:], dtype=np.float64)
    for mu in range(4):
        divergence = divergence + algebra[mu] - np.roll(algebra[mu], 1, axis=mu)
    return divergence


def landau_residual(links: np.ndarray) -> float:
    """RMS lattice divergence of ``A_mu^a``."""
    divergence = _div_algebra(links)
    return float(np.sqrt(np.mean(divergence**2)))


def _apply_site_gauge(links: np.ndarray, index: tuple[int, ...], gauge: np.ndarray) -> None:
    gauge_dag = np.asarray(quat_conj(np, gauge), dtype=np.float64)
    shape = links.shape[1:5]
    for mu in range(4):
        links[(mu, *index)] = np.asarray(
            quat_mul(np, gauge, links[(mu, *index)]), dtype=np.float64
        )
        pred = list(index)
        pred[mu] = (pred[mu] - 1) % shape[mu]
        pred_t = tuple(pred)
        links[(mu, *pred_t)] = np.asarray(
            quat_mul(np, links[(mu, *pred_t)], gauge_dag), dtype=np.float64
        )


def _site_w(links: np.ndarray, index: tuple[int, ...]) -> np.ndarray:
    acc = np.zeros(4, dtype=np.float64)
    shape = links.shape[1:5]
    for mu in range(4):
        acc = acc + links[(mu, *index)]
        pred = list(index)
        pred[mu] = (pred[mu] - 1) % shape[mu]
        acc = acc + np.asarray(quat_conj(np, links[(mu, *tuple(pred))]), dtype=np.float64)
    return acc


def landau_gauge_overrelax_sequential(
    links: np.ndarray,
    *,
    n_steps: int,
    omega: float = DEFAULT_LANDAU_OMEGA,
) -> np.ndarray:
    """Site-sequential SU(2) Landau overrelaxation (numpy)."""
    if n_steps < 0:
        raise ValueError(f"n_steps must be >= 0, got {n_steps}")
    out = _copy_links(links)
    shape = out.shape[1:5]
    for _ in range(int(n_steps)):
        for index in np.ndindex(*shape):
            staple = _site_w(out, index)
            gauge = np.asarray(
                quat_conj(np, normalize_quaternion(np, staple)), dtype=np.float64
            )
            if abs(float(omega) - 1.0) > 1e-15:
                gauge = np.asarray(
                    normalize_quaternion(np, quat_power(np, gauge, float(omega))),
                    dtype=np.float64,
                )
            _apply_site_gauge(out, index, gauge)
    return out


def landau_gauge_fix(
    field: LatticeLinkField,
    *,
    n_steps: int = DEFAULT_LANDAU_STEPS,
    omega: float = DEFAULT_LANDAU_OMEGA,
    sequential: bool = True,
) -> dict[str, Any]:
    """Fix links to lattice Landau gauge. Not a continuum claim."""
    links = np.asarray(field.links, dtype=np.float64)
    if sequential:
        fixed = landau_gauge_overrelax_sequential(links, n_steps=n_steps, omega=omega)
    else:
        fixed = np.asarray(
            landau_gauge_overrelax(np, links, n_steps=n_steps, omega=omega),
            dtype=np.float64,
        )
    return {
        "links": fixed,
        "functional": _landau_functional(fixed),
        "residual": landau_residual(fixed),
        "n_steps": int(n_steps),
        "omega": float(omega),
        "yang_mills_claim": False,
        "continuum_claim": False,
        "ill_posed": False,
    }


def _lattice_hat_p(shape: tuple[int, int, int, int]) -> list[np.ndarray]:
    grids = np.meshgrid(
        *[np.arange(length, dtype=np.float64) for length in shape], indexing="ij"
    )
    return [
        2.0 * np.sin(np.pi * grids[mu] / float(shape[mu])) for mu in range(4)
    ]


def gluon_propagator_p2(
    field: LatticeLinkField,
    *,
    already_fixed: bool = False,
    n_steps: int = DEFAULT_LANDAU_STEPS,
    omega: float = DEFAULT_LANDAU_OMEGA,
) -> tuple[EnsembleObservableTable, dict[str, Any]]:
    """Landau dressing ``D(p^2)`` from a single configuration's ``A_mu^a``.

    Returns an :class:`EnsembleObservableTable` with ``p2`` / ``G_p2`` and a
    report. Zero modes are dropped. Not a continuum gluon propagator.
    """
    report: dict[str, Any]
    if already_fixed:
        links = np.asarray(field.links, dtype=np.float64)
        report = {
            "functional": _landau_functional(links),
            "residual": landau_residual(links),
            "n_steps": 0,
            "yang_mills_claim": False,
            "continuum_claim": False,
            "ill_posed": False,
        }
    else:
        fixed = landau_gauge_fix(field, n_steps=n_steps, omega=omega)
        links = np.asarray(fixed["links"], dtype=np.float64)
        report = {key: value for key, value in fixed.items() if key != "links"}

    algebra = np.asarray(algebra_from_links(np, links), dtype=np.float64)
    lattice = tuple(int(size) for size in algebra.shape[1:5])
    volume = float(np.prod(lattice))
    algebra_k = np.fft.fftn(algebra, axes=(1, 2, 3, 4))
    dressing = np.einsum("m...a,n...a->mn...", algebra_k, np.conjugate(algebra_k)) / volume
    trace = sum(dressing[mu, mu] for mu in range(4))
    g_p2 = np.real(trace) / 3.0
    hat_p = _lattice_hat_p(lattice)
    p2 = sum(component**2 for component in hat_p)
    mask = p2 > 1e-18
    grids = np.meshgrid(
        *[np.arange(length, dtype=np.float64) for length in lattice], indexing="ij"
    )
    # Backward-difference Fourier factor matching div A = A(x) - A(x-μ).
    lattice_p = [
        1.0 - np.exp(-2.0j * np.pi * grids[mu] / float(lattice[mu])) for mu in range(4)
    ]
    contracted = np.zeros(lattice, dtype=np.float64)
    for nu in range(4):
        partial = np.zeros(lattice, dtype=np.complex128)
        for mu in range(4):
            partial = partial + lattice_p[mu] * dressing[mu, nu]
        contracted = contracted + np.abs(partial) ** 2
    denom = float(np.sqrt(np.mean(np.abs(dressing) ** 2))) + 1e-30
    transverse = float(np.sqrt(np.mean(contracted[mask]))) / denom if np.any(mask) else 0.0
    report["transverse_residual"] = transverse
    table = EnsembleObservableTable(
        values={
            ENSEMBLE_P2: np.asarray(p2[mask], dtype=np.float64).reshape(-1),
            ENSEMBLE_G_P2: np.asarray(g_p2[mask], dtype=np.float64).reshape(-1),
        },
        source="landau_gluon",
    )
    return table, report


def random_site_gauge(
    lattice_shape: tuple[int, int, int, int], rng: np.random.Generator
) -> np.ndarray:
    """Haar-ish random unit-quaternion site field."""
    raw = rng.normal(size=(*lattice_shape, 4))
    norms = np.linalg.norm(raw, axis=-1, keepdims=True)
    return raw / np.maximum(norms, 1e-30)


def gauge_transform_numpy(links: np.ndarray, gauge: np.ndarray) -> np.ndarray:
    """Apply a site gauge field to numpy links."""
    return np.asarray(gauge_transform_links(np, links, gauge), dtype=np.float64)


__all__ = [
    "DEFAULT_LANDAU_OMEGA",
    "DEFAULT_LANDAU_STEPS",
    "LANDAU_ORBIT_ATOL",
    "LANDAU_RESIDUAL_ATOL",
    "LANDAU_TRANSVERSE_ATOL",
    "gauge_transform_numpy",
    "gluon_propagator_p2",
    "landau_gauge_fix",
    "landau_gauge_overrelax_sequential",
    "landau_residual",
    "random_site_gauge",
]
