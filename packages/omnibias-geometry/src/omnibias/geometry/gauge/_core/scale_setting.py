# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Lattice scale setting from ``V(r)`` and Wilson flow.

Sommer ``r0`` and Luscher ``t0`` / ``w0`` are finite-spacing scales. They
are not an ``a → 0`` theorem and must not be copied onto sealed Yang-Mills
certificates. ``yang_mills_claim`` stays false.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SOMMER_TARGET = 1.65
WILSON_FLOW_C = 0.3
SCALE_SCOPE = "lattice scale; not a→0 theorem"


@dataclass(frozen=True)
class ScaleSetting:
    """A finite-spacing scale. Not a continuum claim."""

    name: str
    value: float
    target: float
    yang_mills_claim: bool = False
    continuum_claim: bool = False
    scope: str = SCALE_SCOPE


def _first_crossing(x: np.ndarray, y: np.ndarray, target: float) -> float:
    x_arr = np.asarray(x, dtype=float).reshape(-1)
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    if x_arr.size < 2:
        raise ValueError("need at least two samples to interpolate a scale")
    for left, right in zip(range(x_arr.size - 1), range(1, x_arr.size), strict=True):
        y0, y1 = y_arr[left], y_arr[right]
        if not (np.isfinite(y0) and np.isfinite(y1)):
            continue
        if (y0 - target) * (y1 - target) > 0.0 and abs(y0 - target) > 1e-15:
            continue
        if abs(y1 - y0) < 1e-15:
            return float(x_arr[left])
        frac = (target - y0) / (y1 - y0)
        return float(x_arr[left] + frac * (x_arr[right] - x_arr[left]))
    raise ValueError(f"curve never crosses target {target}")


def sommer_r0(
    radii: np.ndarray,
    force: np.ndarray,
    *,
    target: float = SOMMER_TARGET,
) -> ScaleSetting:
    """Solve ``r² F(r) = target`` (default 1.65). Lattice units only."""
    r = np.asarray(radii, dtype=float).reshape(-1)
    f = np.asarray(force, dtype=float).reshape(-1)
    if r.shape != f.shape:
        raise ValueError("radii and force must share shape")
    order = np.argsort(r)
    r2f = (r[order] ** 2) * f[order]
    value = _first_crossing(r[order], r2f, float(target))
    return ScaleSetting(name="r0", value=value, target=float(target))


def t0_from_energy_curve(
    flow_time: np.ndarray,
    energy: np.ndarray,
    *,
    target: float = WILSON_FLOW_C,
) -> ScaleSetting:
    """Solve ``t² E(t) = target`` (default 0.3). Convention, not QCD."""
    t = np.asarray(flow_time, dtype=float).reshape(-1)
    e = np.asarray(energy, dtype=float).reshape(-1)
    value = _first_crossing(t, (t**2) * e, float(target))
    return ScaleSetting(name="t0", value=value, target=float(target))


def w0_from_energy_curve(
    flow_time: np.ndarray,
    energy: np.ndarray,
    *,
    target: float = WILSON_FLOW_C,
) -> ScaleSetting:
    """Solve ``t d/dt (t² E) = target``. Lattice Wilson-flow scale."""
    t = np.asarray(flow_time, dtype=float).reshape(-1)
    e = np.asarray(energy, dtype=float).reshape(-1)
    t2e = (t**2) * e
    deriv = np.gradient(t2e, t)
    value = _first_crossing(t, t * deriv, float(target))
    return ScaleSetting(name="w0", value=value, target=float(target))


__all__ = [
    "SCALE_SCOPE",
    "SOMMER_TARGET",
    "ScaleSetting",
    "WILSON_FLOW_C",
    "sommer_r0",
    "t0_from_energy_curve",
    "w0_from_energy_curve",
]
