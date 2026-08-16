# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Deterministic SU(2) Luscher Wilson flow on quaternion links.

This is not continuum ``yang_mills_gradient_flow_rhs`` (a jet of ``A``)
and not lattice Langevin (stochastic). Energy decrease and a planted
``t² E(t)`` crossing are the only gates. ``yang_mills_claim`` stays false.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from omnibias.geometry.gauge._core.data_paths import LatticeLinkField
from omnibias.geometry.gauge._core.scale_setting import (
    WILSON_FLOW_C,
    ScaleSetting,
    t0_from_energy_curve,
    w0_from_energy_curve,
)
from omnibias.geometry.gauge.lattice._core.kernels import (
    exp_su2,
    plaquette_trace,
    quat_conj,
    quat_mul,
    staple_sum,
)


def mean_plaquette_energy(links: np.ndarray) -> float:
    """``1 - mean_P`` with ``P = (1/2) Re tr U_{μν}``. Identity links have energy 0."""
    traces: list[float] = []
    for mu in range(4):
        for nu in range(mu + 1, 4):
            traces.append(float(np.mean(plaquette_trace(np, links, mu, nu))))
    return float(1.0 - np.mean(np.asarray(traces, dtype=np.float64)))


def wilson_flow_step(links: np.ndarray, eps: float) -> np.ndarray:
    """One Euler Luscher step ``U ← exp(ε Im(Σ U†)) U``."""
    new_dirs = []
    for mu in range(4):
        staple = staple_sum(np, links, mu)
        omega = float(eps) * quat_mul(np, staple, quat_conj(np, links[mu]))[..., 1:]
        new_dirs.append(quat_mul(np, exp_su2(np, omega), links[mu]))
    return np.stack(new_dirs, axis=0)


def run_wilson_flow(
    field: LatticeLinkField,
    *,
    n_steps: int = 8,
    eps: float = 0.02,
) -> dict[str, Any]:
    """Smoke Wilson flow on one SU(2) configuration. Not a continuum scale."""
    links = np.asarray(field.links, dtype=np.float64)
    times = [0.0]
    energies = [mean_plaquette_energy(links)]
    cur = links
    for step in range(int(n_steps)):
        cur = wilson_flow_step(cur, float(eps))
        times.append(float(eps) * (step + 1))
        energies.append(mean_plaquette_energy(cur))
    t = np.asarray(times, dtype=np.float64)
    e = np.asarray(energies, dtype=np.float64)
    decreased = bool(e[-1] <= e[0] + 1e-12)
    t0: ScaleSetting | None = None
    w0: ScaleSetting | None = None
    try:
        t0 = t0_from_energy_curve(t, e, target=WILSON_FLOW_C)
    except ValueError:
        t0 = None
    try:
        w0 = w0_from_energy_curve(t, e, target=WILSON_FLOW_C)
    except ValueError:
        w0 = None
    return {
        "flow_time": t,
        "energy": e,
        "energy_decreased": decreased,
        "t0": None if t0 is None else t0.value,
        "w0": None if w0 is None else w0.value,
        "yang_mills_claim": False,
        "continuum_claim": False,
        "links": cur,
    }


def wilson_flow_scales_from_curve(
    flow_time: Sequence[float],
    energy: Sequence[float],
    *,
    target: float = WILSON_FLOW_C,
) -> dict[str, ScaleSetting]:
    """``t0`` / ``w0`` from a planted or measured ``E(t)`` curve."""
    t = np.asarray(flow_time, dtype=float)
    e = np.asarray(energy, dtype=float)
    return {
        "t0": t0_from_energy_curve(t, e, target=target),
        "w0": w0_from_energy_curve(t, e, target=target),
    }


__all__ = [
    "mean_plaquette_energy",
    "run_wilson_flow",
    "wilson_flow_scales_from_curve",
    "wilson_flow_step",
]
