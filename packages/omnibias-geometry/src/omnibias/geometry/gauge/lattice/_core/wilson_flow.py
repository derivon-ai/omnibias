# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Deterministic SU(2) Luscher Wilson flow on quaternion links.

This is not continuum ``yang_mills_gradient_flow_rhs`` (a jet of ``A``)
and not lattice Langevin (stochastic). Energy decrease and single-configuration
``t² E(t)`` crossings are diagnostics; an ensemble average is not supplied. ``yang_mills_claim`` stays false.
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
    """Dimensionless deficit ``1 - mean_P``, ``P = (1/2) Re tr U_{μν}``.

    This preserves the plaquette-proxy API. The conventional four-dimensional
    SU(2) Wilson-flow density is ``a^4 E = 24 * mean_plaquette_energy(links)``.
    Identity links have zero deficit.
    """
    traces: list[float] = []
    for mu in range(4):
        for nu in range(mu + 1, 4):
            traces.append(float(np.mean(plaquette_trace(np, links, mu, nu))))
    return float(1.0 - np.mean(np.asarray(traces, dtype=np.float64)))


def wilson_flow_step(links: np.ndarray, eps: float) -> np.ndarray:
    """One Lie-Euler Wilson-flow step ``U ← exp(ε Im(Σ U†)) U``.

    ``eps`` advances ``t/a^2``. With generators ``i σ^a / 2`` and Wilson
    ``β = 4/g0^2``, the factor ``g0^2`` in the flow cancels that in the action;
    the quaternion vector exponent shown here needs no extra time factor.
    """
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
    """Numerical Wilson flow on one SU(2) configuration.

    ``flow_time`` is ``t/a^2`` and ``energy`` is the lattice-site mean of ``a^4 E``
    with ``E`` normalized by Luscher (2010), eq. (3.1):
    ``a^4 E(x) = 2 sum_{mu<nu} Re tr(I - U_munu(x))``. The SU(2) trace
    and six planes therefore give ``24 * (1 - mean_P)``.

    Returned ``t0``/``w0`` are crossings of this single-configuration curve.
    A physical ensemble scale requires averaging the energy curves before
    extracting a crossing, together with integration and statistical errors.
    """
    links = np.asarray(field.links, dtype=np.float64)
    times = [0.0]
    energies = [24.0 * mean_plaquette_energy(links)]
    cur = links
    for step in range(int(n_steps)):
        cur = wilson_flow_step(cur, float(eps))
        times.append(float(eps) * (step + 1))
        energies.append(24.0 * mean_plaquette_energy(cur))
    t = np.asarray(times, dtype=np.float64)
    e = np.asarray(energies, dtype=np.float64)
    decreased = bool(e[-1] <= e[0] + 1e-12)
    t0: ScaleSetting | None = None
    w0: ScaleSetting | None = None
    try:
        t0 = t0_from_energy_curve(t, e, target=WILSON_FLOW_C)
    except ValueError:
        t0 = None
    if t.size >= 2:
        try:
            w0 = w0_from_energy_curve(t, e, target=WILSON_FLOW_C)
        except ValueError:
            w0 = None
    return {
        "flow_time": t,
        "energy": e,
        "energy_decreased": decreased,
        "flow_time_units": "t/a^2",
        "energy_units": "a^4 E",
        "energy_normalization": "2 sum_{mu<nu} Re tr(I-U_munu); SU(2)",
        "scale_scope": "single_configuration_crossing",
        "ensemble_scale_claim": False,
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
