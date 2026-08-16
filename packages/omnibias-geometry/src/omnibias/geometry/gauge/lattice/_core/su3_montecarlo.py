# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Small-volume SU(3) Wilson heat-bath. Fixed spacing, not continuum QCD."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from omnibias.geometry.gauge.lattice._core.stats import (
    creutz_ratios_ensemble,
    jackknife_std,
    string_tension_from_creutz,
    wilson_loops_ensemble,
)
from omnibias.geometry.gauge.lattice._core.su3_kernels import (
    average_su3_plaquette,
    average_su3_polyakov,
    average_su3_wilson,
    identity_su3_links,
    random_su3_links,
    su3_sweep,
)

EVIDENCE_NOTE = (
    "SU(3) lattice MC evidence at fixed spacing; unproven_claim=False; "
    "not a continuum proof and not a Yang-Mills mass-gap claim."
)


def run_su3_lattice_mc(
    *,
    lattice_shape: tuple[int, int, int, int] = (4, 4, 4, 4),
    beta: float = 5.5,
    n_therm: int = 2,
    n_meas: int = 2,
    n_sep: int = 1,
    seed: int = 0,
    r_max: int | None = None,
    cold_start: bool = False,
) -> dict[str, Any]:
    """Cabibbo–Marinari SU(3) smoke driver (default 4⁴)."""
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    shape = tuple(int(size) for size in lattice_shape)
    links = identity_su3_links(shape) if cold_start else random_su3_links(shape, rng)
    spatial_min = min(shape[0], shape[1], shape[2])
    r_lim = min(2, spatial_min // 2) if r_max is None else min(r_max, spatial_min // 2)
    r_lim = max(r_lim, 1)
    for _ in range(n_therm):
        links = su3_sweep(links, beta, rng)
    plaq: list[float] = []
    poly: list[float] = []
    wilson_samples: dict[tuple[int, int], list[float]] = {
        (r_val, t_val): []
        for r_val in range(1, r_lim + 1)
        for t_val in range(1, r_lim + 1)
    }
    for _ in range(n_meas):
        for _sep in range(n_sep):
            links = su3_sweep(links, beta, rng)
        plaq.append(average_su3_plaquette(links))
        poly.append(average_su3_polyakov(links))
        for r_val in range(1, r_lim + 1):
            for t_val in range(1, r_lim + 1):
                wilson_samples[(r_val, t_val)].append(
                    average_su3_wilson(links, r_val, t_val)
                )
    wilson_out = wilson_loops_ensemble(wilson_samples)
    creutz_out = creutz_ratios_ensemble(wilson_samples)
    return {
        "gauge_group": "su(3)",
        "lattice_shape": list(shape),
        "beta": float(beta),
        "n_therm": int(n_therm),
        "n_meas": int(n_meas),
        "n_sep": int(n_sep),
        "avg_plaquette": float(sum(plaq) / len(plaq)),
        "avg_plaquette_err": jackknife_std(plaq),
        "avg_polyakov": float(sum(poly) / len(poly)),
        "avg_polyakov_err": jackknife_std(poly),
        "wilson_loops": wilson_out,
        "creutz_ratios": creutz_out,
        "string_tension": string_tension_from_creutz(creutz_out),
        "evidence_note": EVIDENCE_NOTE,
        "yang_mills_claim": False,
        "continuum_claim": False,
        "seed": int(seed),
        "elapsed_s": float(time.perf_counter() - t0),
    }


__all__ = ["EVIDENCE_NOTE", "run_su3_lattice_mc"]
