# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""SU(2) lattice Monte Carlo driver for plaquette and glueball correlator evidence."""

from __future__ import annotations

import time
from typing import Any

import torch
from omnibias.geometry.gauge.lattice.observables import (
    average_plaquette,
    average_polyakov_loop,
    average_wilson_loop,
    connected_correlator_ensemble,
    creutz_ratios_ensemble,
    gevp_ground_mass,
    gevp_plateau,
    glueball_operator_timeslice,
    jackknife_std,
    string_tension_from_creutz,
    wilson_loops_ensemble,
)
from omnibias.geometry.gauge.lattice.su2 import random_links, sweep

EVIDENCE_NOTE = (
    "Lattice MC evidence at fixed spacing; unproven_claim=False; not a continuum proof."
)
DEFAULT_GEVP_LEVELS = (2, 8, 20)


def run_lattice_mc(
    *,
    gauge_group: str = "su(2)",
    lattice_shape: tuple[int, int, int, int] = (8, 8, 8, 8),
    beta: float = 2.3,
    n_therm: int = 200,
    n_meas: int = 200,
    n_sep: int = 2,
    n_smear: int = 10,
    smear_alpha: float = 0.5,
    device: str = "cpu",
    seed: int = 12345,
    r_max: int | None = None,
    gevp_levels: tuple[int, ...] = DEFAULT_GEVP_LEVELS,
) -> dict[str, Any]:
    """Run thermalized SU(2) heat-bath MC and return plaquette + glueball correlator evidence.

    This is **numerical evidence** for a mass gap in the **lattice** theory at fixed
    spacing; it does not constitute a continuum Yang-Mills mass-gap proof.
    """
    if gauge_group != "su(2)":
        msg = f"only gauge_group='su(2)' is implemented, got {gauge_group!r}"
        raise ValueError(msg)

    t0 = time.perf_counter()
    dev = torch.device(device)
    gen = torch.Generator(device=dev) if dev.type == "cuda" else torch.Generator()
    gen.manual_seed(seed)

    shape = tuple(int(s) for s in lattice_shape)
    spatial_min = min(shape[0], shape[1], shape[2])
    r_lim = min(4, spatial_min // 2) if r_max is None else min(r_max, spatial_min // 2)
    r_lim = max(r_lim, 1)

    links = random_links(shape, device=dev, dtype=torch.float64, generator=gen)

    for _ in range(n_therm):
        sweep(links, beta, generator=gen)

    plaquette_samples: list[float] = []
    polyakov_samples: list[float] = []
    operator_samples: list[torch.Tensor] = []
    gevp_operator_samples: list[torch.Tensor] = []
    wilson_samples: dict[tuple[int, int], list[float]] = {
        (r_val, t_val): [] for r_val in range(1, r_lim + 1) for t_val in range(1, r_lim + 1)
    }

    for _ in range(n_meas):
        for _ in range(n_sep):
            sweep(links, beta, generator=gen)
        plaquette_samples.append(average_plaquette(links))
        polyakov_samples.append(average_polyakov_loop(links))
        operator_samples.append(
            glueball_operator_timeslice(
                links,
                smeared=True,
                n_smear=n_smear,
                smear_alpha=smear_alpha,
            )
        )
        gevp_ops = [
            glueball_operator_timeslice(
                links,
                smeared=True,
                n_smear=level,
                smear_alpha=smear_alpha,
            )
            for level in gevp_levels
        ]
        gevp_operator_samples.append(torch.stack(gevp_ops))
        for r_val in range(1, r_lim + 1):
            for t_val in range(1, r_lim + 1):
                wilson_samples[(r_val, t_val)].append(average_wilson_loop(links, r_val, t_val))

    avg_p = float(sum(plaquette_samples) / len(plaquette_samples))
    p_err = jackknife_std(plaquette_samples)

    o_samples = torch.stack(operator_samples)
    corr_mean, corr_err = connected_correlator_ensemble(o_samples)
    corr_list = [float(x) for x in corr_mean.tolist()]
    corr_jk = [float(x) for x in corr_err.tolist()]

    gevp_o = torch.stack(gevp_operator_samples)
    gevp_out = gevp_ground_mass(
        gevp_o,
        smear_levels=gevp_levels,
        t0=0,
        dt=1,
        smear_alpha=smear_alpha,
    )
    gevp_plateau_out = gevp_plateau(
        gevp_o,
        smear_levels=gevp_levels,
        t0_values=(0, 1, 2),
        dt_values=(1, 2, 3),
        smear_alpha=smear_alpha,
    )

    avg_poly = float(sum(polyakov_samples) / len(polyakov_samples))
    poly_err = jackknife_std(polyakov_samples)

    wilson_out = wilson_loops_ensemble(wilson_samples)
    creutz_out = creutz_ratios_ensemble(wilson_samples)
    sigma_out = string_tension_from_creutz(creutz_out)

    elapsed = time.perf_counter() - t0

    return {
        "gauge_group": gauge_group,
        "lattice_shape": list(shape),
        "beta": float(beta),
        "n_therm": int(n_therm),
        "n_meas": int(n_meas),
        "n_sep": int(n_sep),
        "device": str(dev),
        "avg_plaquette": avg_p,
        "avg_plaquette_err": p_err,
        "avg_polyakov": avg_poly,
        "avg_polyakov_err": poly_err,
        "glueball_correlator": corr_list,
        "glueball_correlator_err": corr_jk,
        "smearing": {
            "n_smear": int(n_smear),
            "smear_alpha": float(smear_alpha),
        },
        "wilson_loops": wilson_out,
        "creutz_ratios": creutz_out,
        "string_tension": sigma_out,
        "gevp": gevp_out,
        "gevp_plateau": gevp_plateau_out,
        "evidence_note": EVIDENCE_NOTE,
        "seed": int(seed),
        "elapsed_s": float(elapsed),
    }
