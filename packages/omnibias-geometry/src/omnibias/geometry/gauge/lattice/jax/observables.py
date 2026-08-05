# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Lattice observables (JAX backend): plaquette, Wilson/Polyakov loops, glueball.

The deterministic array math is delegated to
:mod:`omnibias.geometry.gauge.lattice._core.kernels` and the pure-Python jackknife / Creutz
helpers to :mod:`omnibias.geometry.gauge.lattice._core.stats`, so this module and its torch
twin are bit-identical on identical configs.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax import Array
from omnibias.geometry.gauge.lattice._core import kernels
from omnibias.geometry.gauge.lattice._core.stats import (
    creutz_ratios_ensemble,
    effective_mass,
    ensemble_mean_jackknife,
    jackknife_std,
    string_tension_from_creutz,
    wilson_loops_ensemble,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


def plaquette_trace(links: Array, mu: int, nu: int) -> Array:
    """``(1/2) Re tr U_plaquette`` at each site (mu, nu must differ)."""
    return kernels.plaquette_trace(jnp, links, mu, nu)


def average_plaquette(links: Array) -> float:
    """Mean ``(1/2) Re tr U`` over all distinct plaquettes."""
    vals = [
        kernels.plaquette_trace(jnp, links, mu, nu)
        for mu in range(4)
        for nu in range(mu + 1, 4)
    ]
    return float(jnp.stack(vals).mean())


def ape_smear_spatial_links(
    links: Array,
    *,
    n_steps: int = 10,
    alpha: float = 0.5,
) -> Array:
    """APE smear spatial links (mu=0,1,2); project back to SU(2) by normalization."""
    return kernels.ape_smear_spatial_links(jnp, links, n_steps=n_steps, alpha=alpha)


def glueball_operator_timeslice(
    links: Array,
    *,
    smeared: bool = True,
    n_smear: int = 10,
    smear_alpha: float = 0.5,
) -> Array:
    """0++ operator ``O(t)``: sum over spatial volume of spatial plaquette traces."""
    return kernels.glueball_operator_timeslice(
        jnp, links, smeared=smeared, n_smear=n_smear, smear_alpha=smear_alpha
    )


def polyakov_loop(links: Array, *, t_dir: int = 3) -> Array:
    """Per-site ``(1/2) Re tr P(x)`` of the Polyakov loop winding the time axis."""
    return kernels.polyakov_loop_field(jnp, links, t_dir=t_dir)


def average_polyakov_loop(links: Array, *, t_dir: int = 3) -> float:
    """Volume-averaged Polyakov loop (order parameter; ~0 in the confined phase)."""
    return float(kernels.polyakov_loop_field(jnp, links, t_dir=t_dir).mean())


def gauge_transform_links(links: Array, g: Array) -> Array:
    r"""Apply a lattice gauge transformation ``U_mu(x) -> g(x) U_mu(x) g(x+mu)^\dagger``."""
    return kernels.gauge_transform_links(jnp, links, g)


def gauge_orbit_distance(links_a: Array, links_b: Array) -> float:
    """Gauge-invariant orbit-distance proxy (RMS plaquette-trace difference)."""
    return kernels.gauge_orbit_distance(jnp, links_a, links_b)


def connected_correlator_ensemble(
    o_samples: Array,
) -> tuple[Array, Array]:
    """Ensemble connected glueball correlator with global vacuum subtraction."""
    return kernels.connected_correlator_ensemble(jnp, o_samples)


def wilson_loop_trace(
    links: Array,
    mu: int,
    r_extent: int,
    t_extent: int,
    *,
    t_dir: int = 3,
) -> Array:
    """Planar ``R x T`` Wilson loop trace ``(1/2) Re tr W`` in the ``mu``-temporal plane."""
    return kernels.wilson_loop_trace(jnp, links, mu, r_extent, t_extent, t_dir=t_dir)


def average_wilson_loop(
    links: Array,
    r_extent: int,
    t_extent: int,
    *,
    t_dir: int = 3,
) -> float:
    """Mean ``(1/2) Re tr W`` over sites and spatial directions ``mu in {0,1,2}``."""
    spatial = (0, 1, 2)
    vals = [
        kernels.wilson_loop_trace(jnp, links, mu, r_extent, t_extent, t_dir=t_dir).mean()
        for mu in spatial
    ]
    return float(jnp.stack(vals).mean())


def connected_correlator_matrix_ensemble(
    o_samples: Array,
) -> tuple[Array, Array]:
    """Connected operator matrix ``C_ab(tau)`` with global vacuum subtraction."""
    return kernels.connected_correlator_matrix_ensemble(jnp, o_samples)


def gevp_ground_mass(
    o_samples: Array,
    *,
    smear_levels: Sequence[int],
    t0: int = 0,
    dt: int = 1,
    smear_alpha: float = 0.5,
) -> dict[str, object]:
    """Solve the small-basis GEVP and return the ground-state mass estimate."""
    c_mat, _ = kernels.connected_correlator_matrix_ensemble(jnp, o_samples)
    tau0 = t0
    tau1 = t0 + dt
    if tau1 >= c_mat.shape[-1]:
        msg = f"GEVP requires tau0+dt < T//2+1, got t0={t0}, dt={dt}, max_tau={c_mat.shape[-1] - 1}"
        raise ValueError(msg)

    lam = kernels.gevp_ground_lambda(jnp, c_mat[:, :, tau0], c_mat[:, :, tau1])
    mass = float("nan") if lam <= 0.0 or lam >= 1.0 else -math.log(lam) / dt

    n_meas = o_samples.shape[0]
    jk_masses: list[float] = []
    if n_meas >= 2:
        for i in range(n_meas):
            mask = jnp.arange(n_meas) != i
            c_jk, _ = kernels.connected_correlator_matrix_ensemble(jnp, o_samples[mask])
            lam_i = kernels.gevp_ground_lambda(jnp, c_jk[:, :, tau0], c_jk[:, :, tau1])
            if 0.0 < lam_i < 1.0:
                jk_masses.append(-math.log(lam_i) / dt)

    err = ensemble_mean_jackknife(jk_masses)[1] if len(jk_masses) >= 2 else float("nan")
    return {
        "levels": [{"n_smear": int(n), "smear_alpha": float(smear_alpha)} for n in smear_levels],
        "t0": int(t0),
        "dt": int(dt),
        "ground_mass": {"value": mass, "err": err},
    }


def gevp_plateau(
    o_samples: Array,
    *,
    smear_levels: Sequence[int],
    t0_values: Sequence[int],
    dt_values: Sequence[int],
    smear_alpha: float = 0.5,
    rel_tol: float = 0.25,
) -> dict[str, object]:
    """Scan ``(t0, dt)`` GEVP masses and report the plateau (fixed-spacing evidence)."""
    scan = kernels.gevp_plateau(
        jnp, o_samples, t0_values=t0_values, dt_values=dt_values, rel_tol=rel_tol
    )
    scan["levels"] = [
        {"n_smear": int(n), "smear_alpha": float(smear_alpha)} for n in smear_levels
    ]
    return scan


__all__ = [
    "ape_smear_spatial_links",
    "average_plaquette",
    "average_polyakov_loop",
    "average_wilson_loop",
    "connected_correlator_ensemble",
    "connected_correlator_matrix_ensemble",
    "creutz_ratios_ensemble",
    "effective_mass",
    "ensemble_mean_jackknife",
    "gauge_orbit_distance",
    "gauge_transform_links",
    "gevp_ground_mass",
    "gevp_plateau",
    "glueball_operator_timeslice",
    "jackknife_std",
    "plaquette_trace",
    "polyakov_loop",
    "string_tension_from_creutz",
    "wilson_loop_trace",
    "wilson_loops_ensemble",
]
