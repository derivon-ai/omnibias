# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Torch lattice backend (the default ``omnibias.geometry.gauge.lattice`` implementation).

This namespace re-exports the torch SU(2) Monte-Carlo surface so that it mirrors
:mod:`omnibias.geometry.gauge.lattice.jax`; the deterministic array math is shared
bit-identically through :mod:`omnibias.geometry.gauge.lattice._core.kernels`.
"""

from __future__ import annotations

from omnibias.geometry.gauge.lattice.langevin import langevin_sweep, langevin_update_links
from omnibias.geometry.gauge.lattice.montecarlo import run_lattice_mc
from omnibias.geometry.gauge.lattice.observables import (
    ape_smear_spatial_links,
    average_plaquette,
    average_polyakov_loop,
    average_wilson_loop,
    connected_correlator_ensemble,
    connected_correlator_matrix_ensemble,
    gauge_orbit_distance,
    gauge_transform_links,
    gevp_ground_mass,
    gevp_plateau,
    glueball_operator_timeslice,
    plaquette_trace,
    polyakov_loop,
    wilson_loop_trace,
)
from omnibias.geometry.gauge.lattice.su2 import (
    heatbath_update_links,
    identity_links,
    normalize_quaternion,
    overrelax_update_links,
    quat_mul,
    quat_to_matrix,
    random_links,
    staple_sum,
    sweep,
)

__all__ = [
    "ape_smear_spatial_links",
    "average_plaquette",
    "average_polyakov_loop",
    "average_wilson_loop",
    "connected_correlator_ensemble",
    "connected_correlator_matrix_ensemble",
    "gauge_orbit_distance",
    "gauge_transform_links",
    "gevp_ground_mass",
    "gevp_plateau",
    "glueball_operator_timeslice",
    "heatbath_update_links",
    "identity_links",
    "langevin_sweep",
    "langevin_update_links",
    "normalize_quaternion",
    "overrelax_update_links",
    "plaquette_trace",
    "polyakov_loop",
    "quat_mul",
    "quat_to_matrix",
    "random_links",
    "run_lattice_mc",
    "staple_sum",
    "sweep",
    "wilson_loop_trace",
]
