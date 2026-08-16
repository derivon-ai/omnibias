# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""SU(2) Wilson flow energy decreases on a 4^4 smoke lattice."""

from __future__ import annotations

import numpy as np
from omnibias.geometry.gauge._core.data_paths import LatticeLinkField
from omnibias.geometry.gauge._core.loop_language import (
    identity_numpy_links,
    random_numpy_links,
)
from omnibias.geometry.gauge.lattice._core.wilson_flow import (
    mean_plaquette_energy,
    run_wilson_flow,
    wilson_flow_scales_from_curve,
)


def test_identity_energy_is_zero() -> None:
    links = identity_numpy_links((4, 4, 4, 4))
    assert mean_plaquette_energy(links) == 0.0 or abs(mean_plaquette_energy(links)) < 1e-12


def test_flow_decreases_energy_on_random_4_4() -> None:
    rng = np.random.default_rng(3)
    field = LatticeLinkField(links=random_numpy_links((4, 4, 4, 4), rng))
    out = run_wilson_flow(field, n_steps=6, eps=0.03)
    assert out["energy_decreased"] is True
    assert out["yang_mills_claim"] is False
    assert out["continuum_claim"] is False
    assert out["energy"][-1] <= out["energy"][0] + 1e-12


def test_planted_curve_scales() -> None:
    t = np.linspace(0.05, 3.0, 40)
    energy = 2.0 * np.exp(-t)
    scales = wilson_flow_scales_from_curve(t, energy)
    assert scales["t0"].value > 0.0
    assert scales["w0"].yang_mills_claim is False
