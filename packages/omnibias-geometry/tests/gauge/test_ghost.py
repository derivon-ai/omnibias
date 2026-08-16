# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Landau ghost: free identity and planted decoupling."""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.geometry.gauge._core.data_paths import LatticeLinkField
from omnibias.geometry.gauge._core.ghost import (
    ghost_propagator_ensemble,
    ghost_propagator_p2,
    planted_ghost_table,
)
from omnibias.geometry.gauge._core.loop_language import identity_numpy_links
from omnibias.symbolic.ensemble_families import NamedFamilyDiscoverer


def test_identity_ghost_is_free_field() -> None:
    field = LatticeLinkField(links=identity_numpy_links((2, 2, 2, 2)))
    table, report = ghost_propagator_p2(field, already_fixed=True)
    p2 = table.values["p2"]
    ghost = table.values["ghost_G"]
    np.testing.assert_allclose(ghost * p2, np.ones_like(p2), rtol=1e-5, atol=1e-5)
    assert report["yang_mills_claim"] is False
    assert report["continuum_claim"] is False


def test_ghost_ensemble_refuses_one_config() -> None:
    field = LatticeLinkField(links=identity_numpy_links((2, 2, 2, 2)))
    with pytest.raises(ValueError, match="single"):
        ghost_propagator_ensemble([field], already_fixed=True)


def test_planted_ghost_recovers_decoupling() -> None:
    table = planted_ghost_table(mass2=0.5, z0=1.0)
    # Named family reads G_p2; copy ghost into that slot.
    from omnibias.geometry.gauge._core.ensemble_language import EnsembleObservableTable

    fitted = EnsembleObservableTable(
        values={"p2": table.values["p2"], "G_p2": table.values["ghost_G"]},
        source="planted",
    )
    out = NamedFamilyDiscoverer().fit(fitted, family="decoupling")
    assert out.passed is True
    assert out.parameters["M2"] == pytest.approx(0.5, rel=5e-2)
    assert out.yang_mills_claim is False
