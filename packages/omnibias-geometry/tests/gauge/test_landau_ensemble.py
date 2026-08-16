# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Ensemble Landau gluon: two configs required, identity is flat."""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.geometry.gauge._core.data_paths import LatticeLinkField
from omnibias.geometry.gauge._core.landau_gluon import gluon_propagator_ensemble
from omnibias.geometry.gauge._core.loop_language import (
    identity_numpy_links,
    random_numpy_links,
)


def test_one_config_is_refused() -> None:
    field = LatticeLinkField(links=identity_numpy_links((2, 2, 2, 2)))
    with pytest.raises(ValueError, match="single"):
        gluon_propagator_ensemble([field], already_fixed=True)


def test_two_identity_configs_are_flat() -> None:
    fields = [
        LatticeLinkField(links=identity_numpy_links((4, 4, 4, 4))),
        LatticeLinkField(links=identity_numpy_links((4, 4, 4, 4))),
    ]
    table, report = gluon_propagator_ensemble(fields, already_fixed=True)
    assert table.metadata.n_configs == 2
    assert table.metadata.scheme == "landau"
    assert float(np.max(np.abs(table.values["G_p2"]))) < 1e-12
    assert report["yang_mills_claim"] is False
    assert report["continuum_claim"] is False


def test_two_random_configs_bin_p2() -> None:
    rng = np.random.default_rng(5)
    fields = [
        LatticeLinkField(links=random_numpy_links((4, 4, 4, 4), rng)),
        LatticeLinkField(links=random_numpy_links((4, 4, 4, 4), rng)),
    ]
    table, report = gluon_propagator_ensemble(fields, n_steps=8, omega=1.0)
    assert table.values["p2"].shape == table.values["G_p2"].shape
    assert table.values["p2"].shape[0] >= 2
    assert report["n_configs"] == 2
