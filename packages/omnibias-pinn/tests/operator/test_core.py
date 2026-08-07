# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Backend-free operator-learning schemas."""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.pinn import ComponentSpec, CoordinateSpec
from omnibias.pinn.operator import OperatorSpec, SensorGrid, sample_fourier_ics


def test_operator_spec_rejects_nonpositive_sizes() -> None:
    cs = CoordinateSpec(("x", "t"))
    comps = ComponentSpec(("u",))
    with pytest.raises(ValueError, match="n_sensors"):
        OperatorSpec(cs, comps, n_sensors=0, trunk_width=4)
    with pytest.raises(ValueError, match="trunk_width"):
        OperatorSpec(cs, comps, n_sensors=8, trunk_width=0)


def test_sensor_grid_uniform_1d() -> None:
    g = SensorGrid.uniform_1d(16, length=2.0 * np.pi)
    assert g.n_sensors == 16
    assert g.points.shape == (16,)
    assert float(g.points[0]) == 0.0
    assert float(g.points[-1]) < 2.0 * np.pi


def test_sample_fourier_ics_is_deterministic() -> None:
    g = SensorGrid.uniform_1d(32)
    a = sample_fourier_ics(5, g, n_modes=4, seed=7)
    b = sample_fourier_ics(5, g, n_modes=4, seed=7)
    c = sample_fourier_ics(5, g, n_modes=4, seed=8)
    assert a.shape == (5, 32)
    np.testing.assert_array_equal(a, b)
    assert not np.allclose(a, c)
