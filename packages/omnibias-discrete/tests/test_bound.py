# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Lower bounds: Gershgorin, the always-valid negative-coeff floor, and Lasserre/SOS."""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.discrete import (
    brute_force_min,
    gershgorin_min_eig_lower,
    lasserre_lower_bound,
    negative_coeff_lower_bound,
)


def test_gershgorin_bounds_the_smallest_eigenvalue() -> None:
    rng = np.random.default_rng(0)
    for _ in range(10):
        m = rng.standard_normal((5, 5))
        sym = m + m.T
        assert gershgorin_min_eig_lower(sym) <= float(np.min(np.linalg.eigvalsh(sym))) + 1e-9


def test_negative_coeff_bound_is_sound_over_the_cube(make_toy) -> None:  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(2)
    for _ in range(8):
        m = rng.standard_normal((4, 4))
        prob = make_toy(m + m.T, c=rng.standard_normal(4), const=float(rng.standard_normal()))
        floor = negative_coeff_lower_bound(prob.to_polynomial())
        _, e_min = brute_force_min(prob)
        assert floor <= e_min + 1e-9  # never exceeds the true minimum


def test_lasserre_bound_is_sound_and_sealed(make_toy) -> None:  # type: ignore[no-untyped-def]
    pytest.importorskip("omnibias.sos")
    prob = make_toy([[1.0, -1.0], [-1.0, 1.0]])  # E = (x0 - x1)^2 >= 0 on the cube
    _, e_min = brute_force_min(prob)
    floor = negative_coeff_lower_bound(prob.to_polynomial())
    result = lasserre_lower_bound(prob, level=1, seed_lower=floor, upper=e_min + 1.0, steps=24)
    assert result is not None
    gamma, sealed = result
    assert gamma <= e_min + 1e-6  # valid lower bound
    assert gamma >= floor - 1e-9  # and at least as tight as the trivial floor
    assert sealed is not None
