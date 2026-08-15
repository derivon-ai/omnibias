# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Hermite ladder G1–G3 (theory 02-10). G4 FermiNet is --full; G5 may lose."""

from __future__ import annotations

import math

import numpy as np
from omnibias.core.ladder import (
    Normalization,
    commutator_residual,
    hermite_function,
    number_operator_apply,
    oscillator_hamiltonian_apply,
    tower_lower,
    tower_raise,
)


def _ulp(a: float, b: float) -> float:
    scale = max(abs(a), abs(b), 1.0)
    return abs(a - b) / (np.finfo(np.float64).eps * scale)


def test_g1_ladder_exactness() -> None:
    x = 0.7
    for n in range(0, 21):
        h_n = hermite_function(n, x, normalization=Normalization.TOWER)
        raised = tower_raise(n, x)
        # finite-d check of -d/dx via 4th order is not the gate; closed form:
        assert _ulp(raised, hermite_function(n + 1, x, normalization=Normalization.TOWER)) <= 4.0
        lowered = tower_lower(n, x)
        if n == 0:
            assert lowered == 0.0
        else:
            expect = n * hermite_function(n - 1, x, normalization=Normalization.TOWER)
            assert _ulp(lowered, expect) <= 4.0
        _ = h_n


def test_g2_two_operators_and_negative() -> None:
    x = 0.7
    for n in range(0, 12):
        h = hermite_function(n, x, normalization=Normalization.TOWER)
        nh = number_operator_apply(n, x)
        assert _ulp(nh, n * h) <= 16.0
        psi = hermite_function(n, x, normalization=Normalization.OSCILLATOR)
        hpsi = oscillator_hamiltonian_apply(n, x, normalization=Normalization.OSCILLATOR)
        assert _ulp(hpsi, (n + 0.5) * psi) <= 16.0
        h_on_tower = oscillator_hamiltonian_apply(n, x, normalization=Normalization.TOWER)
        # Negative: the raw tower is not a QHO eigenfunction.
        if n >= 2:
            assert _ulp(h_on_tower, (n + 0.5) * h) > 4.0


def test_g3_commutator() -> None:
    err = commutator_residual(20, x=0.7)
    scale = max(
        abs(hermite_function(n, 0.7, normalization=Normalization.TOWER))
        for n in range(21)
    )
    assert err / max(scale, 1.0) <= 1e-12


def test_truncation_boundary() -> None:
    assert tower_lower(0, 0.3) == 0.0
    assert math.isfinite(tower_raise(20, 0.3))
