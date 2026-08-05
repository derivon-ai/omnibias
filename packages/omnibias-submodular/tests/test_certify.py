# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""The certified approximation / gap sandwich and its self-check against the oracle."""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.submodular import (
    ONE_MINUS_INV_E,
    Coverage,
    FacilityLocation,
    PartitionMatroid,
    SubmodularProblem,
    UniformMatroid,
    brute_force_max,
    budget_additive,
    certify_submodular_gap,
    certify_unconstrained_gap,
    marginal_upper_bound,
    maximize,
    modular_upper_bound,
    verify_guarantee,
)


def _problems():
    out = []
    for seed in range(6):
        r = np.random.default_rng(seed)
        cov = Coverage((r.random((8, 6)) < 0.4).astype(float), r.random(8) + 0.3)
        out.append(SubmodularProblem(cov, UniformMatroid(6, 3)))
        out.append(SubmodularProblem(cov, PartitionMatroid([[0, 1, 2], [3, 4, 5]], [1, 2])))
        fac = FacilityLocation(r.random((5, 6)), r.random(5) + 0.2)
        out.append(SubmodularProblem(fac, UniformMatroid(6, 2)))
    out.append(budget_additive(np.array([0.4, 0.6, 0.5, 0.9, 0.3, 0.7]), budget=1.3, k=3))
    return out


def test_sandwich_is_sound_and_contains_the_optimum() -> None:
    for prob in _problems():
        sol = maximize(prob, rounding="pipage")
        cert = certify_submodular_gap(prob, sol.selection, fractional=sol.fractional)
        _, opt = brute_force_max(prob.function, prob.matroid)
        assert cert.value <= opt + 1e-9  # decoded value is a lower bound on OPT
        assert opt <= cert.upper_bound + 1e-9  # U(S) is an upper bound on OPT
        assert cert.internal_consistent
        assert cert.absolute_gap >= -1e-9
        assert 0.0 <= cert.certified_ratio <= 1.0 + 1e-12


def test_one_minus_inv_e_guarantee_holds() -> None:
    for prob in _problems():
        sol = maximize(prob, rounding="pipage")
        _, opt = brute_force_max(prob.function, prob.matroid)
        assert sol.value >= ONE_MINUS_INV_E * opt - 1e-9
        cert = certify_submodular_gap(prob, sol.selection, fractional=sol.fractional)
        assert abs(cert.approx_ratio - ONE_MINUS_INV_E) < 1e-12


def test_min_of_bounds_is_sound_and_tighter_or_equal() -> None:
    for prob in _problems():
        sol = maximize(prob, rounding="pipage")
        xv = np.asarray(sol.selection, dtype=float)
        m_bound = marginal_upper_bound(prob.function, prob.matroid, xv)
        mod_bound = modular_upper_bound(prob.function, prob.matroid)
        _, opt = brute_force_max(prob.function, prob.matroid)
        assert opt <= m_bound + 1e-9  # each bound is individually sound
        assert opt <= mod_bound + 1e-9
        cert = certify_submodular_gap(prob, sol.selection, fractional=sol.fractional)
        # the certificate reports the tighter min of the two, and it stays >= OPT
        assert cert.upper_bound == pytest.approx(min(m_bound, mod_bound), abs=1e-12)
        assert cert.upper_bound <= m_bound + 1e-12
        assert opt <= cert.upper_bound + 1e-9


def test_verify_guarantee_passes_for_the_decoded_set() -> None:
    for prob in _problems():
        sol = maximize(prob)
        assert verify_guarantee(prob, sol.selection)


def test_certify_rejects_non_binary_or_infeasible() -> None:
    prob = SubmodularProblem(Coverage(np.eye(4)), UniformMatroid(4, 2))
    with pytest.raises(ValueError, match="0/1"):
        certify_submodular_gap(prob, np.array([0.5, 0.5, 0.0, 0.0]))
    with pytest.raises(ValueError, match="feasible"):
        certify_submodular_gap(prob, np.array([1.0, 1.0, 1.0, 0.0]))  # over cardinality


def test_unconstrained_sos_passthrough_is_sound() -> None:
    pytest.importorskip("omnibias.sos")
    # For monotone f the unconstrained minimum energy (-f) is the full set; the substrate
    # SOS bound must sandwich it soundly. Small n keeps the polynomial cheap.
    prob = SubmodularProblem(Coverage((np.random.default_rng(0).random((5, 4)) < 0.5).astype(float)), UniformMatroid(4, 4))
    full = np.ones(4)
    cert = certify_unconstrained_gap(prob, full, level=1, bisection_steps=16)
    assert cert.is_sound  # a discrete.GapCertificate, which keeps the shared name
    assert cert.lower_bound <= cert.energy + 1e-6
