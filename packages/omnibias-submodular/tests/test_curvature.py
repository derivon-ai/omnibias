# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Total curvature: c in [0,1], the sharpened ratio (1/c)(1-e^{-c}) >= 1-1/e, modular c=0."""

from __future__ import annotations

from math import exp

import numpy as np
from omnibias.submodular import (
    ONE_MINUS_INV_E,
    BudgetAdditive,
    Coverage,
    FacilityLocation,
    UniformMatroid,
    certify_submodular_gap,
    greedy_maximize,
    max_coverage,
    total_curvature,
)


def _coverage(seed: int) -> Coverage:
    rng = np.random.default_rng(seed)
    return Coverage((rng.random((10, 8)) < 0.35).astype(float), rng.random(10) + 0.3)


def test_curvature_is_in_unit_interval() -> None:
    makers = [
        lambda s: _coverage(s),
        lambda s: FacilityLocation(np.random.default_rng(s).random((9, 8))),
    ]
    for maker in makers:
        for seed in range(6):
            c = total_curvature(maker(seed))
            assert 0.0 <= c <= 1.0


def test_modular_function_has_zero_curvature_and_unit_ratio() -> None:
    # BudgetAdditive with a never-binding budget is modular (marginals never shrink).
    a = np.array([0.5, 1.0, 1.5, 2.0, 0.7])
    fn = BudgetAdditive(a, budget=float(a.sum() + 10.0))
    c = total_curvature(fn)
    assert c < 1e-9
    # curvature_ratio -> 1 at c = 0.
    prob = max_coverage([[0], [1], [2]], k=2)  # any problem to build a certificate shell
    cert = certify_submodular_gap(prob, greedy_maximize(prob.function, prob.matroid)[0])
    cert_modular = cert.__class__(
        value=cert.value,
        upper_bound=cert.upper_bound,
        fractional_value=None,
        approx_ratio=ONE_MINUS_INV_E,
        method="marginal",
        curvature=c,
    )
    assert abs(cert_modular.curvature_ratio - 1.0) < 1e-9


def test_saturating_coverage_has_full_curvature() -> None:
    # Every set covers the whole universe: the second element onward adds nothing (c = 1).
    fn = Coverage(np.ones((6, 5)), np.arange(1.0, 7.0))
    c = total_curvature(fn)
    assert abs(c - 1.0) < 1e-9


def test_curvature_ratio_dominates_one_minus_inv_e() -> None:
    for seed in range(6):
        prob = max_coverage(
            [list(np.where(row > 0)[0]) for row in _coverage(seed).membership.T], k=4
        )
        sel, _ = greedy_maximize(prob.function, prob.matroid)
        cert = certify_submodular_gap(prob, sel, with_curvature=True)
        assert cert.curvature is not None
        assert 0.0 <= cert.curvature <= 1.0
        assert cert.curvature_ratio >= ONE_MINUS_INV_E - 1e-12
        assert cert.curvature_ratio <= 1.0 + 1e-12


def test_certify_without_flag_leaves_curvature_none() -> None:
    prob = max_coverage([[0, 1], [1, 2], [2, 3], [0, 3]], k=2)
    sel, _ = greedy_maximize(prob.function, prob.matroid)
    cert = certify_submodular_gap(prob, sel)
    assert cert.curvature is None
    assert cert.curvature_ratio == cert.approx_ratio  # falls back to the a-priori 1-1/e


def test_curvature_ratio_formula_matches_closed_form() -> None:
    prob = max_coverage([[0, 1], [1, 2], [2, 3], [0, 3], [1, 3]], k=3)
    sel, _ = greedy_maximize(prob.function, prob.matroid)
    cert = certify_submodular_gap(prob, sel, with_curvature=True)
    c = cert.curvature
    assert c is not None
    if c > 1e-12:
        assert abs(cert.curvature_ratio - (1.0 / c) * (1.0 - exp(-c))) < 1e-12


def test_total_curvature_accepts_ground_indicator() -> None:
    fn = _coverage(0)
    full = total_curvature(fn, np.ones(fn.n))
    default = total_curvature(fn)
    assert abs(full - default) < 1e-12
    # A sub-ground-set is accepted (indices or indicator) and stays in [0, 1].
    sub = total_curvature(fn, [0, 1, 2, 3])
    assert 0.0 <= sub <= 1.0


def test_uniform_matroid_certificate_curvature_is_consistent() -> None:
    fn = _coverage(1)
    matroid = UniformMatroid(fn.n, 3)
    from omnibias.submodular import SubmodularProblem

    prob = SubmodularProblem(fn, matroid)
    sel, _ = greedy_maximize(fn, matroid)
    cert = certify_submodular_gap(prob, sel, with_curvature=True)
    assert cert.curvature == total_curvature(fn)
