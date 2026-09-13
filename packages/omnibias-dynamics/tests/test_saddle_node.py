# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
"""Exact double-root identities; not a C2 or G1 certificate."""

from __future__ import annotations

import math
from fractions import Fraction

import pytest


def test_double_root_quadratic_and_wall() -> None:
    rstar, x = Fraction(3, 2), Fraction(-1, 5)
    assert rstar**2 + (-2 * rstar) * x + x**2 == (x - rstar) ** 2
    assert (rstar - rstar) ** 2 == 0


def test_first_root_shrinks_with_L() -> None:
    lam1, sep = Fraction(-4), Fraction(2)
    L = (lam1**2 - sep**2) / 4
    r1 = (-lam1 - sep) / 2
    assert r1 * (-lam1 + sep) == 2 * L
    assert r1 == 2 * L / (-lam1 + sep)


def test_sigma_kappa_on_chi_locus() -> None:
    sigma, sep, chi, r1 = Fraction(1, 3), Fraction(1, 7), Fraction(2), Fraction(5)
    assert sigma * (chi * r1 / sep) == (sigma / sep) * chi * r1


def test_fold_scale_exit_height_is_eps_fourth() -> None:
    eps = Fraction(1, 16)
    sigma = Fraction(1, 4)  # sqrt(1/16)
    assert eps**3 * sigma**2 == eps**4


def test_kill_sequence_scale_tension() -> None:
    eps = 0.05
    sep = math.exp(-1.0 / (eps * eps))
    kappa = 1.0 / sep
    assert math.sqrt(eps) * kappa > 10.0
    assert sep * kappa == pytest.approx(1.0)


def test_shrinking_root_saddle_collides_with_center() -> None:
    lam1 = Fraction(-2)
    for n in (2, 5, 20):
        L = Fraction(1, n)
        assert lam1**2 - 4 * L > 0
        # Dropping the positive sep term overestimates r1 = 2L / (|lambda1| + sep).
        r1_upper = 2 * L / (-lam1)
        assert r1_upper == L
        assert 0 < r1_upper < 1
    eps = Fraction(1, 10)
    L = Fraction(1, 100)
    assert -eps * (2 * L / (-lam1)) == Fraction(-1, 1000)
