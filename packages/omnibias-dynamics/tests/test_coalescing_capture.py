# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
"""Exact χ-scale identities; not a Hilbert XVI or G1 certificate."""

from __future__ import annotations

import math
from fractions import Fraction

import pytest


def test_linear_exit_and_chi_threshold_are_exact() -> None:
    sep, radius, kappa = Fraction(1, 5), Fraction(2), Fraction(10)
    chi = sep * kappa / radius
    assert chi == Fraction(1)
    # X_exit = exp(-chi); the algebraic threshold identity is exact.
    assert sep * (Fraction(7) / sep) / radius == Fraction(7) / radius
    assert sep * (chi * radius / sep) / radius == chi


def test_root_sum_product_and_shrinking_wall() -> None:
    lam1, sep, theta = Fraction(-4), Fraction(2), Fraction(1, 5)
    r1 = (-lam1 - sep) / 2
    r2 = (-lam1 + sep) / 2
    rstar = -lam1 / 2
    assert r1 + r2 == -lam1
    assert r1 * r2 == (lam1**2 - sep**2) / 4
    a = r1 - theta * sep
    wall = (a - r1) * (a - r2)
    assert wall == theta * (1 + theta) * sep**2
    assert 2 * (a - rstar) == (a - r1) + (a - r2)


def test_frozen_gamma_fails_on_the_lean_witness() -> None:
    C, gamma, radius = 1.0, 1.0, 1.0
    sep = min(1.0, radius * gamma / 2.0)
    alpha = gamma - sep / radius
    kappa = (radius * C / sep + 1.0) / alpha
    left = C * math.exp(-gamma * kappa)
    right = (sep / radius) * math.exp(-(sep / radius) * kappa)
    assert left < right


@pytest.mark.parametrize("sep_small", [0.25, 0.05, 0.01])
def test_smaller_separation_makes_frozen_ratio_diverge(sep_small: float) -> None:
    C, gamma, radius, kappa = 1.0, 1.0, 1.0, 12.0
    ratio = ((sep_small / radius) * math.exp(-(sep_small / radius) * kappa)) / (
        C * math.exp(-gamma * kappa)
    )
    assert ratio > 1.0
