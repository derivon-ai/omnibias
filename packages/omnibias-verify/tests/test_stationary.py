# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Certified critical points (interval-Newton) and certified basin flatness.

Himmelblau is the reference: it has exactly nine stationary points -- four global
minima (value 0), four saddles, and one local maximum -- so it exercises every
branch of the classifier and the Krawczyk existence/uniqueness test.  Flatness
tests check the rigorous Hessian-eigenvalue enclosures.
"""

from __future__ import annotations

import math
from collections import Counter

from omnibias.core.verified.interval import Interval
from omnibias.verify import (
    CriticalPoint,
    FlatnessResult,
    certified_critical_points,
    certified_flatness,
)

P = Interval.point


def himmelblau_grad(b):  # type: ignore[no-untyped-def]
    x, y = b
    a = x**2 + y - P(11.0)
    c = x + y**2 - P(7.0)
    return (P(4.0) * x * a + P(2.0) * c, P(2.0) * a + P(4.0) * y * c)


def himmelblau_hess(b):  # type: ignore[no-untyped-def]
    x, y = b
    a = x**2 + y - P(11.0)
    c = x + y**2 - P(7.0)
    hxx = P(4.0) * a + P(8.0) * x**2 + P(2.0)
    hxy = P(4.0) * x + P(4.0) * y
    hyy = P(2.0) + P(4.0) * c + P(8.0) * y**2
    return [[hxx, hxy], [hxy, hyy]]


def test_himmelblau_all_nine_critical_points_certified() -> None:
    cps = certified_critical_points(himmelblau_grad, himmelblau_hess, [(-5.0, 5.0), (-5.0, 5.0)], tol=1e-10)
    kinds = Counter(c.kind for c in cps)
    assert kinds["min"] == 4, kinds
    assert kinds["saddle"] == 4, kinds
    assert kinds["max"] == 1, kinds
    assert all(c.unique for c in cps)  # Krawczyk proved existence + uniqueness for each
    # every reported point is a genuine root of the gradient
    for c in cps:
        gx, gy = himmelblau_grad((P(c.point[0]), P(c.point[1])))
        assert max(abs(gx.mid), abs(gy.mid)) < 1e-6


def test_himmelblau_minima_match_known_locations() -> None:
    cps = certified_critical_points(himmelblau_grad, himmelblau_hess, [(-5.0, 5.0), (-5.0, 5.0)], tol=1e-10)
    known = [(3.0, 2.0), (-2.805118, 3.131312), (-3.779310, -3.283186), (3.584428, -1.848126)]
    minima = [c for c in cps if c.kind == "min"]
    for mx, my in known:
        assert any(math.hypot(c.point[0] - mx, c.point[1] - my) < 1e-4 for c in minima)
    # a certified min has a certified-positive-definite Hessian (eig_min > 0)
    assert all(c.eig_min > 0.0 for c in minima)


def test_saddles_are_indefinite() -> None:
    cps = certified_critical_points(himmelblau_grad, himmelblau_hess, [(-5.0, 5.0), (-5.0, 5.0)], tol=1e-10)
    for c in cps:
        if c.kind == "saddle":
            assert c.eig_min < 0.0 < c.eig_max  # a negative and a positive curvature direction


def test_certified_flatness_diagonal_hessian() -> None:
    # f = x^2 + 3 y^2  ->  Hessian = diag(2, 6); eigenvalues exactly {2, 6}
    def hess(b):  # type: ignore[no-untyped-def]
        return [[P(2.0), P(0.0)], [P(0.0), P(6.0)]]

    fr = certified_flatness(hess, [(-1.0, 1.0), (-1.0, 1.0)])
    assert isinstance(fr, FlatnessResult)
    assert fr.eig_min.lo <= 2.0 <= fr.eig_min.hi
    assert fr.eig_max.lo <= 6.0 <= fr.eig_max.hi
    assert fr.certified_positive_definite
    assert abs(fr.sharpness - 6.0) < 1e-3
    assert abs(fr.width_lower_bound - 6.0**-0.5) < 1e-3


def test_flatness_ranks_equal_depth_minima() -> None:
    # all four Himmelblau minima have value 0 but different curvature; flatness
    # (largest Hessian eigenvalue) must distinguish them.
    cps = certified_critical_points(himmelblau_grad, himmelblau_hess, [(-5.0, 5.0), (-5.0, 5.0)], tol=1e-10)
    minima = [c for c in cps if c.kind == "min"]
    sharp = sorted(c.eig_max for c in minima)
    assert sharp[0] < sharp[-1] - 1.0  # basins have materially different worst-case sharpness
    # the (3, 2) basin has the single flattest direction (smallest Hessian eigenvalue)
    flattest_dir = min(minima, key=lambda c: c.eig_min)
    assert math.hypot(flattest_dir.point[0] - 3.0, flattest_dir.point[1] - 2.0) < 1e-3


def test_critical_point_dataclass_shape() -> None:
    cps = certified_critical_points(himmelblau_grad, himmelblau_hess, [(-5.0, 5.0), (-5.0, 5.0)], tol=1e-10)
    c = cps[0]
    assert isinstance(c, CriticalPoint)
    assert len(c.point) == 2
    assert len(c.box) == 2 and all(len(iv) == 2 for iv in c.box)
    assert c.kind in {"min", "max", "saddle", "indefinite"}
