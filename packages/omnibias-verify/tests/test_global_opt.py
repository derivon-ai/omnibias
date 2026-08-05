# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Certified global minimization: soundness + convergence on standard non-convex tests.

Every assertion is of one of two kinds:

* **soundness** -- the returned ``f_lower`` is a *true* lower bound on the global
  minimum: a dense grid **and** a random sample of the objective over the box all
  lie at or above ``f_lower`` (the enclosure convention used throughout
  ``omnibias.core.verified``), and the known global-minimum value lies inside
  ``[f_lower, f_upper]``;
* **localization** -- the returned point sits near a known global minimizer.

The functions (six-hump camel, Himmelblau, Branin, Rastrigin, Ackley, Rosenbrock)
are the classic multimodal benchmarks; a *local* method would stall in the wrong
basin, whereas branch-and-bound certifies the global answer.  All float64.
"""

from __future__ import annotations

import math
import random

import pytest
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.transcend import cos_iv, exp_iv
from omnibias.verify import certified_minimize, certify_strict_local_min

P = Interval.point


# --------------------------- objective library ------------------------------


def quadratic(b):  # type: ignore[no-untyped-def]
    return b[0] * b[0] + b[1] * b[1]  # min 0 at origin


def quadratic_grad(b):  # type: ignore[no-untyped-def]
    return (P(2.0) * b[0], P(2.0) * b[1])


def camel(b):  # type: ignore[no-untyped-def]
    x, y = b
    return (P(4.0) - P(2.1) * x**2 + (x**4) * P(1.0 / 3.0)) * x**2 + x * y + (
        P(-4.0) + P(4.0) * y**2
    ) * y**2


def camel_grad(b):  # type: ignore[no-untyped-def]
    x, y = b
    dx = P(8.0) * x - P(8.4) * x**3 + P(2.0) * x**5 + y
    dy = x - P(8.0) * y + P(16.0) * y**3
    return (dx, dy)


def himmelblau(b):  # type: ignore[no-untyped-def]
    x, y = b
    return (x**2 + y - P(11.0)) ** 2 + (x + y**2 - P(7.0)) ** 2


def himmelblau_grad(b):  # type: ignore[no-untyped-def]
    x, y = b
    dx = P(4.0) * x * (x**2 + y - P(11.0)) + P(2.0) * (x + y**2 - P(7.0))
    dy = P(2.0) * (x**2 + y - P(11.0)) + P(4.0) * y * (x + y**2 - P(7.0))
    return (dx, dy)


def rosenbrock(b):  # type: ignore[no-untyped-def]
    x, y = b
    return (P(1.0) - x) ** 2 + P(100.0) * (y - x**2) ** 2  # min 0 at (1, 1)


def rosenbrock_grad(b):  # type: ignore[no-untyped-def]
    x, y = b
    dx = P(-2.0) * (P(1.0) - x) + P(200.0) * (y - x**2) * (P(-2.0) * x)
    dy = P(200.0) * (y - x**2)
    return (dx, dy)


_BRANIN_B = 5.1 / (4.0 * math.pi**2)
_BRANIN_C = 5.0 / math.pi
_BRANIN_T = 1.0 / (8.0 * math.pi)


def branin(b):  # type: ignore[no-untyped-def]
    x, y = b
    inner = y - P(_BRANIN_B) * x**2 + P(_BRANIN_C) * x - P(6.0)
    return inner**2 + P(10.0 * (1.0 - _BRANIN_T)) * cos_iv(x) + P(10.0)  # min 0.397887


def rastrigin(b):  # type: ignore[no-untyped-def]
    acc = P(10.0 * len(b))
    for xi in b:
        acc = acc + xi**2 - P(10.0) * cos_iv(P(2.0 * math.pi) * xi)
    return acc  # min 0 at origin


def _nonneg(iv):  # type: ignore[no-untyped-def]
    # a sum of squares is >= 0 by maths; outward rounding can push lo to -5e-324,
    # so clamp before sqrt (sound: the true value is non-negative).
    return Interval(max(0.0, iv.lo), iv.hi)


def ackley(b):  # type: ignore[no-untyped-def]
    x, y = b
    sq = _nonneg((x**2 + y**2) * P(0.5))
    term1 = P(-20.0) * exp_iv(P(-0.2) * sq.sqrt())
    term2 = -exp_iv(P(0.5) * (cos_iv(P(2.0 * math.pi) * x) + cos_iv(P(2.0 * math.pi) * y)))
    return term1 + term2 + P(20.0 + math.e)  # min 0 at origin


# --------------------------- soundness helper -------------------------------


def _assert_sound_lower_bound(f, box, f_lower, *, n_grid=41, n_rand=2000, seed=0):  # type: ignore[no-untyped-def]
    """Every grid + random sample of ``f`` over ``box`` is >= the certified ``f_lower``."""
    rng = random.Random(seed)
    los = [lo for lo, _ in box]
    his = [hi for _, hi in box]
    worst = math.inf
    # dense grid (2-D)
    for i in range(n_grid):
        for j in range(n_grid):
            x = los[0] + (his[0] - los[0]) * i / (n_grid - 1)
            y = los[1] + (his[1] - los[1]) * j / (n_grid - 1)
            val_lo = Interval.from_value(f((P(x), P(y)))).lo
            worst = min(worst, val_lo)
    # random sample
    for _ in range(n_rand):
        pt = tuple(P(lo + (hi - lo) * rng.random()) for lo, hi in zip(los, his, strict=True))
        val_lo = Interval.from_value(f(pt)).lo
        worst = min(worst, val_lo)
    # f_lower must under-estimate every attained value (tiny slack for rounding)
    assert worst >= f_lower - 1e-9, f"lower bound not sound: sample {worst} < f_lower {f_lower}"
    return worst


# --------------------------- tests ------------------------------------------


def test_convex_quadratic_converges_to_origin() -> None:
    r = certified_minimize(quadratic, [(-5.0, 5.0), (-5.0, 5.0)], tol=1e-6, grad=quadratic_grad)
    assert r.converged
    assert r.f_lower <= 0.0 <= r.f_upper
    assert abs(r.x[0]) < 1e-3 and abs(r.x[1]) < 1e-3
    _assert_sound_lower_bound(quadratic, [(-5.0, 5.0), (-5.0, 5.0)], r.f_lower)


def test_six_hump_camel_global_min() -> None:
    box = [(-2.0, 2.0), (-1.0, 1.0)]
    true_min = -1.0316284535
    r = certified_minimize(camel, box, tol=1e-3, max_boxes=200_000, grad=camel_grad)
    assert r.converged
    assert r.f_lower <= true_min <= r.f_upper  # enclosure contains the known global min
    # one of the two symmetric minimizers (+-0.0898, -+0.7126)
    assert abs(abs(r.x[0]) - 0.0898) < 5e-2 and abs(abs(r.x[1]) - 0.7126) < 5e-2
    _assert_sound_lower_bound(camel, box, r.f_lower)


def test_himmelblau_four_global_minima() -> None:
    box = [(-5.0, 5.0), (-5.0, 5.0)]
    r = certified_minimize(himmelblau, box, tol=1e-3, max_boxes=300_000, grad=himmelblau_grad)
    assert r.converged
    assert r.f_lower <= 0.0 <= r.f_upper
    minimizers = [(3.0, 2.0), (-2.805118, 3.131312), (-3.779310, -3.283186), (3.584428, -1.848126)]
    assert min(math.hypot(r.x[0] - mx, r.x[1] - my) for mx, my in minimizers) < 5e-2
    _assert_sound_lower_bound(himmelblau, box, r.f_lower)


def test_rosenbrock_curved_valley() -> None:
    box = [(-2.0, 2.0), (-1.0, 3.0)]
    r = certified_minimize(rosenbrock, box, tol=1e-3, max_boxes=300_000, grad=rosenbrock_grad)
    assert r.converged
    assert r.f_lower <= 0.0 <= r.f_upper
    assert abs(r.x[0] - 1.0) < 5e-2 and abs(r.x[1] - 1.0) < 5e-2
    _assert_sound_lower_bound(rosenbrock, box, r.f_lower)


def test_branin_three_global_minima() -> None:
    box = [(-5.0, 10.0), (0.0, 15.0)]
    true_min = 0.39788735772973816  # precise Branin global minimum
    r = certified_minimize(branin, box, tol=1e-2, max_boxes=300_000)
    # enclosure is sound regardless of convergence
    assert r.f_lower <= true_min <= r.f_upper
    _assert_sound_lower_bound(branin, box, r.f_lower)


def test_rastrigin_multimodal_global_min() -> None:
    # asymmetric box so the centre is not trivially the minimizer
    box = [(-4.0, 5.12), (-3.0, 5.12)]
    r = certified_minimize(rastrigin, box, tol=1e-2, max_boxes=500_000)
    assert r.converged
    assert r.f_lower <= 0.0 <= r.f_upper
    assert abs(r.x[0]) < 5e-2 and abs(r.x[1]) < 5e-2  # global min at origin, not a nearby local one
    _assert_sound_lower_bound(rastrigin, box, r.f_lower)


def test_ackley_multimodal_global_min() -> None:
    box = [(-5.0, 5.0), (-5.0, 5.0)]
    r = certified_minimize(ackley, box, tol=1e-2, max_boxes=500_000)
    assert r.f_lower <= 0.0 <= r.f_upper  # global min 0 enclosed
    assert abs(r.x[0]) < 1e-1 and abs(r.x[1]) < 1e-1
    _assert_sound_lower_bound(ackley, box, r.f_lower)


def test_enclosure_sound_even_without_convergence() -> None:
    # a tiny budget must NOT break soundness -- only widen the gap
    box = [(-5.0, 10.0), (0.0, 15.0)]
    r = certified_minimize(branin, box, tol=1e-9, max_boxes=50)
    assert not r.converged
    assert r.f_lower <= 0.39788735772973816 <= r.f_upper
    _assert_sound_lower_bound(branin, box, r.f_lower)


def test_gradient_accelerates_convergence() -> None:
    # the mean-value form + monotonicity test (exact interval gradient) cut the box count
    box = [(-2.0, 2.0), (-1.0, 1.0)]
    no_grad = certified_minimize(camel, box, tol=1e-4, max_boxes=400_000)
    with_grad = certified_minimize(camel, box, tol=1e-4, max_boxes=400_000, grad=camel_grad)
    assert with_grad.converged
    assert with_grad.boxes_explored < no_grad.boxes_explored / 10


def test_monotone_reduction_is_exact() -> None:
    # f(x, y) = (x - 3)^2 + (y + 2), monotone increasing in y on [-1, 1] -> min at y = -1
    def f(b):  # type: ignore[no-untyped-def]
        return (b[0] - P(3.0)) ** 2 + (b[1] + P(2.0))

    def g(b):  # type: ignore[no-untyped-def]
        return (P(2.0) * (b[0] - P(3.0)), P(1.0))

    box = [(0.0, 5.0), (-1.0, 1.0)]
    r = certified_minimize(f, box, tol=1e-6, grad=g)
    assert r.converged
    assert abs(r.x[0] - 3.0) < 1e-3 and abs(r.x[1] - (-1.0)) < 1e-6
    assert r.f_lower <= 1.0 <= r.f_upper  # true min = 0 + (-1 + 2) = 1


def test_certify_strict_local_min_pd_and_saddle() -> None:
    # convex bowl: Hessian = diag(2, 2) > 0 everywhere
    def bowl_hess(b):  # type: ignore[no-untyped-def]
        return [[P(2.0), P(0.0)], [P(0.0), P(2.0)]]

    assert certify_strict_local_min(bowl_hess, [(-1.0, 1.0), (-1.0, 1.0)])

    # saddle x^2 - y^2: Hessian = diag(2, -2), indefinite -> not a local min
    def saddle_hess(b):  # type: ignore[no-untyped-def]
        return [[P(2.0), P(0.0)], [P(0.0), P(-2.0)]]

    assert not certify_strict_local_min(saddle_hess, [(-1.0, 1.0), (-1.0, 1.0)])


def test_validation() -> None:
    with pytest.raises(ValueError):
        certified_minimize(quadratic, [(-1.0, 1.0)], tol=0.0)
    with pytest.raises(ValueError):
        certified_minimize(quadratic, [(-1.0, 1.0)], max_boxes=0)
