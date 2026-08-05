# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Second-order lower bound + interval-Newton (Krawczyk) contractor in the B&B.

Two properties matter and are tested separately:

* **soundness is never broken** -- adding the Hessian-based accelerators only
  intersects more sound enclosures, so ``[f_lower, f_upper]`` must still bracket
  the true global minimum on a dense grid + random sample.  The critical case is a
  function whose *global* minimum sits on the domain **boundary** while an interior
  *local* minimum competes: the Krawczyk contractor must never prune the boundary
  minimizer (that is what :func:`_interior_full_dim` guards);
* **it accelerates** -- on a function with first-order dependency overestimation
  (six-hump camel) the second-order form + Newton contraction cut the box count.
"""

from __future__ import annotations

import math
import random

from omnibias.core.verified.interval import Interval
from omnibias.verify import certified_minimize

P = Interval.point


# --------------------------- objectives (hand-coded exact grad + hess) --------
def camel(b):  # type: ignore[no-untyped-def]
    x, y = b
    return (P(4.0) - P(2.1) * x**2 + (x**4) * P(1.0 / 3.0)) * x**2 + x * y + (P(-4.0) + P(4.0) * y**2) * y**2


def camel_grad(b):  # type: ignore[no-untyped-def]
    x, y = b
    return (P(8.0) * x - P(8.4) * x**3 + P(2.0) * x**5 + y, x - P(8.0) * y + P(16.0) * y**3)


def camel_hess(b):  # type: ignore[no-untyped-def]
    x, y = b
    return [[P(8.0) - P(25.2) * x**2 + P(10.0) * x**4, P(1.0)], [P(1.0), P(-8.0) + P(48.0) * y**2]]


# f = x^3 - x + y^2 : interior local min at (1/sqrt3, 0) ~ -0.385, but the GLOBAL
# min is on the domain boundary at (-2, 0) = -6.
def cubic(b):  # type: ignore[no-untyped-def]
    x, y = b
    return x**3 - x + y**2


def cubic_grad(b):  # type: ignore[no-untyped-def]
    x, y = b
    return (P(3.0) * x**2 - P(1.0), P(2.0) * y)


def cubic_hess(b):  # type: ignore[no-untyped-def]
    x, y = b
    return [[P(6.0) * x, P(0.0)], [P(0.0), P(2.0)]]


CAMEL_MIN = -1.0316284535


def _assert_sound(f, box, f_lower, *, n_grid=41, n_rand=1500, seed=0):  # type: ignore[no-untyped-def]
    rng = random.Random(seed)
    los = [lo for lo, _ in box]
    his = [hi for _, hi in box]
    worst = math.inf
    for i in range(n_grid):
        for j in range(n_grid):
            x = los[0] + (his[0] - los[0]) * i / (n_grid - 1)
            y = los[1] + (his[1] - los[1]) * j / (n_grid - 1)
            worst = min(worst, Interval.from_value(f((P(x), P(y)))).lo)
    for _ in range(n_rand):
        pt = tuple(P(lo + (hi - lo) * rng.random()) for lo, hi in zip(los, his, strict=True))
        worst = min(worst, Interval.from_value(f(pt)).lo)
    assert worst >= f_lower - 1e-9, f"lower bound not sound: sample {worst} < f_lower {f_lower}"


# --------------------------- soundness ---------------------------------------
def test_second_order_newton_sound_camel() -> None:
    box = [(-2.0, 2.0), (-1.0, 1.0)]
    r = certified_minimize(camel, box, tol=1e-4, max_boxes=400_000, grad=camel_grad, hess=camel_hess)
    assert r.converged
    assert r.f_lower <= CAMEL_MIN <= r.f_upper
    _assert_sound(camel, box, r.f_lower)


def test_boundary_minimum_not_lost_by_newton() -> None:
    # THE guard test: Krawczyk must not prune the boundary global min (-2, 0) = -6
    # in favour of the interior local min (~ -0.385).
    box = [(-2.0, 2.0), (-1.0, 1.0)]
    r = certified_minimize(cubic, box, tol=1e-4, max_boxes=300_000, grad=cubic_grad, hess=cubic_hess)
    assert r.f_lower <= -6.0 <= r.f_upper, f"UNSOUND: f_lower={r.f_lower} excludes boundary min -6"
    assert abs(r.x[0] + 2.0) < 1e-2 and abs(r.x[1]) < 1e-2  # found the boundary minimizer, not the interior one
    _assert_sound(cubic, box, r.f_lower)


def test_second_order_only_without_newton_is_sound() -> None:
    box = [(-2.0, 2.0), (-1.0, 1.0)]
    r = certified_minimize(camel, box, tol=1e-4, max_boxes=400_000, grad=camel_grad, hess=camel_hess, use_newton=False)
    assert r.f_lower <= CAMEL_MIN <= r.f_upper
    _assert_sound(camel, box, r.f_lower)


# --------------------------- acceleration ------------------------------------
def test_hessian_accelerators_cut_box_count() -> None:
    box = [(-2.0, 2.0), (-1.0, 1.0)]
    first_order = certified_minimize(camel, box, tol=1e-4, max_boxes=400_000, grad=camel_grad)
    second_order = certified_minimize(camel, box, tol=1e-4, max_boxes=400_000, grad=camel_grad, hess=camel_hess)
    assert second_order.converged
    assert second_order.boxes_explored < first_order.boxes_explored
    # both are sound and agree on the enclosure
    assert first_order.f_lower <= CAMEL_MIN <= first_order.f_upper
    assert second_order.f_lower <= CAMEL_MIN <= second_order.f_upper


def test_enclosure_agrees_with_first_order() -> None:
    box = [(-2.0, 2.0), (-1.0, 1.0)]
    a = certified_minimize(camel, box, tol=1e-3, max_boxes=400_000, grad=camel_grad)
    b = certified_minimize(camel, box, tol=1e-3, max_boxes=400_000, grad=camel_grad, hess=camel_hess)
    assert abs(a.f_upper - b.f_upper) < 1e-2
    assert a.f_lower <= b.f_upper and b.f_lower <= a.f_upper  # overlapping certified enclosures
