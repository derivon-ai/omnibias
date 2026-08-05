# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Closed-form-jet warm-start for the certified minimiser.

Two contracts:

* **soundness is unconditional** -- ``seeds=`` can only lower the incumbent
  ``f_upper``; the certified enclosure ``f_lower <= min f <= f_upper`` is identical
  to the un-seeded run (never widened), and out-of-box seeds are clamped in;
* **acceleration** -- a good incumbent prunes the branch-and-bound sooner, so
  ``boxes_explored`` never exceeds the no-seed baseline; the differentiable torch /
  jax descent seeds are inside the box and strictly reduce the readout.
"""

from __future__ import annotations

import math

import pytest
from omnibias.core.verified.interval import Interval
from omnibias.verify import certified_minimize, certified_network_minimize

P = Interval.point

Layer = tuple[list[list[float]], list[float] | None, str | None]

# a tanh MLP with a loose value enclosure -> genuine branch-and-bound work.
TANH_MLP: list[Layer] = [
    ([[1.5, -0.7], [0.8, 1.2], [-1.1, 0.5], [0.3, -1.4]], [0.1, -0.2, 0.05, 0.2], "tanh"),
    ([[0.9, -1.3, 0.6, 0.7]], [0.15], None),
]
TANH_BOX = [(-2.0, 2.0), (-2.0, 2.0)]

# a gaussian well: min -2 at the origin; box centre (1, 1) sits off the minimum.
GAUSS_WELL: list[Layer] = [
    ([[1.0, 0.0], [0.0, 1.0]], [0.0, 0.0], "gaussian"),
    ([[-1.0, -1.0]], [0.0], None),
]
GAUSS_BOX = [(-1.0, 3.0), (-1.0, 3.0)]


def eval_net(layers: list[Layer], x: list[float]) -> list[float]:
    v = list(x)
    for weight, bias, name in layers:
        pre = []
        for i, row in enumerate(weight):
            s = sum(row[j] * v[j] for j in range(len(v)))
            if bias is not None:
                s += bias[i]
            pre.append(s)
        if name == "tanh":
            v = [math.tanh(z) for z in pre]
        elif name == "gaussian":
            v = [math.exp(-z * z / 2.0) for z in pre]
        elif name is None:
            v = pre
        else:  # pragma: no cover - defensive
            raise ValueError(name)
    return v


def grid_min(layers: list[Layer], box: list[tuple[float, float]], n: int = 81) -> float:
    (xlo, xhi), (ylo, yhi) = box
    best = math.inf
    for i in range(n):
        for j in range(n):
            x = xlo + (xhi - xlo) * i / (n - 1)
            y = ylo + (yhi - ylo) * j / (n - 1)
            best = min(best, eval_net(layers, [x, y])[0])
    return best


def _double_well() -> tuple[object, object]:
    """``f(x) = (x^2 - 1)^2``: two global minima at ``x = +/-1`` (value 0)."""

    def f(b: list[Interval]) -> Interval:
        (x,) = b
        return (x * x - P(1.0)) ** 2

    def g(b: list[Interval]) -> list[Interval]:
        (x,) = b
        return [P(4.0) * x * (x * x - P(1.0))]

    return f, g


# --------------------------- backend-agnostic seeds -------------------------


def test_seeds_preserve_enclosure_and_cut_boxes() -> None:
    f, g = _double_well()
    box = [(-2.0, 2.0)]
    base = certified_minimize(f, box, grad=g, tol=1e-6)  # type: ignore[arg-type]
    seeded = certified_minimize(f, box, grad=g, tol=1e-6, seeds=[(1.0,), (-1.0,)])  # type: ignore[arg-type]
    # both enclose the true minimum 0 soundly ...
    assert base.f_lower <= 0.0 <= base.f_upper
    assert seeded.f_lower <= 0.0 <= seeded.f_upper
    # ... the seed never widens the enclosure, only tightens the incumbent ...
    assert seeded.f_upper <= base.f_upper + 1e-12
    # ... and the strong incumbent cuts branch-and-bound work.
    assert seeded.boxes_explored <= base.boxes_explored


def test_out_of_box_seed_is_clamped() -> None:
    f, g = _double_well()
    box = [(-0.5, 0.5)]  # min on this box is at x = +/-0.5, value (0.25 - 1)^2
    r = certified_minimize(f, box, grad=g, tol=1e-6, seeds=[(5.0,)])  # type: ignore[arg-type]
    true_min = (0.25 - 1.0) ** 2
    assert r.f_lower <= true_min <= r.f_upper  # clamping keeps the enclosure sound


def test_seed_arity_mismatch_raises() -> None:
    f, _ = _double_well()
    with pytest.raises(ValueError, match="coords"):
        certified_minimize(f, [(-1.0, 1.0)], seeds=[(0.1, 0.2)])  # type: ignore[arg-type]


def test_network_minimize_seeds_are_sound_and_accelerate() -> None:
    base = certified_network_minimize(TANH_MLP, TANH_BOX, tol=1e-3, max_boxes=400_000)
    # the analytic minimiser is unknown, but a dense grid gives a strong incumbent
    gx = grid_min(TANH_MLP, TANH_BOX, n=41)
    # feed the grid argmin as a seed (any feasible point is a valid warm start)
    best_pt = min(
        ((x, y) for x in _lin(-2, 2, 41) for y in _lin(-2, 2, 41)),
        key=lambda p: eval_net(TANH_MLP, [p[0], p[1]])[0],
    )
    seeded = certified_network_minimize(
        TANH_MLP, TANH_BOX, tol=1e-3, max_boxes=400_000, seeds=[best_pt]
    )
    assert seeded.converged
    assert seeded.f_lower <= gx  # certified lower bound stays sound
    assert seeded.boxes_explored <= base.boxes_explored


def _lin(a: float, b: float, n: int) -> list[float]:
    return [a + (b - a) * i / (n - 1) for i in range(n)]


# --------------------------- torch backend ----------------------------------


def test_torch_descent_seeds_inside_box_and_reduce_readout() -> None:
    pytest.importorskip("torch")
    from omnibias.verify.torch import descent_seeds

    seeds = descent_seeds(GAUSS_WELL, GAUSS_BOX, starts=6, steps=200, lr=0.3, seed=0)
    for s in seeds:
        assert GAUSS_BOX[0][0] - 1e-6 <= s[0] <= GAUSS_BOX[0][1] + 1e-6
        assert GAUSS_BOX[1][0] - 1e-6 <= s[1] <= GAUSS_BOX[1][1] + 1e-6
    f_center = eval_net(GAUSS_WELL, [1.0, 1.0])[0]
    f_best = min(eval_net(GAUSS_WELL, list(s))[0] for s in seeds)
    assert f_best < f_center - 1e-3  # descent strictly improved on the box centre


def test_torch_warm_started_network_minimize_sound_and_faster() -> None:
    pytest.importorskip("torch")
    from omnibias.verify.torch import warm_started_network_minimize

    base = certified_network_minimize(TANH_MLP, TANH_BOX, tol=1e-3, max_boxes=400_000)
    warm = warm_started_network_minimize(
        TANH_MLP, TANH_BOX, tol=1e-3, max_boxes=400_000, starts=8, steps=150, lr=0.2
    )
    assert warm.converged
    gx = grid_min(TANH_MLP, TANH_BOX, n=41)
    assert warm.f_lower <= gx  # sound lower bound
    assert warm.f_lower <= warm.f_upper
    assert warm.boxes_explored <= base.boxes_explored


# --------------------------- jax backend ------------------------------------


def test_jax_descent_seeds_inside_box_and_reduce_readout() -> None:
    pytest.importorskip("jax")
    from omnibias.verify.jax import descent_seeds

    seeds = descent_seeds(GAUSS_WELL, GAUSS_BOX, starts=6, steps=200, lr=0.3, seed=0)
    for s in seeds:
        assert GAUSS_BOX[0][0] - 1e-5 <= s[0] <= GAUSS_BOX[0][1] + 1e-5
        assert GAUSS_BOX[1][0] - 1e-5 <= s[1] <= GAUSS_BOX[1][1] + 1e-5
    f_center = eval_net(GAUSS_WELL, [1.0, 1.0])[0]
    f_best = min(eval_net(GAUSS_WELL, list(s))[0] for s in seeds)
    assert f_best < f_center - 1e-3


def test_torch_and_jax_seeds_both_descend() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("jax")
    from omnibias.verify.jax import descent_seeds as jax_seeds
    from omnibias.verify.torch import descent_seeds as torch_seeds

    f_center = eval_net(GAUSS_WELL, [1.0, 1.0])[0]
    ts = torch_seeds(GAUSS_WELL, GAUSS_BOX, starts=6, steps=200, lr=0.3, seed=0)
    js = jax_seeds(GAUSS_WELL, GAUSS_BOX, starts=6, steps=200, lr=0.3, seed=0)
    t_best = min(eval_net(GAUSS_WELL, list(s))[0] for s in ts)
    j_best = min(eval_net(GAUSS_WELL, list(s))[0] for s in js)
    assert t_best < f_center - 1e-3
    assert j_best < f_center - 1e-3
    # loose parity: both land near the interior minimum (exact bit-parity not required)
    assert abs(t_best - j_best) < 1e-2
