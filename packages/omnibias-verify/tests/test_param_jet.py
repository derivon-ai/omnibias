# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Tests for the verified parameter-jet (arbitrary-order rigorous Taylor model).

The order-``N`` jet is validated against independent references: the gradient and
Hessian against central finite differences and the ``ParamSpaceLoss`` hyper-dual
Hessian (the "one TM pass = the O(P^2) sweep" win); the third-derivative tensor
against a finite-difference stencil; and the whole enclosure (polynomial + remainder)
against a dense grid **and** a random sample of the true objective. The order-3
subspace jet is checked to reproduce the `enclose_subspace_model` coefficients that
`subspace_step` now builds on.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

import pytest
from omnibias.core.verified.interval import Interval
from omnibias.verify import (
    MLPArchitecture,
    ParamSpaceLoss,
    enclose_subspace_model,
    param_jet,
)

Data = list[tuple[tuple[float, ...], tuple[float, ...]]]


# --------------------------------------------------------------------------- #
# Independent float reference.
# --------------------------------------------------------------------------- #
def _net_out(arch: MLPArchitecture, theta: Sequence[float], x: Sequence[float]) -> list[float]:
    off = 0
    layers: list[tuple[list[list[float]], list[float]]] = []
    for n_out, n_in in arch.layer_shapes:
        weight = [[theta[off + r * n_in + c] for c in range(n_in)] for r in range(n_out)]
        off += n_out * n_in
        bias = [theta[off + r] for r in range(n_out)]
        off += n_out
        layers.append((weight, bias))
    a = [float(xi) for xi in x]
    last = len(layers) - 1
    for depth, (weight, bias) in enumerate(layers):
        z = [bias[o] + math.fsum(weight[o][j] * a[j] for j in range(len(weight[o]))) for o in range(len(weight))]
        a = z if depth == last else [math.tanh(zo) for zo in z]
    return a


def _loss_float(arch: MLPArchitecture, theta: Sequence[float], data: Data, l2: float = 0.0) -> float:
    total = 0.0
    for x, y in data:
        out = _net_out(arch, theta, x)
        total += math.fsum((out[o] - yo) ** 2 for o, yo in enumerate(y))
    reg = l2 * math.fsum(t * t for t in theta) if l2 != 0.0 else 0.0
    return total / len(data) + reg


ARCH = MLPArchitecture(dims=(1, 1, 1), activation="tanh")  # P = 4  [w, b, v, c]
THETA0 = [0.7, -0.3, 1.1, 0.2]
DATA: Data = [((-1.0,), (0.4,)), ((0.0,), (0.1,)), ((0.8,), (-0.3,)), ((1.5,), (0.6,))]


# --------------------------------------------------------------------------- #
# Order-1 / order-2 readouts vs independent references.
# --------------------------------------------------------------------------- #
def test_full_p_gradient_matches_finite_difference() -> None:
    jet = param_jet(ARCH, DATA, THETA0, order=2, radius=1e-2)
    grad = jet.grad()
    assert len(grad) == ARCH.n_params
    h = 1e-6
    for i in range(ARCH.n_params):
        tp = list(THETA0)
        tp[i] += h
        tm = list(THETA0)
        tm[i] -= h
        fd = (_loss_float(ARCH, tp, DATA) - _loss_float(ARCH, tm, DATA)) / (2 * h)
        assert grad[i].mid == pytest.approx(fd, abs=1e-5)


def test_full_p_hessian_equals_param_loss_hyperdual() -> None:
    """The single order-2 TM pass reproduces the O(P^2) hyper-dual Hessian."""
    jet = param_jet(ARCH, DATA, THETA0, order=2, radius=1e-2)
    hess = jet.hessian()
    problem = ParamSpaceLoss(ARCH, DATA)
    point = [Interval.point(t) for t in THETA0]
    h_pl = problem.hessian(point)
    p = ARCH.n_params
    for i in range(p):
        for j in range(p):
            assert hess[i][j].mid == pytest.approx(h_pl[i][j].mid, abs=1e-6)
            assert hess[i][j].lo <= h_pl[i][j].hi + 1e-9
            assert hess[i][j].hi >= h_pl[i][j].lo - 1e-9


def test_hessian_is_symmetric() -> None:
    hess = param_jet(ARCH, DATA, THETA0, order=2, radius=1e-2).hessian()
    p = ARCH.n_params
    for i in range(p):
        for j in range(p):
            assert hess[i][j].lo == hess[j][i].lo and hess[i][j].hi == hess[j][i].hi


# --------------------------------------------------------------------------- #
# Order-3 tensor.
# --------------------------------------------------------------------------- #
def test_third_tensor_diagonal_matches_stencil() -> None:
    jet = param_jet(ARCH, DATA, THETA0, order=3, radius=1e-2)
    third = jet.tensor(3)
    h = 1e-2
    for i in range(ARCH.n_params):
        def g(t: float, i: int = i) -> float:
            tt = list(THETA0)
            tt[i] += t
            return _loss_float(ARCH, tt, DATA)
        fd = (g(2 * h) - 2 * g(h) + 2 * g(-h) - g(-2 * h)) / (2 * h**3)  # central 3rd-deriv stencil
        assert third[i][i][i].mid == pytest.approx(fd, abs=1e-2, rel=1e-2)


def test_third_tensor_is_symmetric() -> None:
    third = param_jet(ARCH, DATA, THETA0, order=3, radius=1e-2).tensor(3)
    p = ARCH.n_params
    for i in range(p):
        for j in range(p):
            for k in range(p):
                assert third[i][j][k].lo == third[k][j][i].lo
                assert third[i][j][k].hi == third[k][j][i].hi


def test_tensor_shortcuts_agree() -> None:
    jet = param_jet(ARCH, DATA, THETA0, order=3, radius=1e-2)
    assert jet.tensor(0).lo == jet.value().lo and jet.tensor(0).hi == jet.value().hi
    g, gt = jet.grad(), jet.tensor(1)
    assert all(g[i].lo == gt[i].lo and g[i].hi == gt[i].hi for i in range(ARCH.n_params))
    h, ht = jet.hessian(), jet.tensor(2)
    p = ARCH.n_params
    assert all(h[i][j].lo == ht[i][j].lo for i in range(p) for j in range(p))


def test_value_encloses_true_objective() -> None:
    jet = param_jet(ARCH, DATA, THETA0, order=3, radius=5e-2)
    true = _loss_float(ARCH, THETA0, DATA)
    assert jet.value().lo <= true <= jet.value().hi


# --------------------------------------------------------------------------- #
# Remainder / enclosure soundness (repo rule: dense grid + random samples).
# --------------------------------------------------------------------------- #
def _assert_full_p_sound(
    arch: MLPArchitecture, data: Data, theta0: Sequence[float], *, order: int, radius: float,
    per_axis: int, samples: int, l2: float = 0.0, seed: int = 0,
) -> None:
    jet = param_jet(arch, data, theta0, order=order, radius=radius, l2=l2)
    p = arch.n_params
    tol = 1e-12
    axis = [(-radius + 2 * radius * i / (per_axis - 1)) for i in range(per_axis)]

    def _check(delta: list[float]) -> None:
        enc = jet.tm.eval([Interval.point(d) for d in delta])
        theta = [theta0[k] + delta[k] for k in range(p)]
        true = _loss_float(arch, theta, data, l2=l2)
        assert enc.lo - tol <= true <= enc.hi + tol

    def _walk(prefix: list[float], depth: int) -> None:
        if depth == p:
            _check(prefix)
            return
        for v in axis:
            _walk([*prefix, v], depth + 1)

    _walk([], 0)
    rng = random.Random(seed)
    for _ in range(samples):
        _check([rng.uniform(-radius, radius) for _ in range(p)])


def test_full_p_enclosure_sound_order3() -> None:
    _assert_full_p_sound(ARCH, DATA, THETA0, order=3, radius=5e-2, per_axis=5, samples=4_000)


def test_full_p_enclosure_sound_order2() -> None:
    _assert_full_p_sound(ARCH, DATA, THETA0, order=2, radius=3e-2, per_axis=5, samples=3_000)


def test_full_p_enclosure_sound_with_l2() -> None:
    _assert_full_p_sound(ARCH, DATA, THETA0, order=3, radius=5e-2, per_axis=5, samples=3_000, l2=0.1)


def test_higher_order_shrinks_remainder() -> None:
    """A higher-order model has a tighter model-vs-truth remainder on the same box."""
    r = 8e-2
    w2 = param_jet(ARCH, DATA, THETA0, order=2, radius=r).remainder.width
    w3 = param_jet(ARCH, DATA, THETA0, order=3, radius=r).remainder.width
    w4 = param_jet(ARCH, DATA, THETA0, order=4, radius=r).remainder.width
    assert w3 <= w2 and w4 <= w3


# --------------------------------------------------------------------------- #
# Subspace jet + consistency with subspace_step.
# --------------------------------------------------------------------------- #
def _krylov_basis_2d() -> list[list[float]]:
    problem = ParamSpaceLoss(ARCH, DATA)
    point = [Interval.point(t) for t in THETA0]
    g = [iv.mid for iv in problem.grad(point)]
    p = ARCH.n_params
    n = math.sqrt(math.fsum(v * v for v in g))
    q0 = [v / n for v in g]
    hmat = [[cell.mid for cell in row] for row in problem.hessian(point)]
    v1 = [math.fsum(hmat[i][j] * q0[j] for j in range(p)) for i in range(p)]
    dot = math.fsum(v1[i] * q0[i] for i in range(p))
    v1 = [v1[i] - dot * q0[i] for i in range(p)]
    n1 = math.sqrt(math.fsum(v * v for v in v1))
    q1 = [v / n1 for v in v1]
    return [[q0[pp], q1[pp]] for pp in range(p)]


def test_subspace_jet_matches_enclose_subspace_model() -> None:
    basis = _krylov_basis_2d()
    jet = param_jet(ARCH, DATA, THETA0, order=3, radius=5e-2, basis=basis)
    model = enclose_subspace_model(ARCH, DATA, THETA0, basis, radius=5e-2)
    assert jet.dim == model.k == 2
    assert jet.value().lo == model.constant.lo and jet.value().hi == model.constant.hi
    grad = jet.grad()
    for i in range(2):
        assert grad[i].lo == model.grad[i].lo and grad[i].hi == model.grad[i].hi
    hess = jet.hessian()
    third = jet.tensor(3)
    for i in range(2):
        for j in range(2):
            assert hess[i][j].lo == model.hessian[i][j].lo
            for k in range(2):
                assert third[i][j][k].lo == model.third[i][j][k].lo
    assert jet.remainder.lo == model.remainder.lo and jet.remainder.hi == model.remainder.hi


def test_subspace_enclosure_sound() -> None:
    basis = _krylov_basis_2d()
    radius = 5e-2
    jet = param_jet(ARCH, DATA, THETA0, order=3, radius=radius, basis=basis)
    p = ARCH.n_params
    tol = 1e-12

    def _true(a: Sequence[float]) -> float:
        theta = [THETA0[pp] + math.fsum(basis[pp][j] * a[j] for j in range(2)) for pp in range(p)]
        return _loss_float(ARCH, theta, DATA)

    axis = [-radius + 2 * radius * i / 8 for i in range(9)]
    for a0 in axis:
        for a1 in axis:
            enc = jet.tm.eval([Interval.point(a0), Interval.point(a1)])
            assert enc.lo - tol <= _true([a0, a1]) <= enc.hi + tol
    rng = random.Random(1)
    for _ in range(3_000):
        a = [rng.uniform(-radius, radius), rng.uniform(-radius, radius)]
        enc = jet.tm.eval([Interval.point(a[0]), Interval.point(a[1])])
        assert enc.lo - tol <= _true(a) <= enc.hi + tol


# --------------------------------------------------------------------------- #
# Input validation.
# --------------------------------------------------------------------------- #
def test_order_must_be_positive() -> None:
    with pytest.raises(ValueError, match="order must be"):
        param_jet(ARCH, DATA, THETA0, order=0, radius=1e-2)


def test_radius_must_be_positive() -> None:
    with pytest.raises(ValueError, match="radius must be"):
        param_jet(ARCH, DATA, THETA0, order=2, radius=0.0)


def test_negative_l2_rejected() -> None:
    with pytest.raises(ValueError, match="l2"):
        param_jet(ARCH, DATA, THETA0, order=2, radius=1e-2, l2=-1.0)


def test_theta0_length_checked() -> None:
    with pytest.raises(ValueError, match="theta0 has"):
        param_jet(ARCH, DATA, THETA0[:-1], order=2, radius=1e-2)


def test_basis_shape_checked() -> None:
    with pytest.raises(ValueError, match="basis has"):
        param_jet(ARCH, DATA, THETA0, order=2, radius=1e-2, basis=[[1.0, 0.0]] * (ARCH.n_params - 1))


def test_tensor_order_bounds() -> None:
    jet = param_jet(ARCH, DATA, THETA0, order=2, radius=1e-2)
    with pytest.raises(ValueError, match="exceeds"):
        jet.tensor(3)
    with pytest.raises(ValueError, match="tensor order"):
        jet.tensor(-1)
