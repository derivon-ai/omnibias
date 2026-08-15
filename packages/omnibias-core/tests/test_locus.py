# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Equality-locus G1–G4 (theory 01-09)."""

from __future__ import annotations

import math
import random

import pytest
from omnibias.core.locus import (
    EqualitySystem,
    UnitTerm,
    affine_locus,
    branch_signature,
    certify_locus_point,
    hessian_blocks,
    is_transversal,
    jacobian,
    newton_project,
    residual,
    sigma_n,
)


def _case_a() -> EqualitySystem:
    return EqualitySystem(
        (
            UnitTerm(1, 1.0, (1.0, 0.0), 0.0),
            UnitTerm(1, 1.0, (0.0, 1.0), 0.0),
        )
    )


def _case_b() -> EqualitySystem:
    return EqualitySystem(
        (
            UnitTerm(1, 1.0, (1.0, 0.0), 0.0),
            UnitTerm(2, -2.0, (0.0, 1.0), 0.0),
        )
    )


def test_g1_affine_lemma_diagonals() -> None:
    sys = _case_a()
    planes = affine_locus(sys)
    assert planes is not None
    assert len(planes) == 2
    # y = x and y = -x
    for x in (-0.7, -0.2, 0.0, 0.4, 1.1):
        for pt in ((x, x), (x, -x)):
            f = residual(sys, pt)
            assert abs(f[0]) <= 1e-14
    signs_id = branch_signature(sys, (0.3, 0.3))
    signs_m = branch_signature(sys, (0.3, -0.3))
    assert signs_id[0] == 1
    assert signs_m[0] == -1


def test_case_b_worked_point() -> None:
    sys = _case_b()
    # t - t^3 = 1/4, the unique root in (0, 1).
    t = 0.26959443640544456
    y = math.atanh(t)
    f = residual(sys, (0.0, y))
    assert abs(f[0]) <= 1e-14
    df = jacobian(sys, (0.0, y))
    assert abs(df[0][0]) <= 1e-12
    assert df[0][1] != 0.0
    assert is_transversal(sys, (0.0, y))


def test_g2_jacobian_hessian_vs_fd() -> None:
    rng = random.Random(0)
    worst = 0.0

    def f_at(sys: EqualitySystem, pt: tuple[float, float]) -> float:
        return residual(sys, pt)[0]

    def fd4(sys: EqualitySystem, x: tuple[float, float], axis: int, h: float = 1e-4) -> float:
        def shift(s: float) -> float:
            pt = (x[0] + s, x[1]) if axis == 0 else (x[0], x[1] + s)
            return f_at(sys, pt)

        return (-shift(2 * h) + 8 * shift(h) - 8 * shift(-h) + shift(-2 * h)) / (12 * h)

    for _ in range(40):
        n1, n2 = rng.randint(0, 6), rng.randint(0, 6)
        sys = EqualitySystem(
            (
                UnitTerm(n1, rng.uniform(-2, 2), (rng.uniform(-1, 1), rng.uniform(-1, 1))),
                UnitTerm(n2, rng.uniform(-2, 2), (rng.uniform(-1, 1), rng.uniform(-1, 1))),
            )
        )
        x = (rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3))
        df = jacobian(sys, x)
        fd0, fd1 = fd4(sys, x, 0), fd4(sys, x, 1)
        scale = max(abs(df[0][0]), abs(df[0][1]), abs(fd0), abs(fd1), 1e-12)
        rel = max(abs(df[0][0] - fd0), abs(df[0][1] - fd1)) / scale
        worst = max(worst, rel)
        hess = hessian_blocks(sys, x)[0]
        hh = 1e-4
        h00 = (
            f_at(sys, (x[0] + hh, x[1]))
            - 2 * f_at(sys, x)
            + f_at(sys, (x[0] - hh, x[1]))
        ) / (hh * hh)
        scale_h = max(abs(hess[0][0]), abs(h00), 1e-8)
        assert abs(hess[0][0] - h00) / scale_h <= 1e-4
    assert worst <= 1e-10


def test_g3_newton_thousand() -> None:
    rng = random.Random(1)
    n_ok = 0
    n = 0
    while n < 1000:
        x_true = (rng.uniform(-0.4, 0.4), rng.uniform(-0.4, 0.4))
        n1, n2 = rng.randint(0, 3), rng.randint(0, 3)
        w1 = (rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0))
        w2 = (rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0))
        s2 = sigma_n("tanh", w2[0] * x_true[0] + w2[1] * x_true[1], n2)
        s1 = sigma_n("tanh", w1[0] * x_true[0] + w1[1] * x_true[1], n1)
        if abs(s2) < 1e-6:
            continue
        c2 = s1 / s2
        sys = EqualitySystem(
            (UnitTerm(n1, 1.0, w1), UnitTerm(n2, c2, w2)),
        )
        x0 = (x_true[0] + 0.03 * rng.uniform(-1, 1), x_true[1] + 0.03 * rng.uniform(-1, 1))
        result = newton_project(sys, x0, max_iter=5, tol=1e-12)
        n += 1
        if result.converged and result.residual_norm <= 1e-12:
            n_ok += 1
    assert n_ok >= 950


def test_g3_quadratic_rate_case_b() -> None:
    sys = _case_b()
    result = newton_project(sys, (0.0, 0.20), max_iter=5, tol=1e-14)
    assert result.converged
    assert result.iterations <= 5
    assert result.residual_norm <= 1e-14
    t = math.tanh(result.point[1])
    assert t - t**3 == pytest.approx(0.25, rel=1e-12)


def test_g4_certificate_soundness() -> None:
    sys = EqualitySystem(
        (
            UnitTerm(1, 1.0, (0.0,), 0.0),
            UnitTerm(2, -2.0, (1.0,), 0.0),
        )
    )
    y = math.atanh(0.26959443640544456)
    cert = certify_locus_point(sys, ((y - 0.02, y + 0.02),))
    assert cert is not None
    lo, hi = cert.enclosure[0]
    landed = newton_project(sys, (y - 0.01,), max_iter=10, tol=1e-14)
    assert lo <= landed.point[0] <= hi
    empty = certify_locus_point(sys, ((2.0, 2.2),))
    assert empty is None
    scan = [residual(sys, (x,))[0] for x in [2.0 + i * 0.01 for i in range(21)]]
    assert all(abs(v) > 1e-3 for v in scan)


def test_mismatched_is_not_affine() -> None:
    assert affine_locus(_case_b()) is None
