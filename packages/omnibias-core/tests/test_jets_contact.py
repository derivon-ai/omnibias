# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Contact / holonomy tests for the jet vocabulary (theory 01-10).

Vocabulary, not a discovery. No ``omnibias-jetbundle`` package.
"""

from __future__ import annotations

import math
import random

from omnibias.core.jets import contact_residual, is_holonomic, max_abs_contact_residual
from omnibias.core.polynomials import hermite_coeffs, tanh_polynomial_coeffs


def _horner(coeffs: tuple[float, ...], x: float) -> float:
    acc = 0.0
    for c in reversed(coeffs):
        acc = acc * x + c
    return acc


def _tanh_jet(x: float, order: int = 3) -> tuple[float, ...]:
    t = math.tanh(x)
    out = [t]
    for n in range(1, order + 1):
        out.append(_horner(tanh_polynomial_coeffs(n), t))
    return tuple(out)


def _gauss_jet(x: float, order: int = 3) -> tuple[float, ...]:
    g = math.exp(-0.5 * x * x)
    out = [g]
    for n in range(1, order + 1):
        sign = 1.0 if n % 2 == 0 else -1.0
        out.append(sign * _horner(hermite_coeffs(n), x) * g)
    return tuple(out)


def _poly_jet(x: float, coeffs: tuple[float, ...], order: int = 3) -> tuple[float, ...]:
    out: list[float] = []
    for n in range(order + 1):
        acc = 0.0
        for k in range(n, len(coeffs)):
            falling = 1.0
            for j in range(n):
                falling *= k - j
            power = 1.0 if k == n else x ** (k - n)
            acc += coeffs[k] * falling * power
        out.append(acc)
    return tuple(out)


def _exp_jet(x: float, order: int = 3) -> tuple[float, ...]:
    e = math.exp(0.3 * x)
    return tuple((0.3**n) * e for n in range(order + 1))


def test_g1_holonomic_vs_corrupted_no_misclassification() -> None:
    rng = random.Random(0)
    n_ok = 0
    n_bad = 0
    for i in range(220):
        kind = i % 4
        x = rng.uniform(-1.2, 1.2)
        h = 1e-4
        if kind == 0:
            def sampler(z: float) -> tuple[float, ...]:
                return _tanh_jet(z)
        elif kind == 1:
            def sampler(z: float) -> tuple[float, ...]:
                return _gauss_jet(z)
        elif kind == 2:
            coeffs = tuple(rng.uniform(-1.0, 1.0) for _ in range(5))

            def sampler(z: float, c: tuple[float, ...] = coeffs) -> tuple[float, ...]:
                return _poly_jet(z, c)
        else:
            def sampler(z: float) -> tuple[float, ...]:
                return _exp_jet(z)
        assert is_holonomic(sampler, x, h=h)
        n_ok += 1

        def corrupted(z: float, _s=sampler) -> tuple[float, ...]:
            jet = list(_s(z))
            jet[1] = 0.90
            return tuple(jet)

        assert not is_holonomic(corrupted, x, h=h)
        n_bad += 1
    assert n_ok >= 200 and n_bad >= 200


def test_g2_rate_three_halvings() -> None:
    x = 0.5
    h0 = 8e-4
    j0 = _tanh_jet(x)

    def holonomic_res(h: float) -> float:
        return max_abs_contact_residual(j0, _tanh_jet(x + h), h=h)

    def corrupted_res(h: float) -> float:
        good = _tanh_jet(x)
        bad_h = list(_tanh_jet(x + h))
        bad0 = list(good)
        bad0[1] = 0.90
        bad_h[1] = 0.90
        return max_abs_contact_residual(bad0, bad_h, h=h)

    hs = [h0 / (2**k) for k in range(4)]
    r_h = [holonomic_res(h) for h in hs]
    r_c = [corrupted_res(h) for h in hs]
    # Three halvings: holonomic ~4x, corrupted ~2x.
    for i in range(3):
        ratio_h = r_h[i + 1] / r_h[i]
        ratio_c = r_c[i + 1] / r_c[i]
        assert 0.15 <= ratio_h <= 0.35, (i, ratio_h, r_h)
        assert 0.40 <= ratio_c <= 0.60, (i, ratio_c, r_c)


def test_contact_residual_worked_example() -> None:
    # Spec 01-10 worked example at x=0.5, h=1e-4, tanh.
    x = 0.5
    h = 1e-4
    res = contact_residual(_tanh_jet(x, 2), _tanh_jet(x + h, 2), h=h)
    assert abs(res[0]) < 1e-8
    bad0 = list(_tanh_jet(x, 2))
    badh = list(_tanh_jet(x + h, 2))
    bad0[1] = 0.90
    badh[1] = 0.90
    res_bad = contact_residual(bad0, badh, h=h)
    assert abs(res_bad[0]) > 1e-6
