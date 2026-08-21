# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Theory 03-07: scale flow and coarse-graining, gates G1–G3 and G6."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
from omnibias.core.composed_curvature import eval_tanh_derivative
from omnibias.core.scale import (
    FlowSystem,
    ScaledPack,
    coarse_grain_linear,
    eval_gaussian_derivative,
    flow_coefficients,
    honesty_payload,
    overlap,
    report_exponents,
    rescale_pack,
)


def _rel_err(got: float, ref: float) -> float:
    return abs(got - ref) / max(1.0, abs(ref))


def test_g1_exact_rescaling() -> None:
    factors = (1.0 / 1024.0, 1.0 / 8.0, 8.0, 1024.0)
    for n in range(11):
        pack = ScaledPack(order=n, mean=0.0, alpha=1.0, base="tanh")
        for f in factors:
            back = rescale_pack(rescale_pack(pack, f), 1.0 / f)
            assert back.alpha == pack.alpha
            u = 0.2
            got = rescale_pack(pack, f).value(u)
            ref = (f**n) * pack.value(f * u)
            assert abs(got - ref) <= 4.0 * math.ulp(ref if ref != 0.0 else 1.0)


def test_g1_worked_tanh_example() -> None:
    pack = ScaledPack(order=2, mean=0.0, alpha=1.0, base="tanh")
    got = rescale_pack(pack, 2.0).value(0.2)
    ref = 4.0 * eval_tanh_derivative(0.4, 2)
    assert abs(got - ref) <= 4.0 * math.ulp(ref)


def test_g2_overlap_matches_quadrature() -> None:
    cases = (
        ScaledPack(order=0, mean=0.0, alpha=1.0),
        ScaledPack(order=0, mean=0.0, alpha=8.0),
        ScaledPack(order=1, mean=0.0, alpha=1.0),
        ScaledPack(order=1, mean=0.2, alpha=2.0),
        ScaledPack(order=2, mean=-0.1, alpha=1.5),
    )
    for p in cases:
        for q in cases:
            for k in (0, 1, 2):
                closed = overlap(p, q, derivative_order=k)
                xs = np.linspace(-12.0, 12.0, 40001)
                dx = float(xs[1] - xs[0])
                qk = np.array(
                    [
                        q.weight
                        * (q.alpha ** (q.order + k))
                        * eval_gaussian_derivative(q.alpha * (float(x) - q.mean), q.order + k)
                        for x in xs
                    ],
                    dtype=np.float64,
                )
                pv = np.array([p.value(float(x)) for x in xs], dtype=np.float64)
                num = float(np.dot(pv, qk) * dx)
                assert _rel_err(closed, num) <= 1e-12, (p, q, k, closed, num)


def test_g3_linear_exactness() -> None:
    packs = (
        ScaledPack(order=0, mean=-0.4, alpha=1.0),
        ScaledPack(order=0, mean=0.4, alpha=1.0),
        ScaledPack(order=0, mean=0.0, alpha=8.0),
    )
    op = coarse_grain_linear(packs, cutoff=1.0, derivative_order=2)
    assert len(op.slow) == 2
    c = (0.3, -0.7)
    xs = np.linspace(-10.0, 10.0, 30001)
    dx = float(xs[1] - xs[0])
    lu = np.zeros_like(xs)
    for cj, q in zip(c, op.slow, strict=True):
        lu = lu + cj * np.array(
            [
                q.weight
                * (q.alpha ** (q.order + 2))
                * eval_gaussian_derivative(q.alpha * (float(x) - q.mean), q.order + 2)
                for x in xs
            ],
            dtype=np.float64,
        )
    projected = []
    for p in op.slow:
        pv = np.array([p.value(float(x)) for x in xs], dtype=np.float64)
        projected.append(float(np.dot(pv, lu) * dx))
    got = op.apply(c)
    for a, b in zip(got, projected, strict=True):
        assert _rel_err(a, b) <= 1e-12


def test_g6_exponent_refuses_without_order() -> None:
    packs = (
        ScaledPack(order=0, mean=0.0, alpha=1.0),
        ScaledPack(order=0, mean=0.0, alpha=4.0),
    )
    system = flow_coefficients(packs, order=2)
    fps = system.fixed_points()
    exp = report_exponents(system, fps[0])
    assert system.truncation_order == 2
    assert len(exp) == 1
    with pytest.raises(ValueError, match="truncation order"):
        report_exponents(SimpleNamespace(coefficients=system.coefficients), fps[0])
    with pytest.raises(ValueError, match="truncation"):
        system.exponents(fps[0], truncation_order=0)
    study = [FlowSystem(system.coefficients, truncation_order=o).exponents(fps[0])[0] for o in (1, 2, 3)]
    assert len(study) == 3


def test_honesty_third_axis() -> None:
    payload = honesty_payload()
    assert payload["third_axis"] is True
    assert payload["founding_bias_collapse"] is False
    assert payload["temperature_collapse"] is False
