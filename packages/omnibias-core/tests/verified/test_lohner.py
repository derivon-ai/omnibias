# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""QR-Lohner validated flow: wrapping control, soundness, matrix exponential."""

from __future__ import annotations

import math

from omnibias.core.verified.interval import Interval
from omnibias.core.verified.lohner import (
    LohnerSet,
    TaylorSeries,
    constant_jacobian,
    interval_matrix_exp,
    linear_field,
    lohner_flow,
    naive_interval_flow,
)


def test_interval_matrix_exp_diagonal() -> None:
    # exp(diag(a, b)) = diag(e^a, e^b).
    m = [[Interval.point(0.5), Interval.point(0.0)], [Interval.point(0.0), Interval.point(-1.0)]]
    e = interval_matrix_exp(m, order=25)
    assert e[0][0].lo <= math.exp(0.5) <= e[0][0].hi
    assert e[1][1].lo <= math.exp(-1.0) <= e[1][1].hi
    assert e[0][1].lo <= 0.0 <= e[0][1].hi


def test_interval_matrix_exp_rotation_generator() -> None:
    # exp([[0,-t],[t,0]]) = [[cos t, -sin t],[sin t, cos t]].
    t = 0.7
    m = [[Interval.point(0.0), Interval.point(-t)], [Interval.point(t), Interval.point(0.0)]]
    e = interval_matrix_exp(m, order=30)
    assert e[0][0].lo <= math.cos(t) <= e[0][0].hi
    assert e[1][0].lo <= math.sin(t) <= e[1][0].hi


def test_rotation_wrapping_is_controlled() -> None:
    # y' = A y, A a pure rotation generator: the true |y| is conserved, so the
    # enclosure of a small initial box should stay small for many revolutions.
    a = [[0.0, -1.0], [1.0, 0.0]]
    field = linear_field(a)
    jac = constant_jacobian(a)
    y0 = [Interval(0.999, 1.001), Interval(-0.001, 0.001)]

    h = 2.0 * math.pi / 200.0
    steps = 200 * 6  # six full revolutions
    final = lohner_flow(field, jac, y0, h, steps, order=14)
    naive = naive_interval_flow(a, y0, h, steps, order=14)

    # True solution of the centre after `steps` is rotation by total angle ~ 6*2pi.
    angle = h * steps
    true_x = math.cos(angle) * 1.0 - math.sin(angle) * 0.0
    true_y = math.sin(angle) * 1.0 + math.cos(angle) * 0.0
    box = final.to_box()
    assert box[0].lo <= true_x <= box[0].hi
    assert box[1].lo <= true_y <= box[1].hi

    # Wrapping control: Lohner stays tight; naive interval stepping blows up.
    assert final.width() < 0.05
    assert naive[0].width > 10.0 * final.width()


def test_rotation_long_time_tightness() -> None:
    # Even over 20 revolutions the Lohner box width stays bounded (no blow-up).
    a = [[0.0, -1.0], [1.0, 0.0]]
    field = linear_field(a)
    jac = constant_jacobian(a)
    y0 = [Interval(1.0, 1.0), Interval(0.0, 0.0)]  # exact point start
    h = 2.0 * math.pi / 100.0
    final = lohner_flow(field, jac, y0, h, 100 * 20, order=16)
    assert final.width() < 1e-3


def test_nonlinear_cubic_oscillator_soundness() -> None:
    # y' = z, z' = -y^3 (a polynomial, genuinely nonlinear field).
    def field(series: list[TaylorSeries]) -> list[TaylorSeries]:
        y, z = series
        return [z, -(y * y * y)]

    def jac(box: list[Interval]) -> list[list[Interval]]:
        y = box[0]
        return [
            [Interval.point(0.0), Interval.point(1.0)],
            [Interval.point(-3.0) * y * y, Interval.point(0.0)],
        ]

    y0 = [Interval(0.5, 0.5), Interval(0.0, 0.0)]
    h = 0.01
    steps = 50
    final = lohner_flow(field, jac, y0, h, steps, order=12)

    # Non-rigorous RK4 reference, integrated with *fine* substeps so its own
    # truncation error is far below the (very tight) rigorous box width.
    def f(state: tuple[float, float]) -> tuple[float, float]:
        y, z = state
        return (z, -(y**3))

    state = (0.5, 0.0)
    dt = h / 400.0
    for _ in range(steps * 400):
        k1 = f(state)
        k2 = f((state[0] + 0.5 * dt * k1[0], state[1] + 0.5 * dt * k1[1]))
        k3 = f((state[0] + 0.5 * dt * k2[0], state[1] + 0.5 * dt * k2[1]))
        k4 = f((state[0] + dt * k3[0], state[1] + dt * k3[1]))
        state = (
            state[0] + dt / 6.0 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]),
            state[1] + dt / 6.0 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]),
        )

    box = final.to_box()
    assert box[0].lo <= state[0] <= box[0].hi
    assert box[1].lo <= state[1] <= box[1].hi


def test_lohner_set_roundtrip() -> None:
    box = [Interval(0.0, 2.0), Interval(-1.0, 1.0)]
    s = LohnerSet.from_box(box)
    out = s.to_box()
    for orig, got in zip(box, out, strict=True):
        assert got.lo <= orig.lo and orig.hi <= got.hi
