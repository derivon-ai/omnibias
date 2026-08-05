# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Verified ODE integrator: enclosures must contain the closed-form flow."""

from __future__ import annotations

import math

import pytest
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.ode import TaylorSeries, integrate_ivp


def _exp(s: list[TaylorSeries]) -> list[TaylorSeries]:
    return [s[0]]


def _riccati(s: list[TaylorSeries]) -> list[TaylorSeries]:
    return [s[0] * s[0]]


def _harmonic(s: list[TaylorSeries]) -> list[TaylorSeries]:
    return [s[1], -s[0]]


def test_exponential_flow_encloses_e() -> None:
    (y,) = integrate_ivp(_exp, [1.0], 0.0, 1.0, order=16, n_steps=8)
    assert y.contains(math.e)
    assert y.width < 1e-12


@pytest.mark.parametrize("t", [0.5, 1.0, 2.0, 3.0])
def test_exponential_flow_matches_exp(t: float) -> None:
    (y,) = integrate_ivp(_exp, [1.0], 0.0, t, order=16, n_steps=max(8, int(4 * t)))
    assert y.contains(math.exp(t))
    assert y.width < 1e-9 * math.exp(t)


def test_riccati_blowup_encloses_pole() -> None:
    # y' = y^2, y(0)=1  =>  y(t) = 1/(1-t), pole at t=1.
    for t in (0.5, 0.8, 0.9):
        (y,) = integrate_ivp(_riccati, [1.0], 0.0, t, order=18, n_steps=40)
        assert y.contains(1.0 / (1.0 - t))
    # very close to the pole the enclosure widens but still must contain the truth
    (y,) = integrate_ivp(_riccati, [1.0], 0.0, 0.95, order=20, n_steps=120)
    assert y.contains(1.0 / (1.0 - 0.95))


def test_harmonic_oscillator_encloses_cos_sin() -> None:
    # y1'=y2, y2'=-y1, y0=(1,0)  =>  y1 = cos t, y2 = -sin t.
    y1, y2 = integrate_ivp(_harmonic, [1.0, 0.0], 0.0, 1.0, order=16, n_steps=8)
    assert y1.contains(math.cos(1.0))
    assert y2.contains(-math.sin(1.0))
    assert y1.width < 1e-12
    assert y2.width < 1e-12


def test_harmonic_energy_is_conserved_rigorously() -> None:
    # cos^2 + sin^2 = 1 must be enclosed for the whole orbit.
    y1, y2 = integrate_ivp(_harmonic, [1.0, 0.0], 0.0, 2.0, order=16, n_steps=16)
    energy = y1 * y1 + y2 * y2
    assert energy.contains(1.0)


def test_interval_initial_condition_encloses_both_endpoints() -> None:
    # A whole bundle of initial data: every solution must be enclosed.
    (y,) = integrate_ivp(_exp, [Interval(0.9, 1.1)], 0.0, 1.0, order=16, n_steps=8)
    assert y.contains(0.9 * math.e)
    assert y.contains(1.1 * math.e)


def test_taylor_series_algebra() -> None:
    a = TaylorSeries([Interval.point(1.0), Interval.point(2.0), Interval.point(3.0)])
    b = TaylorSeries([Interval.point(1.0), Interval.point(1.0), Interval.point(0.0)])
    prod = a * b  # (1 + 2t + 3t^2)(1 + t) = 1 + 3t + 5t^2 (truncated)
    assert prod.coeffs[0].contains(1.0)
    assert prod.coeffs[1].contains(3.0)
    assert prod.coeffs[2].contains(5.0)
    scaled = a * 2.0
    assert scaled.coeffs[1].contains(4.0)


def test_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError):
        integrate_ivp(_exp, [1.0], 1.0, 0.0)
    with pytest.raises(ValueError):
        integrate_ivp(_exp, [1.0], 0.0, 1.0, n_steps=0)
