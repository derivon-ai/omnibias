# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""W6 -- coefficient-engine -> validated-dynamics bridge.

Soundness: the closed-form tanh-field flow (built by composing the ``sigma``
derivative tower onto the state's time-Taylor series) must *enclose* a fine RK4
reference across K>=8 initial conditions. Best-in-class: the QR-Lohner frame must
beat the naive interval-Taylor integrator (``integrate_ivp``) on wrapping for the
2-D sigma-oscillator. Plus discrete period-orbit proofs of the logistic map.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import pytest
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.lohner import lohner_flow
from omnibias.core.verified.ode import integrate_ivp
from omnibias.dynamics import (
    DiscretePeriodicOrbit,
    discrete_periodic_point,
    sigma_oscillator_field,
    vector_field_from_sigma_tower,
)


def _rk4(f: Callable[[float], float], x0: float, t1: float, steps: int = 8000) -> float:
    """A fine RK4 reference for a scalar autonomous ODE ``x' = f(x)``."""
    x, dt = x0, t1 / steps
    for _ in range(steps):
        k1 = f(x)
        k2 = f(x + 0.5 * dt * k1)
        k3 = f(x + 0.5 * dt * k2)
        k4 = f(x + dt * k3)
        x += dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return x


# K >= 8 initial conditions spanning the saturating and near-linear regimes.
_INITIALS = [-1.5, -0.8, -0.3, 0.0, 0.25, 0.5, 0.9, 1.4]


@pytest.mark.parametrize("x0", _INITIALS)
def test_tanh_flow_encloses_fine_rk4(x0: float) -> None:
    # x' = tanh(x): the field is built by rigorous tower composition, so a single
    # closed-form sigma evaluation drives every Taylor order of the Lohner step.
    field, jac = vector_field_from_sigma_tower("tanh")
    h, n = 0.025, 20
    box = lohner_flow(field, jac, [Interval.point(x0)], h, n, order=6).to_box()
    ref = _rk4(math.tanh, x0, h * n)
    assert box[0].contains(ref)
    # The validated enclosure is genuinely tight, not a vacuous bound.
    assert box[0].width < 1e-9


def test_scalar_field_with_scale_and_bias_encloses_rk4() -> None:
    # x' = tanh(2x - 0.3): exercise the affine pre-composition path.
    field, jac = vector_field_from_sigma_tower("tanh", scale=2.0, bias=-0.3)
    h, n = 0.02, 20
    box = lohner_flow(field, jac, [Interval.point(0.4)], h, n, order=6).to_box()
    ref = _rk4(lambda x: math.tanh(2.0 * x - 0.3), 0.4, h * n)
    assert box[0].contains(ref)


def test_sigmoid_flow_encloses_fine_rk4() -> None:
    # The bridge is activation-generic: swap tanh -> sigmoid, same machinery.
    field, jac = vector_field_from_sigma_tower("sigmoid")
    h, n = 0.03, 20
    box = lohner_flow(field, jac, [Interval.point(0.2)], h, n, order=6).to_box()

    def f(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    assert box[0].contains(_rk4(f, 0.2, h * n))


def test_oscillator_lohner_beats_naive_interval_flow() -> None:
    # 2-D sigma-oscillator x'=y, y'=-tanh(x): starting from an initial *box*, the
    # naive interval-Taylor integrator wraps (the box rotates and its axis-aligned
    # hull inflates); the QR-Lohner frame tracks the rotation and stays tight.
    field, jac = sigma_oscillator_field("tanh", stiffness=1.0, damping=0.0)
    rad = 0.01
    y0 = [Interval(0.5 - rad, 0.5 + rad), Interval(-rad, rad)]
    h, n = 0.03, 60
    loh = lohner_flow(field, jac, y0, h, n, order=8).to_box()
    ivp = integrate_ivp(field, y0, 0.0, h * n, order=8, n_steps=n)
    # Lohner must be strictly tighter on both components (wrapping control).
    assert loh[0].width < ivp[0].width
    assert loh[1].width < ivp[1].width
    # Headline: a clear, not marginal, win.
    assert ivp[0].width / loh[0].width > 2.0


# --- discrete period-orbit proofs (Newton-series / chain-rule derivative) ------

_R = 3.2  # logistic parameter with a stable period-2 orbit


def _logistic() -> tuple[
    Callable[[Interval], Interval], Callable[[Interval], Interval]
]:
    r = Interval.point(_R)
    one, two = Interval.point(1.0), Interval.point(2.0)
    g = lambda x: r * x * (one - x)  # noqa: E731
    dg = lambda x: r * (one - two * x)  # noqa: E731
    return g, dg


def test_logistic_fixed_point_is_proved() -> None:
    g, dg = _logistic()
    orbit = discrete_periodic_point(g, dg, 1.0 - 1.0 / _R, 1)
    assert orbit.exists and orbit.period == 1
    assert orbit.enclosure is not None
    lo, hi = orbit.enclosure
    assert lo <= 1.0 - 1.0 / _R <= hi


def test_logistic_period_two_orbit_is_proved() -> None:
    g, dg = _logistic()
    # Analytic period-2 point: (r+1 + sqrt((r-3)(r+1))) / (2r).
    x_star = (_R + 1 + math.sqrt((_R - 3) * (_R + 1))) / (2 * _R)
    orbit = discrete_periodic_point(g, dg, x_star, 2)
    assert isinstance(orbit, DiscretePeriodicOrbit)
    assert orbit.exists and orbit.period == 2
    assert orbit.enclosure is not None
    lo, hi = orbit.enclosure
    assert lo <= x_star <= hi
    assert orbit.kappa is not None and orbit.kappa < 1.0  # contraction


def test_non_periodic_guess_is_not_certified() -> None:
    # x_bar = 0.1 is far from every fixed point of g^2 ({0, 1-1/r, 0.513, 0.799});
    # no period-2 point lives in any ball up to the max radius -> no false proof.
    g, dg = _logistic()
    orbit = discrete_periodic_point(g, dg, 0.1, 2)
    assert not orbit.exists
    assert orbit.enclosure is None


def test_period_must_be_positive() -> None:
    g, dg = _logistic()
    with pytest.raises(ValueError, match="period must be >= 1"):
        discrete_periodic_point(g, dg, 0.5, 0)
