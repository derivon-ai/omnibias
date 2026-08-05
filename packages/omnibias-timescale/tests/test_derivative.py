# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Delta / nabla derivatives and the three-register dispatch of delta_derivative_tower."""

from __future__ import annotations

import math

import pytest
from omnibias.difference import finite_difference_estimate
from omnibias.qcalculus import q_derivative
from omnibias.timescale import (
    delta_derivative,
    delta_derivative_tower,
    h_integers,
    nabla_derivative,
    quantum,
    reals,
    sigma_value,
)


def test_delta_derivative_scattered_quotient() -> None:
    H = h_integers(0.5)
    f = lambda x: x**2  # noqa: E731
    # f^Delta(t) = (f(t+h) - f(t))/h = 2t + h.
    assert delta_derivative(f, 1.0, H) == pytest.approx(2.0 * 1.0 + 0.5)


def test_delta_derivative_dense_needs_fprime() -> None:
    R = reals()
    f = lambda x: x**3  # noqa: E731
    with pytest.raises(ValueError):
        delta_derivative(f, 1.0, R)
    assert delta_derivative(f, 2.0, R, fprime=lambda x: 3 * x**2) == pytest.approx(12.0)


def test_nabla_derivative_scattered_quotient() -> None:
    H = h_integers(0.25)
    f = lambda x: x**2  # noqa: E731
    # f^nabla(t) = (f(t) - f(t-h))/h = 2t - h.
    assert nabla_derivative(f, 1.0, H) == pytest.approx(2.0 * 1.0 - 0.25)


def test_tower_dispatch_reals_is_closed_form() -> None:
    for t in (-1.0, 0.3, 0.7, 2.0):
        got = delta_derivative_tower("tanh", t, reals())
        assert got == pytest.approx(1.0 - math.tanh(t) ** 2, rel=1e-10, abs=1e-12)


def test_tower_dispatch_hZ_equals_forward_difference() -> None:
    # On hZ the tower dispatch IS the omnibias-difference forward stencil.
    H = h_integers(0.3)
    for t in (0.2, 0.7, 1.5):
        got = delta_derivative_tower("tanh", t, H)
        ref = finite_difference_estimate("tanh", t, 1, 0.3, "forward").estimate
        assert got == ref  # same call, bit-identical


def test_tower_dispatch_quantum_equals_jackson() -> None:
    # On the quantum scale the tower dispatch IS the omnibias-qcalculus Jackson q-derivative.
    Q = quantum(1.5)
    for t in (0.5, 1.0, 2.0):
        got = delta_derivative_tower("tanh", t, Q)
        ref = q_derivative(lambda x: sigma_value("tanh", x), t, 1.5)
        assert got == pytest.approx(ref, rel=1e-12, abs=1e-14)
        # and it matches the raw Jackson quotient of tanh.
        raw = (math.tanh(1.5 * t) - math.tanh(t)) / ((1.5 - 1.0) * t)
        assert got == pytest.approx(raw, rel=1e-9, abs=1e-11)


@pytest.mark.parametrize("scale_factory", ["hZ", "quantum"])
def test_mu_to_zero_collapses_to_derivative(scale_factory: str) -> None:
    t = 0.7
    true = 1.0 - math.tanh(t) ** 2
    if scale_factory == "hZ":
        errs = [abs(delta_derivative_tower("tanh", t, h_integers(h)) - true) for h in (0.4, 0.1, 0.01, 0.001)]
    else:
        errs = [abs(delta_derivative_tower("tanh", t, quantum(q)) - true) for q in (1.4, 1.1, 1.01, 1.001)]
    assert errs == sorted(errs, reverse=True)  # monotone collapse as mu -> 0
    assert errs[-1] < 1e-3
