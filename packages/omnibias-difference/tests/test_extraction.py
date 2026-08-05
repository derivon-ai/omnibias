# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Certified finite-difference -> derivative extraction (the founding collapse).

Soundness (grid AND random), the certified FD-error sandwich, convergence at the
scheme's order, and closed-form-beats-naive-FD where the ``1/delta^m``
cancellation blows the numerical estimate up.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable

import pytest
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.transcend import exp_iv
from omnibias.difference import (
    certified_derivative_enclosure,
    certified_fd_error,
    certified_fd_error_general,
    finite_difference_estimate,
    sigma_deriv_bound,
)

mpmath = pytest.importorskip("mpmath")

SMOOTH = ["tanh", "sigmoid", "gaussian", "silu", "gelu", "softplus", "sech"]


def _mp_f(name: str) -> Callable[[object], object]:
    return {
        "tanh": lambda z: mpmath.tanh(z),
        "sigmoid": lambda z: mpmath.mpf(1) / (1 + mpmath.e ** (-z)),
        "gaussian": lambda z: mpmath.e ** (-(z**2) / 2),
        "silu": lambda z: z / (1 + mpmath.e ** (-z)),
        "gelu": lambda z: z * (1 + mpmath.erf(z / mpmath.sqrt(2))) / 2,
        "softplus": lambda z: mpmath.log(1 + mpmath.e**z),
        "sech": lambda z: mpmath.sech(z),
    }[name]


def _true_derivative(name: str, z0: float, order: int) -> float:
    with mpmath.workdps(50):
        taylor = mpmath.taylor(_mp_f(name), z0, order)
        return float(taylor[order] * math.factorial(order))


# ----- grid-and-random soundness of the closed-form enclosure over a box ----- #
@pytest.mark.parametrize("name", SMOOTH)
def test_enclosure_contains_dense_grid_and_random_sample(name: str) -> None:
    order = 5
    lo, hi = 0.2, 0.9
    rng = random.Random(hash(name) & 0xFFFF)
    enc = certified_derivative_enclosure(name, Interval(lo, hi), order)
    grid = [lo + (hi - lo) * i / 24 for i in range(25)]
    randoms = [rng.uniform(lo, hi) for _ in range(40)]
    for z0 in grid + randoms:
        for k in range(order + 1):
            v = _true_derivative(name, z0, k)
            assert enc.tower[k].lo - 1e-12 <= v <= enc.tower[k].hi + 1e-12
    assert enc.label == "closed-form"


@pytest.mark.parametrize("name", SMOOTH)
def test_pointwise_enclosure_is_tight_and_true(name: str) -> None:
    order = 4
    z0 = 0.4
    enc = certified_derivative_enclosure(name, z0, order)
    true = _true_derivative(name, z0, order)
    assert enc.value.lo <= true <= enc.value.hi
    assert enc.value.width <= 1e-7 * (1 + abs(true))


# ----- the certified FD-error sandwich ----- #
@pytest.mark.parametrize("stencil", ["forward", "central"])
@pytest.mark.parametrize("seed", range(8))
def test_fd_estimate_inside_certified_error(stencil: str, seed: int) -> None:
    rng = random.Random(1000 + seed)
    name = rng.choice(SMOOTH)
    order = rng.randint(1, 4)
    z = rng.uniform(-1.0, 1.0)
    delta = rng.choice([1e-1, 5e-2, 1e-2])
    cert = certified_fd_error(name, z, order, delta, stencil)
    true = _true_derivative(name, z, order)
    # the closed-form enclosure contains the true derivative
    assert cert.enclosure.lo <= true <= cert.enclosure.hi
    # the certified bound really bounds the numerical estimate's error
    assert abs(cert.estimate - true) <= cert.error_bound + 1e-12
    # the self-checking sandwich is consistent, and brackets the truth
    assert cert.certified
    assert cert.true_derivative_interval.lo <= true <= cert.true_derivative_interval.hi


@pytest.mark.parametrize("stencil,p", [("forward", 1), ("central", 2)])
def test_fd_converges_at_scheme_order(stencil: str, p: int) -> None:
    name, z, order = "tanh", 0.6, 3
    true = _true_derivative(name, z, order)
    errs = []
    for delta in (1e-1, 1e-2, 1e-3):
        est = finite_difference_estimate(name, z, order, delta, stencil).estimate
        errs.append(abs(est - true))
    # each 10x shrink of delta cuts the error by ~10^p (allow slack for constants)
    for a, b in zip(errs, errs[1:], strict=False):
        assert b < a
        assert b <= a / (10 ** (p - 1)) * 1.5


@pytest.mark.parametrize("stencil,p", [("forward", 1), ("central", 2)])
def test_certified_bound_shrinks_at_scheme_order(stencil: str, p: int) -> None:
    name, z, order = "sigmoid", -0.3, 2
    b1 = certified_fd_error(name, z, order, 1e-1, stencil).error_bound
    b2 = certified_fd_error(name, z, order, 1e-2, stencil).error_bound
    assert b2 < b1
    # bound is O(delta^p): a 10x shrink drops it ~10^p.
    assert b2 <= b1 / (10 ** (p - 1)) * 2.0


# ----- closed-form beats the naive 1/delta^m finite difference ----- #
def test_closed_form_beats_naive_fd_under_cancellation() -> None:
    """At a tiny delta the naive FD loses to catastrophic cancellation while the
    closed-form tower stays exact -- the whole point of the delta->0 register."""
    name, z, order = "tanh", 0.5, 4
    true = _true_derivative(name, z, order)
    enc = certified_derivative_enclosure(name, z, order)
    # closed form: a razor-tight, guaranteed enclosure of the truth
    assert enc.value.lo <= true <= enc.value.hi
    assert enc.value.width <= 1e-8
    # naive FD at a tiny step: catastrophic cancellation (signs ~ 1/delta^4 = 1e16)
    naive = finite_difference_estimate(name, z, order, 1e-4, "central").estimate
    naive_err = abs(naive - true)
    assert naive_err > 1e3 * enc.value.width  # the closed form wins decisively


def test_negative_order_raises() -> None:
    with pytest.raises(ValueError):
        certified_derivative_enclosure("tanh", 0.0, -1)
    with pytest.raises(ValueError):
        certified_fd_error("tanh", 0.0, -1, 1e-2)


# ----- W5: the remainder engine is decoupled from the activation dictionary ----- #
@pytest.mark.parametrize("name", ["tanh", "sigmoid", "gaussian", "sech"])
@pytest.mark.parametrize("order", [1, 2, 3])
def test_general_engine_matches_dictionary_wrapper(name: str, order: int) -> None:
    """certified_fd_error is exactly certified_fd_error_general bound to the tower."""
    import math as _math

    fdict = {
        "tanh": _math.tanh,
        "sigmoid": lambda z: 1.0 / (1.0 + _math.exp(-z)),
        "gaussian": lambda z: _math.exp(-0.5 * z * z),
        "sech": lambda z: 1.0 / _math.cosh(z),
    }
    z, delta = 0.42, 1e-2
    wrapped = certified_fd_error(name, z, order, delta, "central")
    general = certified_fd_error_general(
        fdict[name], sigma_deriv_bound(name), z, order, delta, "central", name=name
    )
    assert general.estimate == wrapped.estimate
    assert general.error_bound == wrapped.error_bound
    assert general.enclosure == wrapped.enclosure


def test_general_engine_certifies_function_outside_dictionary() -> None:
    """exp is NOT one of the nine activations; the engine still certifies it."""
    exp_bound = lambda k, box: exp_iv(box)  # noqa: E731 -- exp^(k) = exp
    # delta=1e-2 keeps truncation above the 1/delta^order float-cancellation floor
    # (at 1e-3 the order-3 stencil's roundoff would exceed the truncation bound).
    for order in (1, 2, 3):
        cert = certified_fd_error_general(
            math.exp, exp_bound, 0.6, order, 1e-2, "central", name="exp"
        )
        true = math.exp(0.6)  # every derivative of exp equals exp
        assert cert.name == "exp"
        assert cert.certified
        assert cert.true_derivative_interval.contains(true)
        assert cert.enclosure.contains(true)


def test_general_engine_exact_polynomial_stencil() -> None:
    """A user tower with a vanishing high derivative gives a ~zero truncation bound."""

    def cubic_bound(k: int, box: Interval) -> Interval:
        if k == 0:
            return box.pow_int(3)
        if k == 1:
            return Interval.from_rational(3) * box.pow_int(2)
        if k == 2:
            return Interval.from_rational(6) * box
        if k == 3:
            return Interval.from_rational(6)
        return Interval.point(0.0)

    cert = certified_fd_error_general(lambda t: t**3, cubic_bound, 2.0, 2, 0.1, "central")
    assert cert.error_bound < 1e-15  # f'''' = 0 -> central 2nd-deriv is exact
