# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Irregular / Birkhoff stencils (theory 01-04 G1–G4)."""

from __future__ import annotations

import math
import random
from fractions import Fraction

import numpy as np
import pytest
from omnibias.core.verified.interval import Interval
from omnibias.difference import (
    apply_irregular_stencil,
    certified_irregular_error,
    is_poised_exact,
    offsets_exact,
    physical_weights,
    polya_screen,
    signs_exact,
    solve_irregular_stencil,
)
from omnibias.difference._core.irregular import StencilRequest


def _uniform_value_request(order: int, stencil: str) -> StencilRequest:
    h = Fraction(1)
    offs = offsets_exact(order, h, stencil)
    nodes = tuple(o / h for o in offs)
    orders = tuple((0,) for _ in nodes)
    return StencilRequest(nodes, orders, target_order=order)


def test_worked_example_exact_rationals() -> None:
    """Spec §5: f'(0) from f(-h), f(0), f'(h)."""
    req = StencilRequest(
        (Fraction(-1), Fraction(0), Fraction(1)),
        ((0,), (0,), (1,)),
        target_order=1,
    )
    st = solve_irregular_stencil(req)
    assert st is not None
    h = Fraction(1, 10)
    phys = physical_weights(st, h)
    assert phys == ((Fraction(-2, 3) / h,), (Fraction(2, 3) / h,), (Fraction(1, 3),))
    assert st.leading_coeff == Fraction(5, 18)
    assert st.accuracy == 2
    assert polya_screen(req) is True
    assert is_poised_exact(req) is True


def test_g1_reproduces_signs_exact_forward_and_central() -> None:
    h = Fraction(1, 7)
    for order in (0, 1, 2, 3):
        for kind in ("forward", "central"):
            req = _uniform_value_request(order, kind)
            st = solve_irregular_stencil(req)
            assert st is not None, (order, kind)
            phys = physical_weights(st, h)
            flat = tuple(w[0] for w in phys)
            expected = signs_exact(order, h)
            assert flat == expected, (order, kind, flat, expected)


def test_g2_empirical_rate_battery() -> None:
    """Measured log-log rate within 0.15 of reported accuracy over three halvings."""
    req = StencilRequest(
        (Fraction(-1), Fraction(0), Fraction(1)),
        ((0,), (0,), (1,)),
        target_order=1,
    )
    st = solve_irregular_stencil(req)
    assert st is not None
    hs = [0.1, 0.05, 0.025, 0.0125]

    def _rate(fn_samples, truth: float) -> float:
        errs = []
        for h in hs:
            samples = fn_samples(h)
            est = apply_irregular_stencil(st, Fraction(h).limit_denominator(), samples)
            errs.append(abs(est - truth))
        log_h = np.log(np.asarray(hs, dtype=float))
        log_e = np.log(np.asarray(errs, dtype=float))
        slope, _ = np.polyfit(log_h, log_e, 1)
        return float(slope)

    rate_exp = _rate(
        lambda h: ((math.exp(-h),), (1.0,), (math.exp(h),)),
        1.0,
    )
    rate_sin = _rate(
        lambda h: ((math.sin(-h),), (0.0,), (math.cos(h),)),
        1.0,
    )

    def _inv_samples(h: float) -> tuple[tuple[float, ...], ...]:
        # f = 1/(1+x), f' = -1/(1+x)^2, f'(0) = -1
        return ((1.0 / (1.0 - h),), (1.0,), (-1.0 / (1.0 + h) ** 2,))

    rate_inv = _rate(_inv_samples, -1.0)
    for rate in (rate_exp, rate_sin, rate_inv):
        assert abs(rate - st.accuracy) <= 0.15, (rate, st.accuracy)


def test_g3_certificate_covers_grid_and_random() -> None:
    req = StencilRequest(
        (Fraction(-1), Fraction(0), Fraction(1)),
        ((0,), (0,), (1,)),
        target_order=1,
    )
    st = solve_irregular_stencil(req)
    assert st is not None
    rng = random.Random(0)
    hs = [0.2 / (k + 1) for k in range(12)] + [rng.uniform(0.01, 0.2) for _ in range(12)]
    for h in hs:
        hh = Fraction(h).limit_denominator(10_000)
        h_f = float(hh)
        samples = ((math.exp(-h_f),), (1.0,), (math.exp(h_f),))
        est = apply_irregular_stencil(st, hh, samples)
        err = abs(est - 1.0)
        # |f'''| = exp on [-h, h], so M_3 <= exp(h)
        bound = Interval.point(math.exp(h_f))
        cert = certified_irregular_error(st, h=hh, deriv_bound=bound, estimate=est)
        assert err <= cert.error_bound + 1e-15, (h_f, err, cert.error_bound)


def test_g4_poisedness_curated_set() -> None:
    hermite = StencilRequest(
        (Fraction(-1), Fraction(1)),
        ((0, 1), (0, 1)),
        target_order=1,
    )
    worked = StencilRequest(
        (Fraction(-1), Fraction(0), Fraction(1)),
        ((0,), (0,), (1,)),
        target_order=1,
    )
    gap = StencilRequest((Fraction(0),), ((0, 2),), target_order=1)
    # Pólya passes (two values + a derivative) but the even-even residual
    # makes the Vandermonde singular: f(-1), f'(0), f(1).
    polya_not_rank = StencilRequest(
        (Fraction(-1), Fraction(0), Fraction(1)),
        ((0,), (1,), (0,)),
        target_order=0,
    )
    poised = (hermite, worked)
    unpoised = (gap, polya_not_rank)
    for req in poised:
        assert polya_screen(req) is True
        assert is_poised_exact(req) is True
        assert solve_irregular_stencil(req) is not None
    assert polya_screen(gap) is False
    assert is_poised_exact(gap) is False
    assert polya_screen(polya_not_rank) is True  # necessary, not sufficient
    assert is_poised_exact(polya_not_rank) is False
    for req in unpoised:
        assert solve_irregular_stencil(req) is None
    for req in poised:
        assert polya_screen(req) is not False


def test_request_rejects_duplicate_nodes() -> None:
    with pytest.raises(ValueError, match="distinct"):
        StencilRequest((Fraction(0), Fraction(0)), ((0,), (1,)), 0)
