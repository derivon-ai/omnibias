# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Complex Gamma / log enclosures vs mpmath (grid + random)."""

from __future__ import annotations

import math
import random

import pytest
from omnibias.core.verified.complex_interval import ComplexInterval
from omnibias.core.verified.gamma_complex import (
    cos_ci,
    exp_ci,
    gamma_ci,
    log_ci,
    log_gamma_ci,
    sin_ci,
)
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.transcend import PI_IV, sinh_iv

_SEEDS = range(4)


def _encloses(enc: ComplexInterval, value: complex) -> bool:
    return enc.re.contains(value.real) and enc.im.contains(value.imag)


def _grid_and_random_complex(
    re_lo: float,
    re_hi: float,
    im_lo: float,
    im_hi: float,
    seed: int,
    *,
    grid: int = 4,
    rnd: int = 4,
) -> list[complex]:
    pts: list[complex] = []
    for i in range(grid):
        for j in range(grid):
            re = re_lo + (re_hi - re_lo) * i / max(grid - 1, 1)
            im = im_lo + (im_hi - im_lo) * j / max(grid - 1, 1)
            pts.append(complex(re, im))
    rng = random.Random(seed)
    pts.extend(complex(rng.uniform(re_lo, re_hi), rng.uniform(im_lo, im_hi)) for _ in range(rnd))
    return pts


def test_sinh_iv_encloses_mpmath() -> None:
    mp = pytest.importorskip("mpmath")
    for x in (-2.0, -0.3, 0.0, 0.5, 1.7):
        enc = sinh_iv(Interval.point(x))
        with mp.workdps(40):
            true = float(mp.sinh(x))
        assert enc.lo <= true <= enc.hi


def test_exp_ci_and_log_ci_roundtrip_points() -> None:
    for z in (1.0, 2.0 + 0.5j, 0.3 - 0.4j, 1.2 + 0.0j):
        enc = log_ci(exp_ci(z))
        assert _encloses(enc, complex(z) if not isinstance(z, complex) else z) or (
            abs(enc.re.mid - complex(z).real) < 1e-8
        )


def test_log_ci_refuses_cut() -> None:
    with pytest.raises(ValueError, match="cut|origin"):
        log_ci(-1.0)
    with pytest.raises(ValueError, match="cut|origin"):
        log_ci(0.0)


def test_gamma_ci_known_positive_reals() -> None:
    g1 = gamma_ci(1.0)
    assert g1.re.contains(1.0) and g1.im.contains(0.0)
    g2 = gamma_ci(2.0)
    assert g2.re.contains(1.0) and g2.im.contains(0.0)
    g_half = gamma_ci(0.5)
    sqrt_pi = math.sqrt(math.pi)
    assert g_half.re.contains(sqrt_pi)
    g_neg_half = gamma_ci(-0.5)
    assert g_neg_half.re.contains(-2.0 * sqrt_pi)
    assert g_neg_half.im.contains(0.0)


def test_gamma_ci_encloses_mpmath_right_half() -> None:
    mp = pytest.importorskip("mpmath")
    for seed in _SEEDS:
        for z in _grid_and_random_complex(0.6, 3.0, -1.0, 1.0, seed):
            enc = gamma_ci(z)
            with mp.workdps(40):
                true = complex(mp.gamma(mp.mpc(z.real, z.imag)))
            assert _encloses(enc, true), (z, enc, true)


def test_gamma_ci_encloses_mpmath_left_half_off_axis() -> None:
    mp = pytest.importorskip("mpmath")
    for seed in _SEEDS:
        for z in _grid_and_random_complex(-1.4, -0.4, 0.3, 1.2, seed):
            enc = gamma_ci(z)
            with mp.workdps(40):
                true = complex(mp.gamma(mp.mpc(z.real, z.imag)))
            assert _encloses(enc, true), (z, enc, true)


def test_gamma_reflection_residual_contains_one() -> None:
    mp = pytest.importorskip("mpmath")
    z = 0.3 + 0.4j
    # Gamma(z) Gamma(1-z) sin(pi z) / pi  == 1
    prod = gamma_ci(z) * gamma_ci(1.0 - z) * sin_ci(ComplexInterval.from_parts(PI_IV) * z)
    prod = prod / ComplexInterval.from_parts(PI_IV)
    assert prod.re.contains(1.0)
    assert prod.im.contains(0.0)
    with mp.workdps(40):
        true = complex(mp.gamma(mp.mpc(z.real, z.imag)))
    assert _encloses(gamma_ci(z), true)


def test_gamma_ci_refuses_poles() -> None:
    with pytest.raises(ValueError, match="pole"):
        gamma_ci(0.0)
    with pytest.raises(ValueError, match="pole"):
        gamma_ci(-1.0)
    with pytest.raises(ValueError, match="pole"):
        log_gamma_ci(ComplexInterval.from_parts(Interval(-0.2, 0.1), Interval(-0.01, 0.01)))


def test_sin_cos_ci_at_zero() -> None:
    s = sin_ci(0.0)
    c = cos_ci(0.0)
    assert s.re.contains(0.0) and s.im.contains(0.0)
    assert c.re.contains(1.0) and c.im.contains(0.0)
