# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""OMBU wavelet frames G1–G3 (theory 01-06). G4 is smoke-earned."""

from __future__ import annotations

import math

import pytest
from omnibias.core.frames import (
    FrameSpec,
    admissibility_constant,
    compile_bank,
    dilated_sigma_n,
    littlewood_paley_bounds,
    vanishing_moments,
)
from omnibias.core.spectral_design import hat_sigma_magnitude


def test_g1_admissibility_none_for_n1() -> None:
    assert admissibility_constant("gaussian", 1) is None
    assert admissibility_constant("sech", 1) is None
    assert vanishing_moments("gaussian", 1) == 0


def test_g1_gaussian_closed_form_vs_integral() -> None:
    for n in range(2, 7):
        c = admissibility_constant("gaussian", n)
        assert c is not None
        # Numerical Calderón integral int_0^inf xi^{2n-1} |hat g|^2 d xi
        acc = 0.0
        dx = 0.002
        x = dx
        while x < 12.0:
            hat = hat_sigma_magnitude("gaussian", x)
            acc += (x ** (2 * n - 1)) * hat * hat * dx
            x += dx
        rel = abs(c.mid - acc) / c.mid
        assert rel <= 1e-10


def test_g1_sech_matches_integral() -> None:
    c = admissibility_constant("sech", 2)
    assert c is not None
    acc = 0.0
    dx = 0.002
    x = dx
    while x < 20.0:
        hat = hat_sigma_magnitude("sech", x)
        acc += (x ** (2 * 2 - 1)) * hat * hat * dx
        x += dx
    assert c.lo <= acc <= c.hi


def test_g2_lp_bounds_contain_grid_and_sample() -> None:
    spec = FrameSpec("gaussian", 2, scales=(0.5, 1.0, 2.0), offset_spacing=0.25)
    a, b = littlewood_paley_bounds(spec, grid=1024)
    rng = __import__("random").Random(0)
    for xi in [2.0 ** ((i - 200) / 16.0) for i in range(400)] + [
        2.0 ** rng.uniform(-8, 8) for _ in range(40)
    ]:
        acc = 0.0
        for s in spec.scales:
            mag = (abs(s * xi) ** spec.order) * math.sqrt(s) * hat_sigma_magnitude(
                spec.base, s * xi
            )
            acc += mag * mag
        assert a.lo - 1e-12 <= acc <= b.hi + 1e-12


def test_g3_dilation_exactness() -> None:
    import numpy as np

    eps = np.finfo(np.float64).eps
    for base in ("gaussian", "tanh", "sech"):
        for n in range(0, 8):
            for alpha in (0.7, 1.5):
                u = -0.4
                left = dilated_sigma_n(base, u, n, alpha)
                right = (alpha**n) * dilated_sigma_n(base, alpha * u, n, 1.0)
                scale = max(abs(left), abs(right), 1.0)
                ulp = abs(left - right) / (eps * scale)
                assert ulp <= 4.0


def test_compile_bank() -> None:
    spec = FrameSpec("sech", 3, scales=(1.0, 2.0), offset_spacing=0.5, n_offsets=8)
    bank = compile_bank(spec)
    assert bank.n_offsets == 8
    assert bank.scales == (1.0, 2.0)
    assert bank.spacing == pytest.approx(0.5)
