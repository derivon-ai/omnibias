# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
from __future__ import annotations

from fractions import Fraction

from omnibias.dynamics.chebyshev import (
    certify_budan_fourier,
    certify_ect,
    sturm_root_count,
    wronskian_determinant,
)
from omnibias.holonomic._core.poly_n import PolyN


def test_wronskian_of_independent_lines_is_nonzero() -> None:
    x = PolyN.var(1, 0)
    f0, f1 = x, x + PolyN.const(1, 1)
    wronskian = wronskian_determinant((f0, f1), 2)
    ect = certify_ect((f0, f1), order=2)
    assert not wronskian.is_zero()
    assert ect.proved_ect_on_interval


def test_budan_fourier_bounds_negative_roots_of_quadratic() -> None:
    x = PolyN.var(1, 0)
    poly = x * x + x - PolyN.const(1, 2)
    cert = certify_budan_fourier(poly)
    assert cert.negative_root_upper_bound >= 0
    assert sturm_root_count(poly, Fraction(-3, 2), Fraction(-1, 10)) <= cert.negative_root_upper_bound + 1
