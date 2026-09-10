# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Approximate functional equation on a named compact vs mpmath / EM."""

from __future__ import annotations

import pytest
from omnibias.core.verified.dirichlet import zeta_euler_maclaurin
from omnibias.core.verified.riemann_siegel import (
    T_MAX,
    T_MIN,
    afe_honesty,
    hardy_z,
    zeta_approximate_functional_equation,
)


def _encloses(enc, value: complex) -> bool:
    return enc.re.contains(value.real) and enc.im.contains(value.imag)


def test_afe_honesty() -> None:
    h = afe_honesty()
    assert h["rh_claim"] is False
    assert h["pade"] is False
    assert h["gabcke_remainder"] is False


def test_afe_refuses_outside_compact() -> None:
    with pytest.raises(ValueError, match="T_MIN"):
        zeta_approximate_functional_equation(0.5 + 1.0j)
    with pytest.raises(ValueError, match="T_MAX"):
        zeta_approximate_functional_equation(0.5 + (T_MAX + 5.0) * 1j)
    with pytest.raises(ValueError, match="Re"):
        zeta_approximate_functional_equation(1.2 + 10.0j)


def test_afe_encloses_mpmath_on_named_pack() -> None:
    mp = pytest.importorskip("mpmath")
    points = (0.5 + 8.0j, 0.7 + 10.0j, 0.3 + 12.0j)
    for s in points:
        enc = zeta_approximate_functional_equation(s)
        with mp.workdps(40):
            true = complex(mp.zeta(mp.mpc(s.real, s.imag)))
        assert _encloses(enc, true), (s, enc, true)


def test_afe_and_em_overlap_contain_sample() -> None:
    mp = pytest.importorskip("mpmath")
    s = 0.5 + (T_MIN + 1.0) * 1j
    afe = zeta_approximate_functional_equation(s)
    em = zeta_euler_maclaurin(s, num_sum_terms=25, order=6)
    with mp.workdps(40):
        true = complex(mp.zeta(mp.mpc(s.real, s.imag)))
    assert _encloses(afe, true)
    assert _encloses(em, true)


def test_hardy_z_encloses_mpmath() -> None:
    mp = pytest.importorskip("mpmath")
    t = 14.0
    enc = hardy_z(t)
    with mp.workdps(40):
        s = mp.mpc(0.5, t)
        theta = mp.im(mp.loggamma(mp.mpc(0.25, 0.5 * t))) - 0.5 * t * mp.log(mp.pi)
        true = complex(mp.exp(1j * theta) * mp.zeta(s))
    assert _encloses(enc, true)
