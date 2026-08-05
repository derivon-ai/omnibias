# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""W3-ext: the ``omnibias-verify`` certified_integral front-end.

Drives the certified core rules from a single ``deriv_bound(k, box)`` oracle --
the same activation-agnostic interface as ``certified_stencil_truncation`` -- and
proves the returned enclosure contains the true integral for real activation
towers.
"""

from __future__ import annotations

import importlib
import math

import pytest
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.transcend import exp_iv
from omnibias.difference import sigma_deriv_bound
from omnibias.verify import IntegralCertificate, certified_integral


def _mpmath():
    try:
        return importlib.import_module("mpmath")
    except ImportError:  # pragma: no cover
        return None


def _exp_deriv(k: int, box: Interval) -> Interval:  # every derivative of exp is exp
    return exp_iv(box)


class TestCertifiedIntegral:
    @pytest.mark.parametrize(
        "method", ["gauss", "simpson", "euler_maclaurin", "romberg", "clenshaw_curtis"]
    )
    def test_all_methods_enclose_exp(self, method: str) -> None:
        cert = certified_integral(_exp_deriv, 0.0, 1.0, method=method)
        assert isinstance(cert, IntegralCertificate)
        assert cert.enclosure.contains(math.e - 1.0)

    def test_gauss_encloses_tanh_tower(self) -> None:
        # int_0^1 tanh = ln(cosh(1)).
        db = sigma_deriv_bound("tanh")
        cert = certified_integral(db, 0.0, 1.0, method="gauss", n=5)
        exact = math.log(math.cosh(1.0))
        assert cert.enclosure.contains(exact)
        assert cert.order == 10

    @pytest.mark.skipif(_mpmath() is None, reason="mpmath not installed")
    def test_matches_mpmath_for_sigmoid(self) -> None:
        mp = _mpmath()
        db = sigma_deriv_bound("sigmoid")
        cert = certified_integral(db, -1.0, 1.0, method="gauss", n=6)
        ref = float(mp.quad(lambda t: 1.0 / (1.0 + mp.e ** (-t)), [-1, 1]))
        assert cert.enclosure.contains(ref)

    def test_reversed_limits_raise(self) -> None:
        with pytest.raises(ValueError):
            certified_integral(_exp_deriv, 1.0, 0.0)

    def test_bad_gauss_n_raises(self) -> None:
        with pytest.raises(ValueError):
            certified_integral(_exp_deriv, 0.0, 1.0, method="gauss", n=99)
