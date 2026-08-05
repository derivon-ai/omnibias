# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""The public certified constants ``PI_IV`` / ``E_IV`` enclose pi / e soundly."""

from __future__ import annotations

import math

import pytest
from omnibias.core.verified import E_IV, PI_IV
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.transcend import _PI_IV


def test_constants_are_nondegenerate_intervals() -> None:
    assert isinstance(PI_IV, Interval)
    assert isinstance(E_IV, Interval)
    assert PI_IV.lo < PI_IV.hi
    assert E_IV.lo < E_IV.hi


def test_pi_iv_encloses_math_pi_tightly() -> None:
    assert PI_IV.lo <= math.pi <= PI_IV.hi
    assert PI_IV.width < 1e-14


def test_e_iv_encloses_math_e_tightly() -> None:
    assert E_IV.lo <= math.e <= E_IV.hi
    assert E_IV.width < 1e-14


def test_private_alias_is_public_constant() -> None:
    # Trig extremum tests still reach pi via the retained private alias.
    assert _PI_IV is PI_IV


def test_constants_enclose_high_precision_truth() -> None:
    mp = pytest.importorskip("mpmath")
    with mp.workdps(60):
        assert PI_IV.lo <= mp.pi <= PI_IV.hi
        assert E_IV.lo <= mp.e <= E_IV.hi
