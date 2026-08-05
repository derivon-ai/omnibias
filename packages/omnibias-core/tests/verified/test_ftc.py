# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Rigorous Fundamental Theorem of Calculus certificate tests.

The verified identity ``int_a^b sigma^(k) = sigma^(k-1)(b) - sigma^(k-1)(a)`` is
checked three independent ways: the residual encloses ``0`` and is tight, the
integral side encloses an ``mpmath`` quadrature oracle, and the sealed
certificate is tamper-evident.
"""

from __future__ import annotations

import copy

import pytest
from omnibias.core.proof.certificate import verify_certificate_digest
from omnibias.core.verified.ftc import (
    _ftc_parts,
    certified_ftc_residual,
    ftc_certificate,
)

mpmath = pytest.importorskip("mpmath")


def _sigma(name: str):
    table = {
        "tanh": mpmath.tanh,
        "sigmoid": lambda z: 1 / (1 + mpmath.e ** (-z)),
        "gaussian": lambda z: mpmath.e ** (-(z**2) / 2),
        "silu": lambda z: z / (1 + mpmath.e ** (-z)),
        "softplus": lambda z: mpmath.log(1 + mpmath.e**z),
        "gelu": lambda z: z * (1 + mpmath.erf(z / mpmath.sqrt(2))) / 2,
        "sin": mpmath.sin,
        "cos": mpmath.cos,
    }
    return table[name]


# Modest cells so the Taylor-model remainder stays tight for every activation.
_GRID = [
    ("tanh", -0.25, 0.25, 1),
    ("sigmoid", -0.2, 0.3, 1),
    ("gaussian", -0.3, 0.2, 1),
    ("silu", -0.25, 0.25, 1),
    ("softplus", -0.3, 0.25, 1),
    ("gelu", -0.2, 0.3, 1),
    ("sin", 0.1, 0.6, 1),
    ("cos", -0.3, 0.3, 1),
    ("tanh", -0.2, 0.25, 2),
    ("sin", -0.2, 0.3, 3),
]


@pytest.mark.parametrize(("name", "a", "b", "k"), _GRID)
def test_residual_contains_zero_and_is_tight(name: str, a: float, b: float, k: int) -> None:
    residual = certified_ftc_residual(name, a, b, k=k, order=8)
    assert residual.contains(0.0)  # a sound tower forces the true residual to 0
    assert residual.width < 1e-3  # rigorous + tight over a modest cell


@pytest.mark.parametrize(
    "name", ["tanh", "sigmoid", "gaussian", "silu", "softplus", "gelu", "sin", "cos"]
)
def test_integral_side_encloses_mpmath_quadrature(name: str) -> None:
    a, b, k, order = -0.2, 0.3, 1, 8
    lhs, _rhs, residual = _ftc_parts(name, a, b, k=k, order=order)
    assert residual.contains(0.0)
    center, radius = 0.5 * (a + b), 0.5 * (b - a)
    sig = _sigma(name)
    with mpmath.workdps(50):
        val = mpmath.quad(
            lambda z: mpmath.diff(sig, z, k), [center - radius, center + radius]
        )
    assert lhs.lo <= float(val) <= lhs.hi


def test_higher_order_gives_tighter_residual() -> None:
    a, b = -0.4, 0.4
    w_lo = certified_ftc_residual("tanh", a, b, k=1, order=10).width
    w_hi = certified_ftc_residual("tanh", a, b, k=1, order=4).width
    assert w_lo < w_hi  # the Taylor-model remainder shrinks with order


def test_certificate_verifies_and_flips_on_tampered_bound() -> None:
    cert = ftc_certificate("tanh", -0.3, 0.4, k=1, order=6)
    assert verify_certificate_digest(cert)
    assert cert["claim"].startswith("FTC:")
    assert cert["honesty"]["unproven_claim"] is False
    assert cert["honesty"]["rigorous_enclosure"] is True
    assert cert["payload"]["residual_contains_zero"] is True

    tampered = copy.deepcopy(cert)
    hi = float.fromhex(tampered["payload"]["residual"]["hi"])
    tampered["payload"]["residual"]["hi"] = (hi + 0.5).hex()  # widen a sealed bound
    assert not verify_certificate_digest(tampered)


def test_invalid_arguments_raise() -> None:
    with pytest.raises(ValueError, match="k must be >= 1"):
        certified_ftc_residual("tanh", -0.3, 0.4, k=0)
    with pytest.raises(ValueError, match="require a < b"):
        certified_ftc_residual("tanh", 0.4, -0.3, k=1)
    with pytest.raises(ValueError, match="order must be >= 0"):
        certified_ftc_residual("tanh", -0.3, 0.4, k=1, order=-1)
    with pytest.raises(ValueError, match="unsupported activation"):
        certified_ftc_residual("relu", -0.3, 0.4, k=1)
