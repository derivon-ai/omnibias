# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Certified ``L^p`` / Sobolev norms of an ingested network over a box.

Soundness (verified-enclosures rule): each certified enclosure must contain the
true norm, checked against a dense deterministic grid quadrature (with a random
sample for the plain integrals). ReLU nets exercise the non-smooth path.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Sequence

import pytest
from omnibias.core.verified.interval import Interval
from omnibias.verify import (
    LinearLayer,
    LpNormCertificate,
    Network,
    ReLULayer,
    SobolevNormCertificate,
    TanhLayer,
    affine_layer,
    certified_layer_cake_integral,
    certified_lp_norm,
    certified_sobolev_norm,
)


# --------------------------------------------------------------------------- #
# Numerical references.
# --------------------------------------------------------------------------- #
def _forward(net: Network, x: Sequence[float]) -> list[float]:
    vec = list(x)
    for layer in net:
        if isinstance(layer, LinearLayer):
            vec = [
                sum(w * xi for w, xi in zip(row, vec, strict=True)) + b
                for row, b in zip(layer.weight, layer.bias, strict=True)
            ]
        elif isinstance(layer, ReLULayer):
            vec = [max(0.0, v) for v in vec]
        else:
            vec = [math.tanh(v) for v in vec]
    return vec


def _grid_points(box: Sequence[Interval], n: int) -> list[tuple[float, ...]]:
    axes = [[iv.lo + (iv.hi - iv.lo) * (k + 0.5) / n for k in range(n)] for iv in box]
    return list(itertools.product(*axes))


def _cell_volume(box: Sequence[Interval], n: int) -> float:
    dv = 1.0
    for iv in box:
        dv *= (iv.hi - iv.lo) / n
    return dv


def _lp_norm_numeric(
    net: Network, box: Sequence[Interval], p: int, n: int, out_index: int = 0
) -> float:
    dv = _cell_volume(box, n)
    acc = sum(abs(_forward(net, list(pt))[out_index]) ** p for pt in _grid_points(box, n))
    return (dv * acc) ** (1.0 / p)


def _grad_fd(net: Network, pt: Sequence[float], i: int, h: float = 1e-5) -> float:
    pp = list(pt)
    pm = list(pt)
    pp[i] += h
    pm[i] -= h
    return (_forward(net, pp)[0] - _forward(net, pm)[0]) / (2.0 * h)


def _h1_norm_numeric(net: Network, box: Sequence[Interval], n: int) -> float:
    dv = _cell_volume(box, n)
    dim = len(box)
    acc = 0.0
    for pt in _grid_points(box, n):
        u = _forward(net, list(pt))[0]
        acc += u * u + sum(_grad_fd(net, pt, i) ** 2 for i in range(dim))
    return math.sqrt(dv * acc)


def _layer_cake_numeric(
    net: Network, box: Sequence[Interval], n: int, transform: str, power: int
) -> float:
    dv = _cell_volume(box, n)
    acc = 0.0
    for pt in _grid_points(box, n):
        u = _forward(net, list(pt))[0]
        g = abs(u) if transform == "abs" else max(0.0, u)
        acc += g**power
    return dv * acc


# --------------------------------------------------------------------------- #
# Networks.
# --------------------------------------------------------------------------- #
def _tanh_net() -> Network:
    return Network(
        [
            affine_layer([[1.2, 0.7], [-0.9, 1.1]], [0.1, -0.2]),
            TanhLayer(),
            affine_layer([[1.0, -1.5]], [0.05]),
        ]
    )


def _relu_net() -> Network:
    return Network(
        [
            affine_layer([[1.0, -1.0], [0.5, 0.5]], [0.2, -0.1]),
            ReLULayer(),
            affine_layer([[1.0, 2.0]], [0.05]),
        ]
    )


_BOX = [Interval(-0.8, 0.8), Interval(-0.8, 0.8)]


# --------------------------------------------------------------------------- #
# Certified L^p norm.
# --------------------------------------------------------------------------- #
def test_l2_norm_taylor_sound_and_tight() -> None:
    net = _tanh_net()
    cert = certified_lp_norm(net, _BOX, p=2, subdivisions=6)
    assert isinstance(cert, LpNormCertificate)
    assert cert.p == 2 and cert.method == "taylor" and cert.scope == "box"
    ref = _lp_norm_numeric(net, _BOX, 2, 200)
    assert cert.enclosure.lo - 1e-3 <= ref <= cert.enclosure.hi + 1e-3
    assert cert.mass.lo >= 0.0  # mass is non-negative


def test_l2_norm_interval_method_sound() -> None:
    net = _tanh_net()
    cert = certified_lp_norm(net, _BOX, p=2, method="interval", subdivisions=8)
    ref = _lp_norm_numeric(net, _BOX, 2, 200)
    assert cert.method == "interval"
    assert cert.enclosure.lo - 1e-3 <= ref <= cert.enclosure.hi + 1e-2


def test_l4_norm_power_of_two_root() -> None:
    net = _tanh_net()
    cert = certified_lp_norm(net, _BOX, p=4, subdivisions=6)
    ref = _lp_norm_numeric(net, _BOX, 4, 200)
    assert cert.p == 4
    assert cert.enclosure.lo - 1e-3 <= ref <= cert.enclosure.hi + 1e-2


def test_l2_norm_relu_net_sound_both_methods() -> None:
    net = _relu_net()
    ref = _lp_norm_numeric(net, _BOX, 2, 220)
    for method in ("taylor", "interval"):
        cert = certified_lp_norm(net, _BOX, p=2, method=method, subdivisions=10)
        assert cert.enclosure.lo - 2e-2 <= ref <= cert.enclosure.hi + 2e-2, method


def test_l3_odd_norm_interval_method() -> None:
    # odd p requires the interval (|.|) method; taylor must reject it.
    net = _tanh_net()
    cert = certified_lp_norm(net, _BOX, p=3, method="interval", subdivisions=8)
    ref = _lp_norm_numeric(net, _BOX, 3, 200)
    assert cert.enclosure.lo - 1e-2 <= ref <= cert.enclosure.hi + 1e-2


def test_lp_taylor_rejects_odd_p() -> None:
    with pytest.raises(ValueError):
        certified_lp_norm(_tanh_net(), _BOX, p=3, method="taylor")


def test_lp_more_subdivisions_tighten() -> None:
    net = _tanh_net()
    widths = [certified_lp_norm(net, _BOX, p=2, subdivisions=s).width for s in (1, 2, 4, 8)]
    assert all(b <= a + 1e-12 for a, b in zip(widths, widths[1:], strict=False))
    assert widths[-1] < widths[0]


# --------------------------------------------------------------------------- #
# Certified Sobolev norm.
# --------------------------------------------------------------------------- #
def test_sobolev_order0_equals_l2() -> None:
    net = _tanh_net()
    h0 = certified_sobolev_norm(net, _BOX, order=0, subdivisions=6)
    l2 = certified_lp_norm(net, _BOX, p=2, subdivisions=6)
    assert isinstance(h0, SobolevNormCertificate)
    assert h0.order == 0 and h0.seminorm_masses == ()
    # same underlying L^2 mass
    assert abs(h0.value_mass.mid - l2.mass.mid) < 1e-9


def test_sobolev_h1_sound_vs_finite_difference() -> None:
    net = _tanh_net()
    cert = certified_sobolev_norm(net, _BOX, order=1, subdivisions=8)
    assert cert.order == 1 and len(cert.seminorm_masses) == 2
    ref = _h1_norm_numeric(net, _BOX, 120)
    # H^1 must dominate L^2 and enclose the finite-difference reference.
    assert cert.enclosure.lo <= ref <= cert.enclosure.hi
    assert cert.enclosure.hi >= certified_lp_norm(net, _BOX, p=2, subdivisions=8).enclosure.lo


def test_sobolev_h1_relu_net_sound() -> None:
    net = _relu_net()
    cert = certified_sobolev_norm(net, _BOX, order=1, subdivisions=10)
    ref = _h1_norm_numeric(net, _BOX, 120)
    assert cert.enclosure.lo <= ref <= cert.enclosure.hi


def test_sobolev_order_ge_2_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        certified_sobolev_norm(_tanh_net(), _BOX, order=2)


# --------------------------------------------------------------------------- #
# Certified layer-cake integral.
# --------------------------------------------------------------------------- #
def test_layer_cake_abs_sound() -> None:
    net = _tanh_net()
    cert = certified_layer_cake_integral(net, _BOX, transform="abs", subdivisions=10)
    ref = _layer_cake_numeric(net, _BOX, 200, "abs", 1)
    assert cert.enclosure.lo - 1e-2 <= ref <= cert.enclosure.hi + 1e-2


def test_layer_cake_relu_on_relu_net_sound() -> None:
    net = _relu_net()
    cert = certified_layer_cake_integral(net, _BOX, transform="relu", subdivisions=12)
    ref = _layer_cake_numeric(net, _BOX, 200, "relu", 1)
    assert cert.enclosure.lo - 2e-2 <= ref <= cert.enclosure.hi + 2e-2


def test_layer_cake_bad_transform_raises() -> None:
    with pytest.raises(ValueError):
        certified_layer_cake_integral(_tanh_net(), _BOX, transform="square")
