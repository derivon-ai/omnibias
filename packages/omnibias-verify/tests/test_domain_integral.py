# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Certified multivariate domain integral ``int_box f dx``.

Soundness rule (``.cursor/rules/verified-enclosures.mdc``): the returned
enclosure must contain the true integral, checked against a dense deterministic
grid quadrature **and** a random Monte-Carlo estimate. Branch-and-bound may only
tighten -- never break soundness.
"""

from __future__ import annotations

import itertools
import math
import random
from collections.abc import Callable, Sequence

from omnibias.core.verified.interval import Interval
from omnibias.core.verified.taylor_model_mv import TaylorModelMV
from omnibias.verify import (
    DomainIntegralCertificate,
    LinearLayer,
    Network,
    TanhLayer,
    affine_layer,
    certified_domain_integral,
    certified_network_integral,
    network_integrand_model,
)


# --------------------------------------------------------------------------- #
# Numerical references (a "true value" proxy for the integral).
# --------------------------------------------------------------------------- #
def _forward(net: Network, x: Sequence[float]) -> list[float]:
    vec = list(x)
    for layer in net:
        if isinstance(layer, LinearLayer):
            vec = [
                sum(w * xi for w, xi in zip(row, vec, strict=True)) + b
                for row, b in zip(layer.weight, layer.bias, strict=True)
            ]
        else:  # TanhLayer for these tests
            vec = [math.tanh(v) for v in vec]
    return vec


def _midpoint_integral(
    fn: Callable[[tuple[float, ...]], float], box: Sequence[Interval], n: int
) -> float:
    axes = [[iv.lo + (iv.hi - iv.lo) * (k + 0.5) / n for k in range(n)] for iv in box]
    dv = 1.0
    for iv in box:
        dv *= (iv.hi - iv.lo) / n
    return dv * sum(fn(pt) for pt in itertools.product(*axes))


def _mc_integral(
    fn: Callable[[tuple[float, ...]], float], box: Sequence[Interval], n: int, seed: int
) -> float:
    rng = random.Random(seed)
    vol = 1.0
    for iv in box:
        vol *= iv.hi - iv.lo
    acc = 0.0
    for _ in range(n):
        pt = tuple(rng.uniform(iv.lo, iv.hi) for iv in box)
        acc += fn(pt)
    return vol * acc / n


# --------------------------------------------------------------------------- #
# Polynomial oracle: exact analytic integral is known.
# --------------------------------------------------------------------------- #
def _poly_model(cell: tuple[Interval, ...]) -> TaylorModelMV:
    """The integrand ``x^2 + y^2`` as an exact order-2 Taylor model over ``cell``."""
    center = [iv.mid for iv in cell]
    radius = [0.5 * iv.width for iv in cell]
    x = TaylorModelMV.coordinate(0, center, radius, 2)
    y = TaylorModelMV.coordinate(1, center, radius, 2)
    return x.pow_int(2) + y.pow_int(2)


def test_polynomial_domain_integral_is_essentially_exact() -> None:
    # int_[0,1]^2 (x^2 + y^2) dx dy = 1/3 + 1/3 = 2/3, exactly captured by an
    # order-2 TM (remainder 0), so the enclosure is exact up to interval rounding.
    box = [Interval(0.0, 1.0), Interval(0.0, 1.0)]
    exact = 2.0 / 3.0
    for subs in (1, 2, 4):
        cert = certified_domain_integral(_poly_model, box, subdivisions=subs)
        assert isinstance(cert, DomainIntegralCertificate)
        assert cert.dim == 2
        assert cert.cells == subs * subs
        assert cert.enclosure.contains(exact)
        assert cert.width < 1e-9
        assert abs(cert.estimate - exact) < 1e-9


def test_interval_fallback_oracle_uses_volume() -> None:
    # A crude constant interval enclosure f in [1, 3] over the unit square gives
    # int in [1, 3] * area = [1, 3]; contains the true integral of any f in that band.
    box = [Interval(0.0, 1.0), Interval(0.0, 1.0)]

    def const_band(cell: tuple[Interval, ...]) -> Interval:
        return Interval(1.0, 3.0)

    cert = certified_domain_integral(const_band, box, subdivisions=3)
    assert cert.enclosure.contains(2.0)  # e.g. constant f == 2
    assert cert.enclosure.lo <= 1.0 + 1e-12
    assert cert.enclosure.hi >= 3.0 - 1e-12


# --------------------------------------------------------------------------- #
# Network read-out integrals: sound vs dense grid + random sample.
# --------------------------------------------------------------------------- #
def _tanh_net() -> Network:
    return Network(
        [
            affine_layer([[1.2, 0.7], [-0.9, 1.1]], [0.1, -0.2]),
            TanhLayer(),
            affine_layer([[1.0, -1.5]], [0.05]),
        ]
    )


def test_network_integral_sound_vs_grid_and_random() -> None:
    net = _tanh_net()
    box = [Interval(-0.8, 0.8), Interval(-0.8, 0.8)]

    def g(pt: tuple[float, ...]) -> float:
        return _forward(net, list(pt))[0]

    cert = certified_network_integral(net, box, order=4, subdivisions=5)
    grid_ref = _midpoint_integral(g, box, 200)
    mc_ref = _mc_integral(g, box, 40000, seed=0)
    # Sound: the certified enclosure contains both references (up to their own
    # small quadrature / sampling error).
    assert cert.enclosure.lo - 1e-3 <= grid_ref <= cert.enclosure.hi + 1e-3
    assert cert.enclosure.lo - 5e-3 <= mc_ref <= cert.enclosure.hi + 5e-3


def test_network_integral_weights_readout() -> None:
    # Two-output net; certify the linear read-out w . net + bias directly.
    net = Network(
        [
            affine_layer([[0.8, -0.4], [0.5, 0.9]], [0.0, 0.1]),
            TanhLayer(),
            affine_layer([[1.3, -0.7], [0.2, 0.6]], [-0.05, 0.1]),
        ]
    )
    box = [Interval(-0.6, 0.6), Interval(-0.6, 0.6)]
    weights = [0.5, -1.2]
    bias = 0.3

    def g(pt: tuple[float, ...]) -> float:
        y = _forward(net, list(pt))
        return weights[0] * y[0] + weights[1] * y[1] + bias

    cert = certified_network_integral(
        net, box, order=4, subdivisions=5, weights=weights, bias=bias
    )
    grid_ref = _midpoint_integral(g, box, 200)
    assert cert.enclosure.lo - 1e-3 <= grid_ref <= cert.enclosure.hi + 1e-3


def test_network_l2_mass_is_nonneg_and_sound() -> None:
    # power=2 integrates net(x)^2 (the L^2 mass): non-negative and sound.
    net = _tanh_net()
    box = [Interval(-0.7, 0.7), Interval(-0.7, 0.7)]

    def g2(pt: tuple[float, ...]) -> float:
        return _forward(net, list(pt))[0] ** 2

    cert = certified_network_integral(net, box, order=4, subdivisions=5, power=2)
    grid_ref = _midpoint_integral(g2, box, 200)
    assert cert.enclosure.hi >= 0.0
    assert cert.enclosure.lo - 1e-3 <= grid_ref <= cert.enclosure.hi + 1e-3


def test_adaptive_refinement_tightens_and_stays_sound() -> None:
    net = _tanh_net()
    box = [Interval(-0.8, 0.8), Interval(-0.8, 0.8)]

    def g(pt: tuple[float, ...]) -> float:
        return _forward(net, list(pt))[0]

    grid_ref = _midpoint_integral(g, box, 200)
    coarse = certified_network_integral(net, box, order=3, subdivisions=1)
    fine = certified_network_integral(
        net, box, order=3, subdivisions=1, adaptive=True, max_cells=64, tol=1e-9
    )
    assert fine.refined
    assert fine.cells > coarse.cells
    assert fine.width < coarse.width  # branch-and-bound tightened
    for cert in (coarse, fine):
        assert cert.enclosure.lo - 1e-3 <= grid_ref <= cert.enclosure.hi + 1e-3


def test_more_subdivisions_tighten() -> None:
    net = _tanh_net()
    box = [Interval(-0.8, 0.8), Interval(-0.8, 0.8)]
    widths = [
        certified_network_integral(net, box, order=3, subdivisions=s).width
        for s in (1, 2, 4, 8)
    ]
    assert all(b <= a + 1e-15 for a, b in zip(widths, widths[1:], strict=False))
    assert widths[-1] < widths[0]


def test_network_integrand_model_bad_out_index_raises() -> None:
    net = _tanh_net()
    model = network_integrand_model(net, out_index=5)
    try:
        model((Interval(-0.1, 0.1), Interval(-0.1, 0.1)))
    except ValueError:
        return
    raise AssertionError("expected ValueError for out-of-range out_index")
