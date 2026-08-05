# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Containment regressions for the branch-and-bound enclosure.

An enclosure is a *logical claim* -- every value of the read-out over the input
box lies inside ``[lo, hi]`` -- not an estimate of the range.  The direction of
the error is therefore what matters and the magnitude is nearly irrelevant: a
wider interval is merely loose, a narrower one is **false**.  Every assertion
here is exact (no tolerance), because a tolerance-based check cannot tell those
two opposite cases apart, which is precisely how the defect these tests pin
down survived a green suite.

``tol`` is a *search* budget: it may cost tightness, never validity.
"""

from __future__ import annotations

import random

from omnibias.core.verified.interval import Interval
from omnibias.verify import (
    LinearLayer,
    Network,
    certify_robustness,
    output_range,
    scalar_readout_range,
)


def _identity_net() -> Network:
    """``f(x) = x`` -- the exact range over a box is the box itself."""
    return Network([LinearLayer([[1.0]], [0.0])])


def test_output_range_encloses_exact_affine_range() -> None:
    """``f(x) = x`` on ``[0, 2]`` has range exactly ``[0, 2]``; both must be inside."""
    res = output_range(_identity_net(), [Interval(0.0, 2.0)], 0, order=2)
    assert res.enclosure.lo <= 0.0, (
        f"enclosure.lo={res.enclosure.lo!r} excludes f(0)=0 -- the enclosure is not sound"
    )
    assert res.enclosure.hi >= 2.0, (
        f"enclosure.hi={res.enclosure.hi!r} excludes f(2)=2 -- the enclosure is not sound"
    )


def test_output_range_soundness_is_independent_of_tol() -> None:
    """``tol`` trades tightness for speed; it must never buy an invalid bound."""
    net = _identity_net()
    for tol in (1e-12, 1e-9, 1e-6, 1e-3, 1e-2, 0.1):
        enc = output_range(net, [Interval(0.0, 2.0)], 0, order=2, tol=tol).enclosure
        assert enc.lo <= 0.0, f"tol={tol}: enclosure.lo={enc.lo!r} > 0.0"
        assert enc.hi >= 2.0, f"tol={tol}: enclosure.hi={enc.hi!r} < 2.0"


def test_output_range_encloses_random_affine_nets() -> None:
    """Sweep affine nets, whose exact range is known from the two endpoints."""
    rng = random.Random(0)
    for _ in range(300):
        w = rng.uniform(-3.0, 3.0)
        b = rng.uniform(-2.0, 2.0)
        lo_x, hi_x = sorted((rng.uniform(-3.0, 3.0), rng.uniform(-3.0, 3.0)))
        if hi_x - lo_x < 1e-3:
            continue
        net = Network([LinearLayer([[w]], [b])])
        enc = output_range(net, [Interval(lo_x, hi_x)], 0).enclosure
        true_lo = min(w * lo_x + b, w * hi_x + b)
        true_hi = max(w * lo_x + b, w * hi_x + b)
        assert enc.lo <= true_lo, (
            f"w={w!r} b={b!r} box=[{lo_x!r},{hi_x!r}]: "
            f"enclosure.lo={enc.lo!r} > true min {true_lo!r}"
        )
        assert enc.hi >= true_hi, (
            f"w={w!r} b={b!r} box=[{lo_x!r},{hi_x!r}]: "
            f"enclosure.hi={enc.hi!r} < true max {true_hi!r}"
        )


def test_scalar_readout_encloses_exact_affine_range() -> None:
    """The same containment duty on the read-out path used by robustness margins."""
    net = Network([LinearLayer([[1.0], [0.0]], [0.0, 0.0])])
    enc = scalar_readout_range(net, [Interval(0.0, 2.0)], [1.0, -1.0]).enclosure
    assert enc.lo <= 0.0, f"enclosure.lo={enc.lo!r} excludes the margin at x=0"
    assert enc.hi >= 2.0, f"enclosure.hi={enc.hi!r} excludes the margin at x=2"


def test_certify_robustness_rejects_provably_non_robust_network() -> None:
    r"""A network whose true margin dips below zero must never be certified.

    Outputs are ``(x - delta, 0)`` so the margin ``out[0] - out[1]`` is ``x - delta``.
    Over ``|x - 1| <= 1`` (the box ``[0, 2]``) its minimum is ``-delta < 0``, so the
    network is *not* robust and ``certified`` must be ``False``.
    """
    for delta in (5e-7, 8e-7, 9e-7):
        net = Network([LinearLayer([[1.0], [0.0]], [-delta, 0.0])])
        cert = certify_robustness(net, [1.0], eps=1.0, true_label=0)
        assert not cert.certified, (
            f"delta={delta!r}: certified=True although the true margin minimum is "
            f"{-delta!r} < 0 -- this is a false robustness certificate"
        )
