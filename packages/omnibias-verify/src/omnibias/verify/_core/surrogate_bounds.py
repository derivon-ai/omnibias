# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Certified bounds on binary-network *surrogate gradients*.

The binary unit pairs a hard ``sign`` forward with a smooth surrogate backward --
the derivative of a mollified step (see ``docs/theory-binary.md``). This module
turns the theory's claims into **rigorous, outward-rounded interval facts** using
the verified kernel (:mod:`omnibias.core.verified`), and packages the
strictly-one-signed ones as Lean-checkable certificates
(:mod:`omnibias.core.proof`):

* :func:`surrogate_kernel_iv` -- two-sided enclosure of the surrogate gradient
  kernel ``phi_beta(z)`` for the ``tanh`` / ``logistic`` / ``gaussian`` / ``cauchy``
  / ``box`` (STE) families;
* :func:`mollification_bias_iv` / :func:`agreement_margin_iv` -- how far the smooth
  surrogate sits from the hard sign outside a margin (Theorem 1);
* :func:`mollified_lipschitz_iv` -- the certified Lipschitz constant of the smooth
  forward ``tanh(beta z)``;
* :func:`jet_remainder_iv` -- a rigorous ``O(h^4)`` enclosure of the curvature
  (jet-STE) correction residual (Theorem 2);
* :func:`certify_no_dead_unit` (Theorem 3) and :func:`certify_agreement_margin`
  (Theorem 1) -- :class:`CertifiedBound`s whose positive enclosures the Lean kernel
  can discharge via ``enclosed_quantity_pos``.

Everything is **sound by construction**: an enclosure always contains the true
value. ``box`` (STE) has compact support, so its tail enclosure is exactly ``0`` --
the dead-unit failure mode, machine-checked rather than asserted.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from omnibias.core.proof import (
    check_certificate,
    generate_obligation,
    interval_certificate,
    lean_check_available,
)
from omnibias.core.proof.certificate import Cert
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.sigma import sigma_tower_interval
from omnibias.core.verified.transcend import exp_iv, sigmoid_iv, tanh_iv

#: Surrogate kernels with a verified enclosure (peak-normalised: ``phi(0) = 1``).
KERNELS: tuple[str, ...] = ("box", "tanh", "logistic", "gaussian", "cauchy")

_HALF = Interval.point(0.5)
_ONE = Interval.point(1.0)


def _square(u: Interval) -> Interval:
    """Tight non-negative enclosure of ``u**2`` (``pow_int`` alone may straddle 0)."""
    mag = u.mag  # outward-rounded max |x|
    mig = u.mig  # inward-rounded min |x| (0 if the interval straddles 0)
    hi = (Interval.point(mag) * Interval.point(mag)).hi
    lo = (Interval.point(mig) * Interval.point(mig)).lo
    return Interval(max(lo, 0.0), hi)


def _u(beta: float, z: float | Interval) -> Interval:
    """Rigorous enclosure of the scaled argument ``u = beta * z``."""
    return Interval.from_value(beta) * Interval.from_value(z)


def surrogate_kernel_iv(beta: float, z: float | Interval, *, kernel: str = "tanh") -> Interval:
    r"""Rigorous enclosure of the peak-normalised surrogate kernel ``phi_beta(z)``.

    With ``u = beta z`` the kernels are ``box`` \(=\mathbf 1_{|u|\le1}\),
    ``tanh`` \(=\operatorname{sech}^2 u\), ``logistic`` \(=4\sigma(u)(1-\sigma(u))\),
    ``gaussian`` \(=e^{-u^2/2}\) and ``cauchy`` \(=1/(1+u^2)\); all have peak ``1`` at
    ``z = 0``. ``z`` may be a point or an :class:`~omnibias.core.verified.Interval`
    (a region), in which case the result encloses ``phi`` over the whole region.
    """
    if kernel not in KERNELS:
        raise ValueError(f"unknown kernel {kernel!r}; choose from {KERNELS}")
    u = _u(beta, z)
    if kernel == "box":
        if u.lo >= -1.0 and u.hi <= 1.0:
            return Interval(1.0, 1.0)
        if u.lo > 1.0 or u.hi < -1.0:
            return Interval(0.0, 0.0)
        return Interval(0.0, 1.0)
    if kernel == "tanh":
        return _ONE - _square(tanh_iv(u))
    if kernel == "logistic":
        s = sigmoid_iv(u)
        return Interval.point(4.0) * s * (_ONE - s)
    if kernel == "gaussian":
        return exp_iv(-(_square(u) * _HALF))
    # cauchy: 1 / (1 + u^2); denominator >= 1 so the reciprocal is always defined.
    return (_ONE + _square(u)).reciprocal()


def agreement_margin_iv(beta: float, d: float) -> Interval:
    r"""Enclosure of ``tanh(beta d)`` -- the hard/smooth agreement level at margin ``d``.

    For ``z >= d > 0`` the smooth surrogate ``tanh(beta z)`` agrees with
    ``sign(z) = +1`` to within ``1 - tanh(beta d)`` (Theorem 1); this returns the
    rigorous enclosure of ``tanh(beta d)``.
    """
    if d <= 0.0:
        raise ValueError("agreement margin requires d > 0")
    return tanh_iv(_u(beta, d))


def mollification_bias_iv(beta: float, d: float) -> Interval:
    r"""Enclosure of the worst-case mollification bias ``1 - tanh(beta d)`` for ``|z| >= d``.

    The upper endpoint is a rigorous bound on ``|sign(z) - tanh(beta z)|`` outside
    the margin ``d``; it decreases to ``0`` as ``beta -> inf`` (Theorem 1).
    """
    return _ONE - agreement_margin_iv(beta, d)


def mollified_lipschitz_iv(beta: float) -> Interval:
    r"""Certified Lipschitz constant of the smooth forward ``s(z) = tanh(beta z)``.

    ``|s'(z)| = beta (1 - tanh^2(beta z))`` is maximised at ``z = 0`` where it equals
    ``beta``; this returns the rigorous enclosure of that certified supremum.
    """
    if beta <= 0.0:
        raise ValueError("beta must be positive")
    slope0 = sigma_tower_interval("tanh", Interval.point(0.0), 1)[1]
    return Interval.from_value(beta) * slope0


def tail_gradient_iv(beta: float, d: float, *, kernel: str = "tanh") -> Interval:
    r"""Enclosure of the surrogate kernel at margin ``d`` (its tail supremum).

    All kernels here are unimodal and decreasing in ``|z|``, so ``phi(d).hi`` upper
    bounds the gradient any unit with ``|z| >= d`` can receive. For ``box`` with
    ``d > 1/beta`` this is exactly ``0`` -- the certified dead-zone of the STE.
    """
    if d <= 0.0:
        raise ValueError("tail gradient requires d > 0")
    return surrogate_kernel_iv(beta, d, kernel=kernel)


def jet_remainder_iv(beta: float, z: float, *, window: float | None = None) -> Interval:
    r"""Rigorous ``O(h^4)`` enclosure of the jet-STE correction residual (Theorem 2).

    Let ``s(z) = tanh(beta z)`` and ``h`` the window half-width (default ``1/beta``).
    The ideal backward is the window-average ``W = (1/2h) \int_{-h}^{h} s'(z+u) du``;
    the curvature-corrected slope is ``s'(z) + (h^2/6) s'''(z)``. This returns a
    symmetric interval enclosing ``W - (s' + (h^2/6) s''')``, bounded by the Lagrange
    remainder ``h^4/120 * sup_{[z-h, z+h]} |s^{(5)}|`` -- so the correction is
    accurate to fourth order, versus the plain surrogate's ``O(h^2)``.
    """
    h = (1.0 / beta) if window is None else window
    if h <= 0.0:
        raise ValueError("window half-width must be positive")
    window_z = Interval.from_value(z) + Interval(-h, h)
    u = Interval.from_value(beta) * window_z
    tanh5 = sigma_tower_interval("tanh", u, 5)[5]
    s5 = Interval.from_value(beta).pow_int(5) * tanh5  # s^(5) = beta^5 tanh^(5)(beta z)
    coeff = Interval.point(h).pow_int(4) * Interval.from_rational(Fraction(1, 120))
    bound = (coeff * Interval.point(s5.mag)).hi
    return Interval(-bound, bound)


@dataclass(frozen=True)
class CertifiedBound:
    """A surrogate-gradient interval fact plus its sealed (Lean-checkable) certificate."""

    claim: str
    interval: Interval
    certificate: Cert

    @property
    def lean_obligation(self) -> str | None:
        """The generated Lean obligation source, or ``None`` if not one-signed."""
        return generate_obligation(self.certificate)

    @property
    def lean_checkable(self) -> bool:
        """``True`` iff a finite Lean obligation can be emitted (strictly one-signed)."""
        return self.lean_obligation is not None

    def lean_verify(self) -> bool:
        """Run the Lean kernel on the certificate (``False`` if toolchain absent)."""
        return check_certificate(self.certificate).verified


def certify_no_dead_unit(
    beta: float, lo: float, hi: float, *, kernel: str = "tanh"
) -> CertifiedBound:
    r"""Certify that the surrogate gradient is strictly positive on ``z in [lo, hi]``.

    Encloses ``phi_beta`` over the region; when the lower endpoint is positive the
    Lean kernel can prove ``0 < phi`` everywhere on the box (Theorem 3: no dead
    unit). ``cauchy``'s heavy tails make this hold on *any* finite box; ``box``
    (STE) fails it beyond ``|z| > 1/beta`` -- exactly the dead-unit dichotomy.
    """
    if lo > hi:
        raise ValueError("require lo <= hi")
    region = Interval(lo, hi)
    phi = surrogate_kernel_iv(beta, region, kernel=kernel)
    claim = (
        f"surrogate kernel '{kernel}' (beta={beta}) is in [{phi.lo!r}, {phi.hi!r}] "
        f"on z in [{lo}, {hi}] (no dead unit when lo > 0)"
    )
    cert = interval_certificate(
        claim,
        phi,
        meta={"kind": "no_dead_unit", "kernel": kernel, "beta": beta},
    )
    return CertifiedBound(claim=claim, interval=phi, certificate=cert)


def certify_agreement_margin(beta: float, d: float) -> CertifiedBound:
    r"""Certify the hard/smooth agreement margin beyond ``d`` exceeds ``1/2``.

    Encloses ``tanh(beta d) - 1/2`` (Theorem 1); a positive lower endpoint means the
    smooth ``tanh`` surrogate agrees with the hard sign by more than ``1/2`` for all
    ``|z| >= d``, which the Lean kernel discharges via ``enclosed_quantity_pos``.
    """
    q = agreement_margin_iv(beta, d) - _HALF
    claim = (
        f"tanh(beta*d) - 1/2 in [{q.lo!r}, {q.hi!r}] for beta={beta}, d={d} "
        f"(agreement margin beyond 1/2)"
    )
    cert = interval_certificate(
        claim,
        q,
        meta={"kind": "agreement_margin", "beta": beta, "d": d},
    )
    return CertifiedBound(claim=claim, interval=q, certificate=cert)


__all__ = [
    "CertifiedBound",
    "KERNELS",
    "agreement_margin_iv",
    "certify_agreement_margin",
    "certify_no_dead_unit",
    "jet_remainder_iv",
    "lean_check_available",
    "mollification_bias_iv",
    "mollified_lipschitz_iv",
    "surrogate_kernel_iv",
    "tail_gradient_iv",
]
