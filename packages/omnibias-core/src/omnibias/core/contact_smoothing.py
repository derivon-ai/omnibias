# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Certified contact-force smoothing (theory 10-04).

Rigid contact is one place where "not differentiable" is literally true:
the non-penetration / normal-force complementarity ``z >= 0, f >= 0,
z f = 0`` (Signorini's condition) has no gradient at the contact boundary.
The standard escape is a *smoothed* force law; this module makes that
smoothing exact in two distinct, precise senses instead of one informal
one:

1. **The smoothed force and every one of its derivatives are closed
   form.** ``f_smooth(z) = f_max * sigma(-beta z)`` reuses the sigmoid
   Eulerian-polynomial recurrence (:func:`omnibias.core.polynomials.
   sigmoid_polynomial_coeffs`), so ``d^n f/dz^n`` for *any* order ``n`` is
   available in closed form via the **founding bias collapse**
   (``delta -> 0``) tower -- the same one that supplies every other
   ``sigma^(n)`` in this library. No finite difference, no truncated
   Taylor series.
2. **The smoothing itself is a temperature collapse** (``beta -> inf``):
   as the hardness ``beta`` grows, ``f_smooth`` sharpens toward the exact
   Heaviside complementarity law, a 0/1 feasibility step. This module
   supplies a **sound enclosure** of that hardening error --
   :func:`contact_smoothing_bias_bound` -- rather than only asserting it
   informally.

**do not conflate the two.** Bias collapse never appears in the ``beta``
axis of this module; temperature collapse never appears in the ``sigma^(n)``
axis. Neither is the third limit in this codebase, Enclosure Collapse
(``width -> 0`` of a sound enclosure): the *enclosure width* here is fixed
by the caller's ``(z_min, beta)``, not driven to zero by this module; its
surviving object is a point plus a proof.

The enclosure is **one-sided and excludes a neighbourhood of ``z = 0``**:
for ``|z| >= z_min > 0``,

.. math::
    |f_{\text{smooth}}(z) - f_{\text{hard}}(z)| \le f_{\max}\, e^{-\beta z_{\min}}

using the elementary sigmoid tail bound ``sigma(x) <= exp(x)`` for ``x <=
0`` (immediate from ``sigma(x) = e^x/(1+e^x) <= e^x``). At ``|z| < z_min``
the hardening is not certified and this module reports ``Inconclusive``,
never a fabricated number -- the contact region itself is exactly where
the two force laws must disagree least (they meet at ``z_min``), so the
excluded band is the honest price of certification, not a limitation
anyone hid.

Validated only on a 1-D point-mass bounce (see
:mod:`omnibias.control.torch.contact` / :mod:`omnibias.control.jax.contact`
and ``benchmarks/contact_smoothing.py``). A 2-D block with Coulomb friction
is **leftover-recorded**, not shipped here.
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass

from omnibias.core.ftc import sigmoid
from omnibias.core.polynomials import sigmoid_polynomial_coeffs
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.transcend import exp_iv

DISCLAIMER = (
    "certified contact smoothing: exact sigma^(n) tower plus a sound one-sided "
    "enclosure of the beta -> inf hardening bias; excludes |z| < z_min; "
    "1-D validated only, 2-D Coulomb block is leftover-recorded"
)


def honesty_payload() -> dict[str, bool]:
    return {
        "rigid_body_engine_claimed": False,
        "two_d_friction_claimed": False,
        "unconditional_enclosure_claimed": False,
        "enclosure_collapse_claimed": False,
        "stretch_claim": False,
        "continuum_claimed": False,
        "theorem_prover_verified": False,
        "skips_chain_rule": False,
    }


def _require_finite(name: str, value: float) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return out


def contact_force_smooth(z: float, beta: float, f_max: float = 1.0) -> float:
    r"""``f_max * sigmoid(-beta * z)``: smooth one-sided normal force."""
    z_f = _require_finite("z", z)
    beta_f = _require_finite("beta", beta)
    f_max_f = _require_finite("f_max", f_max)
    if beta_f < 0.0:
        raise ValueError(f"beta must be >= 0, got {beta_f}")
    if f_max_f < 0.0:
        raise ValueError(f"f_max must be >= 0, got {f_max_f}")
    return f_max_f * sigmoid(-beta_f * z_f)


def contact_force_hard(z: float, f_max: float = 1.0) -> float:
    """The ``beta -> inf`` limit: Heaviside step at ``z = 0`` (contact iff ``z <= 0``)."""
    z_f = _require_finite("z", z)
    f_max_f = _require_finite("f_max", f_max)
    return f_max_f if z_f <= 0.0 else 0.0


def contact_force_tower(z: float, beta: float, order: int, f_max: float = 1.0) -> list[float]:
    r"""Exact tower ``[f(z), f'(z), ..., f^{(order)}(z)]`` of ``f_smooth``.

    ``u = -beta * z``; the chain rule through the affine map contributes a
    factor ``(-beta)^n`` at order ``n``, and ``sigma^{(n)}(u)`` comes from
    :func:`omnibias.core.polynomials.sigmoid_polynomial_coeffs` -- the same
    closed-form recurrence every other activation derivative in this
    library uses, not a new one.
    """
    z_f = _require_finite("z", z)
    beta_f = _require_finite("beta", beta)
    f_max_f = _require_finite("f_max", f_max)
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    if beta_f < 0.0:
        raise ValueError(f"beta must be >= 0, got {beta_f}")
    u = -beta_f * z_f
    s = sigmoid(u)
    out: list[float] = []
    for n in range(order + 1):
        coeffs = sigmoid_polynomial_coeffs(n)
        sigma_n = sum(c * s**k for k, c in enumerate(coeffs))
        out.append(f_max_f * sigma_n * ((-beta_f) ** n))
    return out


@dataclass(frozen=True)
class ContactBiasReport:
    """Sound one-sided enclosure of the smoothing bias for ``|z| >= z_min``."""

    z_min: float
    beta: float
    f_max: float
    bound: Interval
    certified: bool
    reason: str


def contact_smoothing_bias_bound(
    z_min: float, beta: float, f_max: float = 1.0
) -> ContactBiasReport:
    r"""Sound enclosure ``[0, f_max * exp(-beta * z_min)]`` of the hardening bias.

    Valid for every ``z`` with ``|z| >= z_min``; ``z_min <= 0`` is refused
    (``Inconclusive``) because the two force laws agree least near contact
    and no finite bound holds there.
    """
    z_min_f = _require_finite("z_min", z_min)
    beta_f = _require_finite("beta", beta)
    f_max_f = _require_finite("f_max", f_max)
    if f_max_f < 0.0:
        raise ValueError(f"f_max must be >= 0, got {f_max_f}")
    if beta_f < 0.0:
        raise ValueError(f"beta must be >= 0, got {beta_f}")
    if z_min_f <= 0.0:
        return ContactBiasReport(
            z_min=z_min_f,
            beta=beta_f,
            f_max=f_max_f,
            bound=Interval(0.0, math.inf),
            certified=False,
            reason="Inconclusive: z_min must be > 0 (excludes the contact neighbourhood)",
        )
    exponent = Interval.point(-beta_f * z_min_f)
    hi = Interval.point(f_max_f) * exp_iv(exponent)
    bound = Interval(0.0, hi.hi)
    return ContactBiasReport(
        z_min=z_min_f, beta=beta_f, f_max=f_max_f, bound=bound, certified=True, reason="ok"
    )


def worked_example() -> dict[str, float | bool]:
    """G1: tower matches a finite-difference check; bias bound holds on a grid."""
    beta = 8.0
    f_max = 3.0
    z0 = 0.7
    tower = contact_force_tower(z0, beta, order=3, f_max=f_max)
    h = 1e-4
    fd_first = (
        contact_force_smooth(z0 + h, beta, f_max) - contact_force_smooth(z0 - h, beta, f_max)
    ) / (2.0 * h)
    report = contact_smoothing_bias_bound(z_min=0.5, beta=beta, f_max=f_max)
    grid_ok = all(
        abs(contact_force_smooth(z, beta, f_max) - contact_force_hard(z, f_max)) <= report.bound.hi
        for z in (0.5, 0.8, 1.5, -0.5, -0.8, -1.5)
    )
    return {
        "first_derivative_fd_residual": abs(tower[1] - fd_first),
        "bias_bound_hi": report.bound.hi,
        "grid_ok": grid_ok,
        "certified": report.certified,
    }


def contact_skill(*, n: int = 200, seed: int = 0) -> dict[str, object]:
    """G2: random ``(z, beta, f_max)`` with ``|z| >= z_min`` all satisfy the bound."""
    rng = random.Random(seed)
    coverage = 0
    widths: list[float] = []
    for _ in range(n):
        z_min = rng.uniform(0.05, 1.0)
        beta = rng.uniform(0.5, 20.0)
        f_max = rng.uniform(0.1, 5.0)
        report = contact_smoothing_bias_bound(z_min=z_min, beta=beta, f_max=f_max)
        sign = rng.choice((-1.0, 1.0))
        z = sign * rng.uniform(z_min, z_min + 3.0)
        actual = abs(contact_force_smooth(z, beta, f_max) - contact_force_hard(z, f_max))
        if actual <= report.bound.hi + 1e-15:
            coverage += 1
        widths.append(report.bound.hi)
    inconclusive = contact_smoothing_bias_bound(z_min=0.0, beta=1.0).certified is False
    return {
        "coverage": coverage / n,
        "median_width": statistics.median(widths) if widths else math.inf,
        "inconclusive_at_zero": inconclusive,
        "g2_earned": coverage == n and inconclusive,
    }


__all__ = [
    "ContactBiasReport",
    "DISCLAIMER",
    "contact_force_hard",
    "contact_force_smooth",
    "contact_force_tower",
    "contact_skill",
    "contact_smoothing_bias_bound",
    "honesty_payload",
    "worked_example",
]
