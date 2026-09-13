# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Exact Green bounds and a conditional gate for a clamped cubic beam.

On ``[0,L]``, the inverse ``K`` of the fourth derivative with ``u=u'=0``
at both endpoints has exact ``C -> C`` norm ``L**4/384``. On ``[0,1]``
its symmetric kernel, for ``x <= t``, is

    G(x,t) = x**2 * (1-t)**2 * (3*t - (1+2*t)*x) / 6.

Each branch is cubic in x, its first two derivatives are continuous at x=t,
and its third derivative has jump one. The kernel and its first derivative
vanish at both endpoints. Thus ``K h`` is the unique clamped C4 solution of
``v''''=h`` for every continuous h. The kernel is nonnegative because its
last factor is at least ``2*t*(1-t)`` when x<=t. Since
``K 1=x**2*(1-x)**2/24``, positivity gives the exact sup norm ``1/384``;
equality is attained for h=1 at x=1/2. Scaling gives the stated L**4 law.
Symmetry and the same row integral give ``||K||_L2 <= L**4/384`` by the
Schur estimate. The clamped self-adjoint fourth derivative therefore has
quadratic-form lower bound ``384/L**4``. Adding ``3*lambda*v**2`` for
real bounded v and lambda>=0 preserves this lower bound.

For ``u''''+lambda*u**3=f`` and a clamped C4 candidate q, suppose the
CALLER PROVES ``||q|| <= U`` and ``||q''''+lambda*q**3-f|| <= R`` uniformly
over the whole interval and ``0<=lambda<=lambda_max``. On the complete
C-space ball ``B_r(q)``, the full operator ``T(v)=K(f-lambda*v**3)`` has
defect at most Y=norm(K)*R and Lipschitz constant at most
Z=3*lambda_max*norm(K)*(U+r)**2. The finite gate ``Z<1, Y+Z*r<=r`` implies
a self-map contraction PROVIDED those candidate/residual/boundary hypotheses
are established. A fixed point is C4 and solves the original BVP by the
Green identity. Global classical uniqueness follows by integrating
``(u-v)*(u''''-v'''') + lambda*(u-v)*(u**3-v**3)=0`` twice by parts.

This module computes the exact scalar gate; it DOES NOT verify the supplied
whole-domain bounds or boundary conditions. Its result never independently
asserts PDE existence, a quantum continuum theorem, or a Lean tier. There
are no finite-mode tails in K. No differentiation backend is imported.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any


def _rational(value: int | Fraction) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, int | Fraction):
        raise TypeError("proof inputs must be int or Fraction")
    return Fraction(value)


def clamped_biharmonic_inverse_norm(length: int | Fraction = 1) -> Fraction:
    """Exact sup-norm inverse constant L**4/384 on the clamped interval."""
    length = _rational(length)
    if length <= 0:
        raise ValueError("interval length must be positive")
    return length**4 / 384


def clamped_biharmonic_green_kernel(
    x: int | Fraction,
    t: int | Fraction,
    *,
    length: int | Fraction = 1,
) -> Fraction:
    """Evaluate the exact nonnegative symmetric Green kernel at rational points."""
    x, t, length = (_rational(v) for v in (x, t, length))
    if length <= 0 or not 0 <= x <= length or not 0 <= t <= length:
        raise ValueError("require positive length and x,t in [0,length]")
    x, t = x / length, t / length
    if x > t:
        x, t = t, x
    return length**3 * x**2 * (1 - t) ** 2 * (3 * t - (1 + 2 * t) * x) / 6


@dataclass(frozen=True)
class ClampedCubicContraction:
    """Exact scalar conditions, with candidate proof obligations still external."""

    candidate_sup_upper: Fraction
    residual_sup_upper: Fraction
    coupling_upper: Fraction
    radius: Fraction
    length: Fraction
    inverse_norm: Fraction
    defect: Fraction
    contraction: Fraction
    self_map_margin: Fraction

    @property
    def passed(self) -> bool:
        """Whether the finite contraction inequalities pass, not a PDE claim."""
        return self.contraction < 1 and self.self_map_margin >= 0

    def to_payload(self) -> dict[str, Any]:
        """Record the remaining analytic input obligations explicitly."""
        return {
            "status": "PASS" if self.passed else "INCONCLUSIVE",
            "candidate_sup_upper": str(self.candidate_sup_upper),
            "residual_sup_upper": str(self.residual_sup_upper),
            "coupling_upper": str(self.coupling_upper),
            "radius": str(self.radius),
            "length": str(self.length),
            "green_operator_norm": str(self.inverse_norm),
            "defect_Y": str(self.defect),
            "contraction_Z": str(self.contraction),
            "self_map_margin": str(self.self_map_margin),
            "verification_kind": "EXACT_RATIONAL_FINITE_GATE",
            "external_premises": {
                "forcing_is_real_continuous": "UNVERIFIED",
                "actual_coupling_in_closed_interval_0_to_coupling_upper": "UNVERIFIED",
                "candidate_is_real_C4_and_clamped": "UNVERIFIED",
                "candidate_bound_on_whole_interval": "UNVERIFIED",
                "residual_bound_on_whole_interval_and_parameter_family": "UNVERIFIED",
            },
            "pde_existence_claim": False,
            "continuum_claim": False,
            "yang_mills_mass_gap_claim": False,
            "theorem_prover_verified": False,
            "mathlib_verified": False,
        }


def clamped_cubic_contraction(
    candidate_sup_upper: int | Fraction,
    residual_sup_upper: int | Fraction,
    coupling_upper: int | Fraction,
    radius: int | Fraction,
    *,
    length: int | Fraction = 1,
) -> ClampedCubicContraction:
    """Check a full-Green contraction gate from supplied, unverified U and R."""
    u, residual, coupling, radius, length = (
        _rational(value)
        for value in (candidate_sup_upper, residual_sup_upper, coupling_upper, radius, length)
    )
    if min(u, residual, coupling) < 0 or radius <= 0:
        raise ValueError("require nonnegative U, R, coupling and positive radius")
    norm = clamped_biharmonic_inverse_norm(length)
    defect = norm * residual
    contraction = 3 * coupling * norm * (u + radius) ** 2
    margin = radius - defect - contraction * radius
    return ClampedCubicContraction(
        u, residual, coupling, radius, length, norm, defect, contraction, margin
    )


__all__ = [
    "ClampedCubicContraction",
    "clamped_biharmonic_green_kernel",
    "clamped_biharmonic_inverse_norm",
    "clamped_cubic_contraction",
]
