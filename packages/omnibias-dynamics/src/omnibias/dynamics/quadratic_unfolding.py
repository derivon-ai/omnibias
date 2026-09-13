# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""A uniform height-nonoscillation gate for the full quadratic unfolding.

Start with xdot=A*x-y+x**2+(mu2+mu3)*x*y+mu1*y**2 and
ydot=C*x+x**2+x*y+mu3*y**2, where mu1=nu**2*m1 and
mu2=nu*m2, mu3=nu*m3. For nu>0 set X=nu*x, Y=nu**2*y,
tau=t/nu, then v=X/Y, z=1/Y and d(s)/d(tau)=Y. On Y>0,
both time changes preserve orientation. The resulting exact field is

    v' = F = m1+m2*v-nu*v**3-z+A*nu*v*z-C*nu**2*v**2*z,
    z' = -z*N,  N=m3+v+nu*v**2+C*nu**2*v*z.

At nu=0 these equations define the continuous boundary extension; the
coordinate change from the original plane is not invertible at nu=0.

On the OPEN chart box -2<v<2, 0<z<4, the gate checks N(-2,z)<0,
N(2,z)>0, ell=N_v>0, and D=k*ell+C*nu**2*v*F_v>0, where
k=1-A*nu*v+C*nu**2*v**2. All bounds are evaluated on the closed box.
They hold, in particular, for 0<=nu<=1/100, 1/2<=A<=3/2,
1<=C<=3 and -1<=m1,m2,m3<=1, covering every normalized m direction.

Proof of the gate: the normal-nullcline is a unique graph v=V(z).
For psi(z)=F(V(z),z), implicit differentiation gives psi'=-D/ell<0.
At a height critical point, z''=-z*ell*psi. Nonconstant trajectories
cannot pass through psi=0, which would be an equilibrium. Hence maxima
occur at smaller heights than minima. Two successive extrema would have
the opposite height order, a contradiction. Each connected trajectory
segment staying in the open box has at most one height extremum and at
most two intersections with any horizontal level. Periodic orbits wholly
inside the box are excluded. This does not assert that a passage exists,
bound its duration, or control zeros of a composed return displacement.

The checker verifies finite interval inequalities used in this written
analytic argument. It asserts no formal-proof flag or Hilbert-16 result.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import isfinite

from omnibias.core.verified.interval import Interval, IntervalLike

_V_DOMAIN = Interval(-2.0, 2.0)
_Z_DOMAIN = Interval(0.0, 4.0)
_DEFAULT_NU = Interval(0.0, Interval.from_rational(Fraction(1, 100)).hi)
_DEFAULT_A = Interval(0.5, 1.5)
_DEFAULT_C = Interval(1.0, 3.0)
_DEFAULT_DIRECTION = Interval(-1.0, 1.0)


def _finite(value: IntervalLike, name: str) -> Interval:
    result = Interval.from_value(value)
    if not isfinite(result.lo) or not isfinite(result.hi):
        raise ValueError(f"{name} must have finite endpoints")
    return result


@dataclass(frozen=True)
class UnfoldingParameters:
    """Intervals for nu,A,C and the three weighted unfolding directions."""

    nu: Interval
    a: Interval
    c: Interval
    m1: Interval
    m2: Interval
    m3: Interval


def unfolding_parameters(
    *,
    nu: IntervalLike = _DEFAULT_NU,
    a: IntervalLike = _DEFAULT_A,
    c: IntervalLike = _DEFAULT_C,
    m1: IntervalLike = _DEFAULT_DIRECTION,
    m2: IntervalLike = _DEFAULT_DIRECTION,
    m3: IntervalLike = _DEFAULT_DIRECTION,
) -> UnfoldingParameters:
    """Construct a finite parameter box; defaults cover the proved full box.

    A negative nu or an overly broad box is permitted as input to the
    checker, which leaves unverified obligations unresolved.
    """
    return UnfoldingParameters(
        _finite(nu, "nu"),
        _finite(a, "A"),
        _finite(c, "C"),
        _finite(m1, "m1"),
        _finite(m2, "m2"),
        _finite(m3, "m3"),
    )


def scaled_unfolding_field(
    x: IntervalLike, y: IntervalLike, parameters: UnfoldingParameters
) -> tuple[Interval, Interval]:
    """Enclose d(X,Y)/d(tau), with X=nu*x, Y=nu**2*y and tau=t/nu.

    The equations extend algebraically to nu=0. No coordinate inversion or
    time-orientation claim is made there.
    """
    xx, yy = _finite(x, "X"), _finite(y, "Y")
    p = parameters
    return (
        xx.pow_int(2) - yy + (p.m2 + p.m3) * xx * yy + p.m1 * yy.pow_int(2)
        + p.a * p.nu * xx,
        xx * yy + p.m3 * yy.pow_int(2) + p.nu * xx.pow_int(2)
        + p.c * p.nu.pow_int(2) * xx,
    )


def family_chart_field(
    v: IntervalLike, z: IntervalLike, parameters: UnfoldingParameters
) -> tuple[Interval, Interval]:
    """Enclose the exact (v,z) family-chart field and its nu=0 extension."""
    vv, zz = _finite(v, "v"), _finite(z, "z")
    p = parameters
    f = (
        p.m1 + p.m2 * vv - p.nu * vv.pow_int(3) - zz
        + p.a * p.nu * vv * zz - p.c * p.nu.pow_int(2) * vv.pow_int(2) * zz
    )
    normal = p.m3 + vv + p.nu * vv.pow_int(2) + p.c * p.nu.pow_int(2) * vv * zz
    return f, -zz * normal


@dataclass(frozen=True)
class TurningGeometry:
    """Outward polynomial bounds, including D for psi'=-D/ell on N=0."""

    normal: Interval
    normal_v: Interval
    k: Interval
    f_v: Interval
    determinant_factor: Interval


def turning_geometry(
    v: IntervalLike, z: IntervalLike, parameters: UnfoldingParameters
) -> TurningGeometry:
    """Enclose N, ell, k, F_v and D on a whole phase/parameter box."""
    vv, zz = _finite(v, "v"), _finite(z, "z")
    p = parameters
    nu2 = p.nu.pow_int(2)
    normal = p.m3 + vv + p.nu * vv.pow_int(2) + p.c * nu2 * vv * zz
    ell = 1 + 2 * p.nu * vv + p.c * nu2 * zz
    k = 1 - p.a * p.nu * vv + p.c * nu2 * vv.pow_int(2)
    f_v = p.m2 - 3 * p.nu * vv.pow_int(2) + p.a * p.nu * zz - 2 * p.c * nu2 * vv * zz
    determinant_factor = k * ell + p.c * nu2 * vv * f_v
    return TurningGeometry(normal, ell, k, f_v, determinant_factor)


@dataclass(frozen=True)
class UnfoldingNonoscillationCheck:
    """Interval premises for the written one-height-extremum theorem.

    Phase bounds refer to closed boxes; the trajectory conclusion is for
    connected nonconstant trajectory segments staying in their interior.
    An unresolved result is not a counterexample to nonoscillation.
    """

    parameters: UnfoldingParameters
    v_domain: Interval
    z_domain: Interval
    left_normal: Interval
    right_normal: Interval
    geometry: TurningGeometry
    drift_slope: Interval | None
    unresolved_obligations: tuple[str, ...]

    @property
    def certified_at_most_one_height_extremum(self) -> bool:
        """Every computed sign premise of the analytic implication holds."""
        return not self.unresolved_obligations


def certify_unfolding_nonoscillation(
    parameters: UnfoldingParameters | None = None,
) -> UnfoldingNonoscillationCheck:
    """Check the full-parameter one-extremum gate on -2<v<2, 0<z<4.

    The default checks all unfolding directions simultaneously, including
    m3=0. Broader parameter boxes may also pass if their computed signs do.
    No interval subdivision or sampled-parameter inference is performed.
    """
    p = parameters if parameters is not None else unfolding_parameters()
    for name in ("nu", "a", "c", "m1", "m2", "m3"):
        _finite(getattr(p, name), name)
    left = turning_geometry(-2, _Z_DOMAIN, p).normal
    right = turning_geometry(2, _Z_DOMAIN, p).normal
    geometry = turning_geometry(_V_DOMAIN, _Z_DOMAIN, p)
    unresolved: list[str] = []
    if p.nu.lo < 0:
        unresolved.append("nu_nonnegative")
    if left.hi >= 0:
        unresolved.append("normal_negative_on_left")
    if right.lo <= 0:
        unresolved.append("normal_positive_on_right")
    if geometry.normal_v.lo <= 0:
        unresolved.append("normal_strictly_increasing_in_v")
    if geometry.determinant_factor.lo <= 0:
        unresolved.append("positive_turning_determinant")
    slope = None
    if geometry.normal_v.lo > 0:
        slope = -geometry.determinant_factor / geometry.normal_v
    return UnfoldingNonoscillationCheck(
        p, _V_DOMAIN, _Z_DOMAIN, left, right, geometry, slope, tuple(unresolved)
    )


__all__ = [
    "TurningGeometry",
    "UnfoldingNonoscillationCheck",
    "UnfoldingParameters",
    "certify_unfolding_nonoscillation",
    "family_chart_field",
    "scaled_unfolding_field",
    "turning_geometry",
    "unfolding_parameters",
]
