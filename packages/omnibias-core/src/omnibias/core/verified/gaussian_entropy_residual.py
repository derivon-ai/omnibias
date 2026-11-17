# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Global spatial and value bounds from an exact projected Gaussian proposal."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

from omnibias.core.gaussian_jet import _Arithmetic
from omnibias.core.verified.gaussian_affine import _dot, _validate_vector
from omnibias.core.verified.gaussian_entropy_unique import (
    GaussianEntropyUnique,
    _unique_intact,
    _unique_source_slots,
)
from omnibias.core.verified.gaussian_heat import _finite, _rational
from omnibias.core.verified.gaussian_projection_minimum import _combined_budget, _ProjectionOracle
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.inverse_gaussian_heat import _pair, _sqrt_upper
from omnibias.core.verified.mean_map import _probabilities, _strict
from omnibias.core.verified.polyhedral_projection import _project_exact, _query_work_bound
from omnibias.core.verified.transcend import libm_fallback_used

if TYPE_CHECKING:
    from collections.abc import Sequence

    from omnibias.core.proof.certificate import Cert

Rational = int | Fraction
Vector = tuple[Fraction, ...]
Box = tuple[Interval, ...]


def _residual_intact(residual: GaussianEntropyResidual) -> bool:
    """Check nested digests and result binding after the caller reserves scans."""
    from omnibias.core.proof.certificate import encode_interval
    from omnibias.core.verified.inverse_gaussian_affine import _intact

    if not _intact(residual.certificate) or not _unique_intact(residual.unique):
        return False
    payload = residual.certificate.get("payload")
    if not isinstance(payload, dict) or payload.get("unique_certificate") != residual.unique.certificate:
        return False
    for name in ("probabilities", "projection_source", "point", "normal"):
        if payload.get(name) != [_pair(value) for value in getattr(residual, name)]:
            return False
    for name in ("posterior", "residual", "enclosure"):
        if payload.get(name) != [encode_interval(value) for value in getattr(residual, name)]:
            return False
    for name in ("residual_squared_bound", "contraction_bound", "strong_convexity_bound", "suboptimality_bound"):
        if payload.get(name) != _pair(getattr(residual, name)):
            return False
    return (payload.get("error_bound") == residual.error_bound.hex()
            and payload.get("value") == encode_interval(residual.value)
            and payload.get("minimum") == encode_interval(residual.minimum))


def _residual_source_slots(unique: GaussianEntropyUnique) -> int:
    """Reserve source snapshots, one oracle and its retained output/scratch."""
    model = unique.domain.model
    return 2*_unique_source_slots(unique)+512*(model.mode_count+1)*(model.dimension+1)+4096


def _residual_work_bound(unique: GaussianEntropyUnique) -> int:
    """Conservative work for oracle setup and one complete evaluation."""
    model = unique.domain.model
    return _query_work_bound(unique.domain.projector)+2048*(model.mode_count+1)*(model.dimension+1)+4096


@dataclass(frozen=True)
class _ResidualData:
    probabilities: Vector
    projection_source: Vector
    point: Vector
    normal: Vector
    posterior: Box
    residual: Box
    residual_squared_bound: Fraction
    contraction_bound: Fraction
    strong_convexity_bound: Fraction
    error_bound: float
    enclosure: Box
    value: Interval
    minimum: Interval
    suboptimality_bound: Fraction


class _ResidualOracle:
    """Shared-meter kernel; caller validates sources and enters strict mode."""

    def __init__(self, unique: GaussianEntropyUnique, arithmetic: _Arithmetic) -> None:
        self.unique, self.arithmetic = unique, arithmetic
        self.domain, self.model = unique.domain, unique.domain.model
        self.projection = _ProjectionOracle(self.model, self.domain.projector, self.domain.target, arithmetic)
        threshold = arithmetic.mul(arithmetic.mul(Fraction(2), self.model.viscosity), self.model.time)
        self.contraction = arithmetic._coefficient(unique.variance_upper_bound/threshold)
        self.curvature = arithmetic._coefficient(unique.uniqueness_margin/arithmetic.mul(threshold, self.model.time))

    def evaluate(self, probabilities: Vector) -> _ResidualData:
        """Evaluate exact simplex weights; source/input validity is the caller's job."""
        a, model, projection = self.arithmetic, self.model, self.projection
        a._work(512*(model.mode_count+1)*(model.dimension+1)+128)
        mean = tuple(a.sum(tuple(a.mul(p, row[axis]) for p, row in zip(probabilities, projection.relative, strict=True)))
                     for axis in range(model.dimension))
        source = tuple(a.add(shift, value) for shift, value in zip(projection.shift, mean, strict=True))
        solved = _project_exact(self.domain.projector, source, a)
        if solved is None:
            raise ArithmeticError("no feasible exact residual projection")
        point = solved[0]
        normal = tuple(a._coefficient(a.add(q, -x)/model.time) for q, x in zip(source, point, strict=True))
        relative_point = tuple(a.add(x, -origin) for x, origin in zip(point, projection.origin, strict=True))
        logits = tuple(a.add(offset, _dot(row, relative_point, a))
                       for row, offset in zip(projection.rates, projection.offsets, strict=True))
        # One exact common logit is removed before conversion, including a
        # common bias or a huge translated center. Equal logits share exactly
        # the same posterior, so grouping preserves their linear cancellations.
        reference_logit = max(logits)
        shifted = tuple(a.add(value, -reference_logit) for value in logits)
        if all(value == shifted[0] for value in shifted):
            probability = _rational(a._coefficient(Fraction(1, model.mode_count)))
            posterior = (probability,)*model.mode_count
        else:
            posterior = _probabilities(tuple(_rational(value) for value in shifted))
        grouped: dict[Fraction, tuple[Interval, Vector]] = {}
        for logit, weight, row in zip(shifted, posterior, projection.relative, strict=True):
            coefficient = tuple(a.add(value, -center) for value, center in zip(mean, row, strict=True))
            if logit in grouped:
                prior_weight, prior = grouped[logit]
                grouped[logit] = (prior_weight, tuple(a.add(left, right) for left, right in zip(prior, coefficient, strict=True)))
            else:
                grouped[logit] = (weight, coefficient)
        residual = []
        squared = Fraction(0)
        for axis in range(model.dimension):
            lower = upper = Fraction(0)
            for weight, coefficients in grouped.values():
                scalar_coefficient = a._coefficient(coefficients[axis]/model.time)
                if scalar_coefficient:
                    products = a.mul(scalar_coefficient, a.exact(Fraction(weight.lo))), a.mul(scalar_coefficient, a.exact(Fraction(weight.hi)))
                    lower, upper = a.add(lower, min(products)), a.add(upper, max(products))
            component = Interval(_rational(lower).lo, _rational(upper).hi)
            residual.append(component)
            magnitude = max(abs(a.exact(Fraction(component.lo))), abs(a.exact(Fraction(component.hi))))
            squared = a.add(squared, a.mul(magnitude, magnitude))
        radius_squared = a._coefficient(squared/a.mul(self.curvature, self.curvature))
        a._work(64)
        radius = _sqrt_upper(radius_squared, a.limits)
        exact_radius = a.exact(Fraction(radius))
        enclosure = tuple(Interval(_rational(max(a.add(x, -exact_radius), limits[0])).lo,
                                   _rational(min(a.add(x, exact_radius), limits[1])).hi)
                          for x, limits in zip(point, self.domain.physical_bounds, strict=True))
        value = projection.objective(tuple((x, x) for x in point))
        gap = a._coefficient(squared/a.mul(Fraction(2), self.curvature))
        lower_value = a.add(a.exact(Fraction(value.lo)), -gap)
        minimum = _finite(Interval(_rational(lower_value).lo, value.hi))
        return _ResidualData(probabilities, source, point, normal, posterior, tuple(residual), squared,
                             self.contraction, self.curvature, radius, enclosure, value, minimum, gap)


@dataclass(frozen=True, init=False)
class GaussianEntropyResidual:
    r"""A feasible proposal, global minimum enclosure and spatial error bound.

    ``point`` is the exact projection of ``s*target+sum(p_j*a_j)`` for the
    supplied exact simplex probabilities, including boundary weights. They
    need not lie in the source posterior domain. ``normal`` is an exact member
    of the polyhedral normal cone at that point. ``residual`` encloses
    ``sum(a_j*(p_j-posterior_j))/s``, the original objective gradient plus that
    normal. It is not the unconstrained gradient at a constrained optimum.

    The source uniqueness margin supplies spatial curvature only throughout
    its recorded physical box. That box contains the proposal, optimizer and
    their segment. ``error_bound`` bounds their Euclidean distance;
    ``suboptimality_bound`` bounds the original objective gap exactly.
    Values use the same common-constant normalization as Gaussian entropy and
    projection minimization. No projection derivative or search is performed.
    """

    unique: GaussianEntropyUnique
    probabilities: Vector
    projection_source: Vector
    point: Vector
    normal: Vector
    posterior: Box
    residual: Box
    residual_squared_bound: Fraction
    contraction_bound: Fraction
    strong_convexity_bound: Fraction
    error_bound: float
    enclosure: Box
    value: Interval
    minimum: Interval
    suboptimality_bound: Fraction
    certificate: Cert

    def __init__(self) -> None:
        raise TypeError("use certify_gaussian_entropy_residual")


def _seal_residual(unique: GaussianEntropyUnique, data: _ResidualData) -> GaussianEntropyResidual:
    """Seal one already evaluated result; the caller reserves dependency copies."""
    from omnibias.core.proof.certificate import encode_interval, make_certificate

    if libm_fallback_used():
        raise RuntimeError("Gaussian entropy residual refuses prior conditional libm fallback history")
    certificate = make_certificate(
        claim="A feasible Gaussian proposal with global spatial and objective-gap bounds to its unique constrained minimizer.",
        payload={
            "type": "gaussian_entropy_residual", "unique_certificate": deepcopy(unique.certificate),
            "probabilities": [_pair(value) for value in data.probabilities],
            "projection_source": [_pair(value) for value in data.projection_source],
            "point": [_pair(value) for value in data.point], "normal": [_pair(value) for value in data.normal],
            "posterior": [encode_interval(value) for value in data.posterior],
            "residual": [encode_interval(value) for value in data.residual],
            "residual_squared_bound": _pair(data.residual_squared_bound),
            "contraction_bound": _pair(data.contraction_bound),
            "strong_convexity_bound": _pair(data.strong_convexity_bound),
            "error_bound": data.error_bound.hex(), "enclosure": [encode_interval(value) for value in data.enclosure],
            "value": encode_interval(data.value), "minimum": encode_interval(data.minimum),
            "suboptimality_bound": _pair(data.suboptimality_bound),
            "argument": "The source posterior domain encloses the original posterior pi(z) at EVERY z in its convex physical box K, not only at minimizers. Its expected squared-radius bound B therefore gives Cov_pi(z)(a)<=B I on K. The original objective Hessian is I/s-Cov/(2nu*s²), hence at least mu I with mu=(2nu*s-B)/(2nu*s²)>0 throughout K. For any exact simplex p, projection gives x=P_C(s*v+A*p), n=(s*v+A*p-x)/s in N_C(x), and r=grad F(x)+n=A*(p-pi(x))/s. The entire projected simplex and every original global minimizer lie in K. Thus the feasible segment from x to the unique optimizer x* stays in K. Strong convexity along that segment and the normal-cone inequality give F(x*)>=F(x)+r.(x*-x)+mu*|x*-x|²/2>=F(x)-|r|²/(2mu). Strong monotonicity of grad F+N_C similarly gives |x-x*|<=|r|/mu. Exact logit centering precedes rigorous posterior evaluation; equal-logit groups share a probability and their exact linear coefficients are combined. Exact endpoint arithmetic bounds |r|²; an exact-comparison outward square root yields the distance bound. The optimizer box intersects this ball's coordinate bounds with K. The original objective is rigorously enclosed at the exactly feasible x, completing the global value interval.",
            "normalization": "For c=a_0 and z=x-c, F=|z|²/(2s)-target.z-2nu*logsumexp((a_j-c).z/(2nu*s)+(b_j-b_0)-|a_j-c|²/(4nu*s)).",
            "scope": "Fixed exact Gaussian model, polyhedron and target bound by the source uniqueness certificate. Input weights can lie anywhere in the full simplex; no log(p), projection derivative, iterative convergence, general physical convexity, perturbation radius or external formal theorem-prover verification is asserted.",
        }, honesty={"unproven_claim": False, "unconditional_transcendentals": True, "continuum_pde_claim": False},
        meta={"transcend_backend": "mpmath"},
    )
    result = object.__new__(GaussianEntropyResidual)
    object.__setattr__(result, "unique", unique)
    for name in _ResidualData.__dataclass_fields__:
        object.__setattr__(result, name, getattr(data, name))
    object.__setattr__(result, "certificate", certificate)
    return result


def certify_gaussian_entropy_residual(
    unique: GaussianEntropyUnique, probabilities: Sequence[Rational],
) -> GaussianEntropyResidual | None:
    """Bound a feasible proposal's distance and gap without any minimizer search.

    Exact probability types, shape and nonnegativity are checked before
    resource preflight; their sum is checked by budgeted exact arithmetic.
    Invalid data raises. Damaged/conditional dependencies, exhausted resources
    or nonfinite enclosures return None. Missing rigorous backend or sticky
    libm history raises RuntimeError, without clearing that history.
    """
    if not isinstance(unique, GaussianEntropyUnique):
        raise TypeError("unique must be a GaussianEntropyUnique")
    _validate_vector(probabilities, unique.domain.model.mode_count, "probabilities")
    if any(value < 0 for value in probabilities):
        raise ValueError("probabilities must be nonnegative")
    a = _Arithmetic(_combined_budget(unique.domain.model.budget, unique.domain.projector.budget))
    try:
        source = _residual_source_slots(unique)
        a._entries(source)
        a.limits._work(4*source+_residual_work_bound(unique))
        a._work(4*source)
        weights = tuple(a.exact(value) for value in probabilities)
        total = a.sum(weights)
    except (ArithmeticError, ValueError):
        return None
    if total != 1:
        raise ValueError("probabilities must sum exactly to one")
    try:
        if not _unique_intact(unique):
            return None
        from omnibias.core.proof.certificate import certificate_is_unconditional

        if not certificate_is_unconditional(unique.certificate):
            return None
        if libm_fallback_used():
            raise RuntimeError("Gaussian entropy residual refuses prior conditional libm fallback history")
        with _strict():
            return _seal_residual(unique, _ResidualOracle(unique, a).evaluate(weights))
    except (ArithmeticError, ValueError):
        return None


__all__ = ["GaussianEntropyResidual", "certify_gaussian_entropy_residual"]
