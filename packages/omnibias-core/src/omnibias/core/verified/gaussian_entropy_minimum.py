# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Global Gaussian optimization by a differentiable interior entropy envelope."""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

from omnibias.core.gaussian_jet import _Arithmetic
from omnibias.core.verified.gaussian_affine import _dot
from omnibias.core.verified.gaussian_heat import _finite, _rational
from omnibias.core.verified.gaussian_posterior_domain import (
    GaussianPosteriorDomain,
    _domain_intact,
    _domain_source_slots,
)
from omnibias.core.verified.gaussian_projection_minimum import _combined_budget, _ProjectionOracle
from omnibias.core.verified.global_min import BoxMinimum, MinimumStatus, minimize_box
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.inverse_gaussian_heat import _pair
from omnibias.core.verified.mean_map import _strict
from omnibias.core.verified.polyhedral_projection import _project_exact, _query_work_bound
from omnibias.core.verified.transcend import libm_fallback_used, ln_iv

if TYPE_CHECKING:
    from collections.abc import Sequence

    from omnibias.core.proof.certificate import Cert

Vector = tuple[Fraction, ...]
Box = tuple[Interval, ...]
Pair = tuple[Fraction, Fraction]
Bounds = tuple[Pair, ...]
_ZERO = Interval.point(0.0)


def _interval(bounds: Pair) -> Interval:
    return Interval(_rational(bounds[0]).lo, _rational(bounds[1]).hi)


class _EntropyOracle:
    """Continuous envelope with a C1 interior gradient, even at projection switches."""

    def __init__(self, domain: GaussianPosteriorDomain, arithmetic: _Arithmetic) -> None:
        self.domain, self.arithmetic = domain, arithmetic
        self.model, self.projector = domain.model, domain.projector
        self.projection = _ProjectionOracle(domain.model, domain.projector, domain.target, arithmetic)
        self.relative = self.projection.relative
        self.scale = self.projection.scale
        self.shift = tuple(arithmetic.mul(self.model.time, value) for value in domain.target)
        twice_time = arithmetic.mul(Fraction(2), self.model.time)
        self.coefficients = tuple(arithmetic.add(arithmetic._coefficient(_dot(row, row, arithmetic)/twice_time),
                                 -arithmetic.mul(self.scale, arithmetic.add(bias, -self.model.biases[0])))
                                 for row, bias in zip(self.relative, self.model.biases, strict=True))
        arithmetic._work(256*(self.model.mode_count+1)*(self.model.dimension+1))
        self.logs = tuple(_finite(ln_iv(_rational(Fraction(count)))) for count in range(1, self.model.mode_count+1))
        self.log_two = _finite(ln_iv(Interval.point(2.0)))

    def _coordinates(self, box: Box) -> tuple[Pair, ...]:
        a = self.arithmetic
        return tuple((a.exact(Fraction(axis.lo)), a.exact(Fraction(axis.hi))) for axis in box)

    def _interpolate(self, vertex: Fraction, tail: Pair, coordinate: Pair) -> Pair:
        a = self.arithmetic
        values = tuple(a.add(value, a.mul(u, a.add(vertex, -value))) for u in coordinate for value in tail)
        return min(values), max(values)

    def _binary_entropy(self, coordinate: Pair) -> Interval:
        """Range of u log(u)+(1-u)log(1-u), continuously extended at 0 and 1."""
        a = self.arithmetic

        def at(value: Fraction) -> Interval:
            if value == 0 or value == 1:
                return _ZERO
            other = a.add(Fraction(1), -value)
            u, complement = _rational(value), _rational(other)
            return _finite(u*ln_iv(u)+complement*ln_iv(complement))

        left, right = at(coordinate[0]), at(coordinate[1])
        lower, upper = min(left.lo, right.lo), max(left.hi, right.hi)
        if coordinate[0] <= Fraction(1, 2) <= coordinate[1]:
            lower = min(lower, -self.log_two.hi)
        return Interval(max(lower, -self.log_two.hi), min(0.0, upper))

    def _tails(self, box: Box) -> tuple[list[Bounds], list[Pair], list[Interval]]:
        """Normalized tail means, linear expectations and entropies in O(m*d)."""
        a, modes = self.arithmetic, self.model.mode_count
        coordinates = self._coordinates(box)
        means: list[Bounds] = [()] * modes
        linear: list[Pair] = [(Fraction(0), Fraction(0))] * modes
        entropy = [_ZERO] * modes
        means[-1] = tuple((value, value) for value in self.relative[-1])
        linear[-1] = (self.coefficients[-1], self.coefficients[-1])
        for j in reversed(range(modes-1)):
            means[j] = tuple(self._interpolate(vertex, tail, coordinates[j])
                             for vertex, tail in zip(self.relative[j], means[j+1], strict=True))
            linear[j] = self._interpolate(self.coefficients[j], linear[j+1], coordinates[j])
            complement = (a.add(Fraction(1), -coordinates[j][1]), a.add(Fraction(1), -coordinates[j][0]))
            value = _finite(self._binary_entropy(coordinates[j])+_interval(complement)*entropy[j+1])
            # An entropy of n normalized weights is in [-log(n),0].
            entropy[j] = Interval(max(value.lo, -self.logs[modes-j-1].hi), min(0.0, value.hi))
        return means, linear, entropy

    def _middle(self, box: Box) -> tuple[Vector, Vector]:
        a = self.arithmetic
        coordinates = tuple(a._coefficient(a.add(lower, upper)/2) for lower, upper in self._coordinates(box))
        mean = self.relative[-1]
        for j in reversed(range(self.model.mode_count-1)):
            mean = tuple(a.add(value, a.mul(coordinates[j], a.add(vertex, -value)))
                         for vertex, value in zip(self.relative[j], mean, strict=True))
        w = tuple(a.add(shift, value) for shift, value in zip(self.shift, mean, strict=True))
        source = tuple(a.add(origin, value) for origin, value in zip(self.projection.origin, w, strict=True))
        solved = _project_exact(self.projector, source, a)
        if solved is None:
            raise ArithmeticError("no feasible exact entropy projection")
        r = tuple(a.add(value, -origin) for value, origin in zip(solved[0], self.projection.origin, strict=True))
        return w, r

    def _quadratic(self, box: Box, mean: Bounds) -> Pair:
        """Stable enclosure of H(w)=dist²(c+w,C)-||w||²."""
        a = self.arithmetic
        bounds = tuple((a.add(shift, lower), a.add(shift, upper))
                       for shift, (lower, upper) in zip(self.shift, mean, strict=True))
        if not self.projector.inequalities:
            lower = upper = Fraction(0)
            for left, right in bounds:
                squares = a.mul(left, left), a.mul(right, right)
                lower = a.add(lower, -max(squares))
                upper = a.add(upper, -(Fraction(0) if left <= 0 <= right else min(squares)))
            return lower, upper
        w0, r0 = self._middle(box)
        h0 = a.add(_dot(r0, r0, a), -a.mul(Fraction(2), _dot(w0, r0, a)))
        lower = upper = h0
        squared_radius = Fraction(0)
        for (left, right), center, projected in zip(bounds, w0, r0, strict=True):
            delta = a.add(left, -center), a.add(right, -center)
            slope = a.mul(Fraction(-2), projected)
            values = a.mul(slope, delta[0]), a.mul(slope, delta[1])
            lower, upper = a.add(lower, min(values)), a.add(upper, max(values))
            deviation = max(abs(delta[0]), abs(delta[1]))
            squared_radius = a.add(squared_radius, a.mul(deviation, deviation))
        # Projection optimality gives T-||w-w0||² <= H(w) <= T,
        # T=H0-2*r0.(w-w0), without differentiating projection.
        return a.add(lower, -squared_radius), upper

    def value(self, box: Box) -> Interval:
        a = self.arithmetic
        a._work(128*(self.model.mode_count+1)*(self.model.dimension+1))
        means, linear, entropy = self._tails(box)
        h = self._quadratic(box, means[0])
        denominator = a.mul(Fraction(2), self.model.time)
        exact = (a.add(a._coefficient(h[0]/denominator), linear[0][0]),
                 a.add(a._coefficient(h[1]/denominator), linear[0][1]))
        return _finite(_interval(exact)+_rational(self.scale)*entropy[0])

    def gradient(self, box: Box) -> Box:
        if self.model.mode_count == 1:
            return (_ZERO,)
        if not self.domain.gradient_safe or any(axis.lo <= 0 or axis.hi >= 1 for axis in box):
            raise ValueError("entropy gradient requires a certified strict interior domain")
        a = self.arithmetic
        a._work(192*(self.model.mode_count+1)*(self.model.dimension+1))
        means, linear, entropy = self._tails(box)
        coordinates = self._coordinates(box)
        projected = self.projection.image(box)
        centered = tuple((a.add(lower, -origin), a.add(upper, -origin))
                         for (lower, upper), origin in zip(projected, self.projection.origin, strict=True))
        prefix: Pair = (Fraction(1), Fraction(1))
        result = []
        for j in range(self.model.mode_count-1):
            dot_lower = dot_upper = Fraction(0)
            for (rlo, rhi), vertex, (tlo, thi) in zip(centered, self.relative[j], means[j+1], strict=True):
                difference = a.add(vertex, -thi), a.add(vertex, -tlo)
                products = tuple(a.mul(r, delta) for r in (rlo, rhi) for delta in difference)
                dot_lower, dot_upper = a.add(dot_lower, min(products)), a.add(dot_upper, max(products))
            exact = (a.add(a.add(self.coefficients[j], -linear[j+1][1]), -a._coefficient(dot_upper/self.model.time)),
                     a.add(a.add(self.coefficients[j], -linear[j+1][0]), -a._coefficient(dot_lower/self.model.time)))
            complement = a.add(Fraction(1), -coordinates[j][1]), a.add(Fraction(1), -coordinates[j][0])
            logarithms = _finite(ln_iv(_interval(coordinates[j]))-ln_iv(_interval(complement))-entropy[j+1])
            bracket = _finite(_interval(exact)+_rational(self.scale)*logarithms)
            result.append(_finite(_interval(prefix)*bracket))
            # Exact prefix products may be below the binary64 range. Only
            # their final conversion rounds; logarithms never see that product.
            prefix = a.mul(prefix[0], complement[0]), a.mul(prefix[1], complement[1])
        return tuple(result)

    def point(self, parameters: Sequence[float]) -> Vector:
        return self.projection.point(parameters)

    def physical_box(self, box: Box) -> Box:
        return self.projection.physical_box(box)


@dataclass(frozen=True, init=False)
class GaussianEntropyMinimum:
    """Global original-objective bounds from the equivalent Gibbs envelope.

    ``parameter_result`` and ``sample_entropy`` concern E(p); ``sample_value``
    encloses the original normalized F at the exactly feasible sample_point.
    E=F+2nu*KL implies F<=E and the same global minimum. The public upper bound
    uses the feasible original value; its gap can be tighter than the internal
    entropy gap. All physical global minima are covered through their unique
    posterior parameter representations, not every arbitrary projection preimage.

    ``gradient_used`` is false when posterior bounds touch a simplex boundary.
    Then continuous entropy values still give a complete value-only search.
    No derivative of the projection or selected global decision is asserted.
    """

    domain: GaussianPosteriorDomain
    minimum: Interval
    sample_point: Vector
    sample_value: Interval
    sample_entropy: Interval
    candidate_boxes: tuple[Box, ...]
    candidate_parameters: tuple[Box, ...]
    parameter_result: BoxMinimum
    gradient_used: bool
    status: MinimumStatus
    gap_upper: float
    value_tolerance: float
    max_boxes: int
    certificate: Cert

    def __init__(self) -> None:
        raise TypeError("use certify_gaussian_entropy_minimum")


def certify_gaussian_entropy_minimum(
    domain: GaussianPosteriorDomain, *, value_tolerance: float = 1e-6, max_boxes: int = 255,
) -> GaussianEntropyMinimum | None:
    """Bound every original global minimum using a certified posterior domain.

    The Gibbs envelope has the same global value as the original objective and
    a C1 interior gradient even when the polyhedral projection changes face.
    Strict certified posterior coordinates enable rigorous gradient pruning.
    Boundary-inclusive domains retain a continuous, value-only fallback.
    Entropy log products are evaluated through log(u) and log(1-u), preserving
    finite gradient bounds when tiny posterior products underflow numerically.

    Invalid types/domains raise. Damaged nested proofs, nonfinite arithmetic
    or exhausted joint budgets return None, never partial global coverage.
    Missing rigorous backend or prior conditional fallback raises RuntimeError.
    Requested tolerance is not guaranteed on budget/resolution termination.
    """
    if not isinstance(domain, GaussianPosteriorDomain):
        raise TypeError("domain must be a GaussianPosteriorDomain")
    if type(max_boxes) is not int:
        raise TypeError("max_boxes must be an integer excluding bool")
    if max_boxes < 1:
        raise ValueError("max_boxes must be positive")
    if isinstance(value_tolerance, bool) or not isinstance(value_tolerance, (int, float, Fraction)):
        raise TypeError("value_tolerance must be a finite nonnegative real")
    try:
        tolerance = float(value_tolerance)
    except OverflowError as exc:
        raise ValueError("value_tolerance must be finite and nonnegative") from exc
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("value_tolerance must be finite and nonnegative")
    model, projector = domain.model, domain.projector
    a = _Arithmetic(_combined_budget(model.budget, projector.budget))
    dimension, modes = model.dimension, model.mode_count
    try:
        source = 2*_domain_source_slots(domain)+256*(modes+1)*(dimension+1)+4096
        queries = 4*max_boxes+1
        callback_work = _query_work_bound(projector)+512*(modes+1)*(dimension+1)+1024
        a._entries(source+128*(max_boxes+1)*(dimension+max(1, modes-1)+1)+4096)
        a.limits._work(source+queries*callback_work+4096)
        a._work(source)
        a.exact(max_boxes)
        if not _domain_intact(domain):
            return None
        if libm_fallback_used():
            raise RuntimeError("Gaussian entropy minimization refuses prior conditional libm fallback history")
        with _strict():
            oracle = _EntropyOracle(domain, a)
            result = minimize_box(oracle.value, domain.parameter_box,
                                  gradient=oracle.gradient if domain.gradient_safe else None,
                                  value_tolerance=tolerance, max_boxes=max_boxes)
            _finite(result.minimum)
            sample = oracle.point(result.point)
            original = oracle.projection.objective(tuple((value, value) for value in sample))
            sample_value = Interval(original.lo, min(original.hi, result.point_value.hi))
            minimum = Interval(result.minimum.lo, sample_value.hi)
            boxes = tuple(oracle.physical_box(box) for box in result.candidates)
            gap = 0.0 if minimum.lo == minimum.hi else math.nextafter(minimum.hi-minimum.lo, math.inf)
            status: MinimumStatus = "value_tolerance" if gap <= tolerance else result.status
            from omnibias.core.proof.certificate import encode_interval, make_certificate

            certificate = make_certificate(
                claim="Global Gaussian minimum bounds and complete physical minimizer cover via the equivalent Gibbs entropy envelope.",
                payload={
                    "type": "gaussian_entropy_minimum", "domain_certificate": deepcopy(domain.certificate),
                    "minimum": encode_interval(minimum), "sample_point": [_pair(value) for value in sample],
                    "sample_value": encode_interval(sample_value), "sample_entropy": encode_interval(result.point_value),
                    "candidate_boxes": [[encode_interval(value) for value in box] for box in boxes],
                    "candidate_parameters": [[encode_interval(value) for value in box] for box in result.candidates],
                    "parameter_minimum": encode_interval(result.minimum), "parameter_point": [value.hex() for value in result.point],
                    "gradient_used": domain.gradient_safe, "status": status, "gap_upper": gap.hex(),
                    "value_tolerance": tolerance.hex(), "max_boxes": max_boxes, "evaluated_boxes": result.evaluated_boxes,
                    "value_evaluations": result.value_evaluations, "gradient_evaluations": result.gradient_evaluations,
                    "argument": "Let c=a_0, delta_j=a_j-c, w=s*v+sum(p_j*delta_j), and x_p=P_C(c+w). The Gibbs variational identity gives E(p)=[dist²(c+w,C)-|w|²+sum(p_j*|delta_j|²)]/(2s)+2nu*sum(p_j*(log(p_j)-(b_j-b_0)))=F(x_p)+2nu*KL(p||posterior(x_p)). Minimizing jointly over x and p proves min(E)=min(F) and every original global minimizer is paired with its unique positive posterior. The source posterior-domain certificate covers all those parameters. Stick-breaking tail recurrences enclose expectations and entropy continuously, including 0*log(0)=0. For H(w)=dist²(c+w,C)-|w|² and an exact midpoint projection r0=x0-c, projection optimality gives T-|w-w0|²<=H(w)<=T with T=|r0|²-2*w0.r0-2*r0.(w-w0). This yields stable rational quadratic bounds. In the simplex interior the envelope gradient is |delta_j|²/(2s)-delta_j.(x_p-c)/s+2nu*(log(p_j)+1-(b_j-b_0)), up to a common simplex-normal offset. Squared-distance differentiability supplies C1 regularity across projection switches without a projection derivative. The recursive chain rule uses normalized tail expectations, tail entropy, exact prefix products, and log(z_j)-log(1-z_j); no logarithm of an underflowed product is taken. Gradient interval pruning is used only on a strictly interior certified domain; otherwise continuous value-only search applies. All entropy-global parameter minima are retained, and their projected images cover every original global minimizer. The feasible original sample value can strengthen the upper bound because F(x_p)<=E(p).",
                    "normalization": "F(x)=|x-c|²/(2s)-v.(x-c)-2nu*logsumexp(delta_j.(x-c)/(2nu*s)+(b_j-b_0)-|delta_j|²/(4nu*s)), c=a_0. The heat normalization, common bias and target translation constants are omitted, as in the projection-image optimizer.",
                    "scope": "Fixed finite rational Gaussian data and exact polyhedral constraints. No selected-optimizer derivative, uniqueness, guaranteed requested accuracy or universal speed, arbitrary PDE data, or external formal theorem-prover verification claim. Interval bounds and combinatorial projection work may remain costly or inconclusive.",
                }, honesty={"unproven_claim": False, "unconditional_transcendentals": True, "continuum_pde_claim": False},
                meta={"transcend_backend": "mpmath"},
            )
        output = object.__new__(GaussianEntropyMinimum)
        for name, value in (("domain", domain), ("minimum", minimum), ("sample_point", sample),
                            ("sample_value", sample_value), ("sample_entropy", result.point_value),
                            ("candidate_boxes", boxes), ("candidate_parameters", result.candidates), ("parameter_result", result),
                            ("gradient_used", domain.gradient_safe), ("status", status), ("gap_upper", gap),
                            ("value_tolerance", tolerance), ("max_boxes", max_boxes), ("certificate", certificate)):
            object.__setattr__(output, name, value)
        return output
    except (ArithmeticError, ValueError):
        return None


__all__ = ["GaussianEntropyMinimum", "certify_gaussian_entropy_minimum"]
