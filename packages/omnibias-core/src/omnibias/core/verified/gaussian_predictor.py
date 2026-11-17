# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""One-pass Gaussian inverse predictors with uniform rigorous error bounds."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

from omnibias.core.gaussian_jet import _Arithmetic
from omnibias.core.verified._gaussian_moments import _add, _multiply, _posterior_moments
from omnibias.core.verified.gaussian_heat import GaussianHeatMap, _finite, _rational
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.inverse_gaussian_heat import _pair, _sqrt_upper
from omnibias.core.verified.mean_map import _probabilities, _strict

if TYPE_CHECKING:
    from omnibias.core.proof.certificate import Cert

Box = tuple[Interval, ...]


@dataclass(frozen=True, init=False)
class GaussianInversePredictor:
    r"""Uniform inverse enclosure from one limiting-posterior evaluation.

    ``predictor_enclosure`` encloses the explicit predictor throughout the
    target box, including interval rounding. ``enclosure`` additionally
    includes the analytic approximation error and intersects the inverse's
    global center hull. ``approximation_error_bound`` is a Euclidean error
    between the exact predictor and true inverse at the same target, not the
    radius of either returned box. Bounds do not certify predictor derivatives.

    ``relative_mean`` encloses E_infinity[a-a0]; ``third_trace`` encloses
    E_infinity[(a-mu_infinity)*|a-mu_infinity|²]. All model parameters are fixed.
    """

    model: GaussianHeatMap
    target_box: Box
    order: int
    relative_mean: Box
    third_trace: Box
    predictor_enclosure: Box
    approximation_error_bound: Fraction
    enclosure: Box
    certificate: Cert

    def __init__(self) -> None:
        raise TypeError("use certify_gaussian_inverse_predictor")


def certify_gaussian_inverse_predictor(
    model: GaussianHeatMap, target_box: Sequence[Interval], *, order: int = 1,
) -> GaussianInversePredictor | None:
    r"""Enclose a Gaussian inverse without root iteration or a linear solve.

    Require q=D²/(8*nu*s)<1 for a unique global inverse. Define limiting
    probabilities p_infinity proportional to exp(b_j+a_j.v/(2*nu)), their
    mean mu and third trace T=E[(a-mu)*|a-mu|²]. The order-zero predictor
    s*v+mu has uniform Euclidean error <=D*q/2. The order-one predictor
    s*v+mu-T/(4*nu*s) has error <=D*q². The latter bound is smaller only
    when q<1/2. Both hold for every real target; interval evaluation encloses
    them throughout the supplied finite target box. These are value bounds,
    not derivative bounds or guarantees for a separate tensor evaluation.

    Exact center differences and common-bias subtraction precede binary
    conversion. Affine terms are combined exactly before interval conversion.
    The stored model already knows its diameter; each query costs O(m*d)
    arithmetic and storage, without a covariance matrix or inverse iteration.

    Invalid types/domains raise. Noncontractive models, exhausted budgets and
    nonfinite enclosures return None. Missing rigorous transcendental support
    raises except for identical centers, which need only rational arithmetic.
    """
    if not isinstance(model, GaussianHeatMap):
        raise TypeError("model must be a GaussianHeatMap")
    if not isinstance(target_box, Sequence):
        raise TypeError("target_box must be a finite sequence of Interval objects")
    box = model._box(target_box)
    if type(order) is not int:
        raise TypeError("order must be an integer, excluding bool")
    if order not in (0, 1):
        raise ValueError("order must be zero or one")
    q = model.contraction_bound
    if q >= 1:
        return None
    try:
        arithmetic = _Arithmetic(model.budget)
        checked = arithmetic._coefficient
        count, dimension = model.mode_count, model.dimension
        arithmetic._entries(32 * (count + 1) * (dimension + 1) + 64)
        arithmetic._work(256 * (count + 1) * (dimension + 1) + 64)
        zero = Interval.point(0.0)
        posterior: Box = ()
        mean: Box = (zero,) * dimension
        third_trace: Box = mean
        error = Fraction(0)
        if q:
            relative = tuple(tuple(arithmetic.add(value, -reference) for value, reference in zip(row, model.centers[0], strict=True))
                             for row in model.centers)
            converted = tuple(tuple(_rational(value) for value in row) for row in relative)
            twice_nu = arithmetic.mul(Fraction(2), model.viscosity)
            logits = []
            for row, bias in zip(relative, model.biases, strict=True):
                logit = _rational(arithmetic.add(bias, -model.biases[0]))
                for value, target in zip(row, box, strict=True):
                    logit = _add(logit, _multiply(_rational(checked(value / twice_nu)), target))
                logits.append(logit)
            with _strict():
                posterior = _probabilities(tuple(logits))
                moments = _posterior_moments(posterior, converted, covariance=False)
            mean, third_trace = moments.relative_mean, moments.third_trace
            diameter = checked(Fraction(_sqrt_upper(model.diameter_squared, model.budget)))
            error = checked(arithmetic.mul(diameter, q) / 2) if order == 0 else arithmetic.mul(diameter, arithmetic.mul(q, q))
        scale = _rational(checked(1 / arithmetic.mul(arithmetic.mul(Fraction(4), model.viscosity), model.time))) if q and order else zero
        error_upper = _rational(error).hi
        predictor = []
        enclosed = []
        for axis, (target, reference) in enumerate(zip(box, model.centers[0], strict=True)):
            # Positive age makes endpoint-affine evaluation exact before the
            # binary interval boundary, preserving large common translations.
            low_affine = arithmetic.add(arithmetic.mul(model.time, Fraction(target.lo)), reference)
            high_affine = arithmetic.add(arithmetic.mul(model.time, Fraction(target.hi)), reference)
            affine = Interval(_rational(low_affine).lo, _rational(high_affine).hi)
            prediction = _add(affine, mean[axis])
            if q and order:
                prediction = _finite(prediction - _multiply(scale, third_trace[axis]))
            predictor.append(prediction)
            expanded = _add(prediction, Interval(-error_upper, error_upper))
            low_hull = arithmetic.add(arithmetic.mul(model.time, Fraction(target.lo)), min(a[axis] for a in model.centers))
            high_hull = arithmetic.add(arithmetic.mul(model.time, Fraction(target.hi)), max(a[axis] for a in model.centers))
            enclosed.append(_finite(Interval(max(expanded.lo, _rational(low_hull).lo),
                                             min(expanded.hi, _rational(high_hull).hi))))
        from omnibias.core.proof.certificate import encode_interval, make_certificate

        certificate = make_certificate(
            claim="Uniform Gaussian inverse enclosure from a limiting-posterior moment predictor.",
            payload={
                "type": "gaussian_inverse_predictor", "centers": [[_pair(value) for value in row] for row in model.centers],
                "biases": [_pair(value) for value in model.biases], "viscosity": _pair(model.viscosity), "time": _pair(model.time),
                "target_box": [encode_interval(value) for value in box], "order": order,
                "diameter_squared": _pair(model.diameter_squared), "contraction_bound": _pair(q),
                "posterior": [encode_interval(value) for value in posterior],
                "relative_mean": [encode_interval(value) for value in mean], "third_trace": [encode_interval(value) for value in third_trace],
                "predictor_enclosure": [encode_interval(value) for value in predictor],
                "approximation_error_bound": _pair(error), "enclosure": [encode_interval(value) for value in enclosed],
                "predictor": "s*v+mu_infinity for order zero; s*v+mu_infinity-T_infinity/(4*nu*s) for order one",
                "argument": "At an inverse point m*=X-s*v lies in the center hull. Its exact posterior is the limiting posterior tilted by -theta*|a-m*|^2/(4*nu*s) at theta=1. Freeze m*. The tilt-mean derivative is a covariance whose vector norm is <=D^3/(16*nu*s), proving D*q/2. Its second derivative has norm <=D^5/(64*nu^2*s^2), so the first-order Taylor remainder is <=D*q^2/2. Replacing m* by the limiting mean in the linear correction adds <=q*(D*q/2). The sum is D*q^2. Both estimates are uniform over every real target. Rigorous limiting-posterior intervals and exact affine arithmetic supply the reported enclosure; the center hull is an independent intersection.",
                "error_scope": "Euclidean analytic error at equal targets plus outward interval evaluation on target_box; no derivative or separate tensor-roundoff claim.",
            },
            honesty={"unproven_claim": False, "unconditional_transcendentals": q != 0, "continuum_pde_claim": False},
            meta={"transcend_backend": "not_used" if q == 0 else "mpmath"},
        )
        result = object.__new__(GaussianInversePredictor)
        for name, stored in (("model", model), ("target_box", box), ("order", order), ("relative_mean", mean),
                             ("third_trace", third_trace), ("predictor_enclosure", tuple(predictor)),
                             ("approximation_error_bound", error), ("enclosure", tuple(enclosed)), ("certificate", certificate)):
            object.__setattr__(result, name, stored)
        return result
    except (ArithmeticError, ValueError):
        return None


__all__ = ["GaussianInversePredictor", "certify_gaussian_inverse_predictor"]
