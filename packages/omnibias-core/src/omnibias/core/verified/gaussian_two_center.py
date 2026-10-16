# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Certified global decisions for nonconvex two-center Gaussian potentials."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING, Literal

from omnibias.core.gaussian_jet import _Arithmetic
from omnibias.core.verified.gaussian_affine import _dot, _validate_vector
from omnibias.core.verified.gaussian_heat import GaussianHeatMap, _rational
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.inverse_gaussian_affine import _linear
from omnibias.core.verified.inverse_gaussian_heat import _pair
from omnibias.core.verified.transcend import (
    exp_iv,
    libm_fallback_used,
    require_rigorous_backend,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from omnibias.core.proof.certificate import Cert

Rational = int | Fraction
Vector = tuple[Fraction, ...]
Matrix = tuple[Vector, ...]
Box = tuple[Interval, ...]
Termination = Literal["tolerance", "precision", "steps", "budget"]


def _probe(
    point: Fraction, inverse_q: Fraction, magnitude: Fraction, arithmetic: _Arithmetic,
) -> tuple[int, dict[str, Any], bool]:
    """Return a proved sign of lambda*y-tanh(y)-W, or zero if unresolved."""
    arithmetic._work(256)
    linear = arithmetic.add(arithmetic.mul(inverse_q, point), -magnitude)
    evidence: dict[str, Any] = {"point": _pair(point), "linear": _pair(linear)}
    if linear <= 0:
        return -1, {**evidence, "kind": "linear_nonpositive"}, False
    if linear >= 1:
        return 1, {**evidence, "kind": "linear_at_least_one"}, False
    threshold = arithmetic._coefficient(arithmetic.add(Fraction(1), -linear)
                                        /arithmetic.add(Fraction(1), linear))
    # exp(2*y)>1+2*y for y>0. This exact sign proof also handles y beyond
    # binary64 range, without demanding an exponential or rounding it to zero.
    exponential_upper = arithmetic._coefficient(Fraction(1)/arithmetic.add(
        Fraction(1), arithmetic.mul(Fraction(2), point),
    ))
    if exponential_upper <= threshold:
        return -1, {**evidence, "kind": "rational_exponential_upper", "threshold": _pair(threshold),
                    "exp_upper_strict": _pair(exponential_upper)}, False
    argument = arithmetic.mul(Fraction(-2), point)
    # Do not enter certificate_mode: its fresh-scope convenience clears history.
    require_rigorous_backend()
    if libm_fallback_used():
        raise RuntimeError("two-center refinement refuses prior conditional libm fallback history")
    exponential = exp_iv(_rational(argument))
    evidence.update(kind="exponential_threshold", threshold=_pair(threshold),
                    exp_lower=exponential.lo.hex(), exp_upper=exponential.hi.hex())
    # Comparing binary64 endpoints as EXACT rationals avoids rounding the
    # threshold, and avoids subtraction of almost equal tanh/linear values.
    upper = arithmetic._coefficient(Fraction(exponential.hi))
    lower = arithmetic._coefficient(Fraction(exponential.lo))
    sign = -1 if upper < threshold else 1 if lower > threshold else 0
    return sign, evidence, True


def _refine(
    inverse_q: Fraction, magnitude: Fraction, tolerance: Fraction, max_steps: int,
    dimension: int, arithmetic: _Arithmetic,
) -> tuple[tuple[Fraction, Fraction], int, Termination, dict[str, Any], dict[str, Any], bool]:
    lower = arithmetic.add(Fraction(1), -inverse_q)
    upper = arithmetic._coefficient(arithmetic.add(magnitude, Fraction(1))/inverse_q)
    low_proof: dict[str, Any] = {"kind": "analytic_lower", "point": _pair(lower)}
    high_proof: dict[str, Any] = {"kind": "analytic_upper", "point": _pair(upper)}
    iterations, used_transcendentals = 0, False
    # Each probe uses fewer than 512 work units. Reserve enough for the final
    # two signed vector lifts; failures consume the same cumulative meter.
    reserve = 128*(dimension+1)**2 + 512
    while True:
        try:
            width = arithmetic.add(upper, -lower)
        except ValueError:
            return (lower, upper), iterations, "budget", low_proof, high_proof, used_transcendentals
        if width <= tolerance:
            return (lower, upper), iterations, "tolerance", low_proof, high_proof, used_transcendentals
        if iterations >= max_steps:
            return (lower, upper), iterations, "steps", low_proof, high_proof, used_transcendentals
        if arithmetic.used+512+reserve > arithmetic.limits.max_elimination_work:
            return (lower, upper), iterations, "budget", low_proof, high_proof, used_transcendentals
        iterations += 1
        try:
            middle = arithmetic._coefficient(arithmetic.add(lower, upper)/2)
            sign, evidence, used = _probe(middle, inverse_q, magnitude, arithmetic)
            used_transcendentals = used_transcendentals or used
        except ValueError:
            return (lower, upper), iterations, "budget", low_proof, high_proof, used_transcendentals
        except ArithmeticError:
            return (lower, upper), iterations, "precision", low_proof, high_proof, used_transcendentals
        if sign < 0:
            lower, low_proof = middle, evidence
        elif sign > 0:
            upper, high_proof = middle, evidence
        else:
            return (lower, upper), iterations, "precision", low_proof, high_proof, used_transcendentals


@dataclass(frozen=True, init=False)
class GaussianTwoCenterMinimum:
    r"""All global minimizers of a target-tilted two-center Gaussian potential.

    For centers c-a,c+a, alpha=|a|² and q=alpha/(2*nu*s)>1, let lambda=1/q,
    delta=b_plus-b_minus, P=I-aa^T/alpha and w=s*(a.target+nu*delta)/alpha.
    The positive interval encloses the unique positive solution of
    lambda*y-tanh(y)=abs(w). If w!=0 its sign selects the UNIQUE global
    minimum; if w=0 both signs give exactly the two global minima. The lift
    is x=c+s*P*target+lambda*a*(y-delta/2). Other stationary points may exist.

    ``minimizer_boxes`` follows ``branch_signs`` order. ``termination`` says
    whether the scalar positive-root width reached ``tolerance``, midpoint
    signs were unresolved at binary64 enclosure precision, ``max_steps``
    ran out, or the exact resource budget stopped refinement. Every returned
    box remains certified in every case. The tolerance is not a vector-box
    width promise. A certificate digest records integrity, not formal proof.
    """

    model: GaussianHeatMap
    target: Vector
    midpoint: Vector
    half_difference: Vector
    squared_half_distance: Fraction
    orthogonal_projector: Matrix
    bias_contrast: Fraction
    inverse_q: Fraction
    selection_value: Fraction
    field: Fraction
    branch_signs: tuple[int, ...]
    positive_root_bounds: tuple[Fraction, Fraction]
    minimizer_boxes: tuple[Box, ...]
    tolerance: Fraction
    max_steps: int
    iterations: int
    termination: Termination
    certificate: Cert

    def __init__(self) -> None:
        raise TypeError("use certify_gaussian_two_center_minimum")

    @property
    def unique(self) -> bool:
        """Whether the original potential has exactly one global minimizer."""
        return len(self.branch_signs) == 1


def certify_gaussian_two_center_minimum(
    model: GaussianHeatMap, target: Sequence[Rational], *,
    tolerance: Rational = Fraction(1, 10**12), max_steps: int = 256,
) -> GaussianTwoCenterMinimum | None:
    r"""Certify all global minimizers for exactly two original centers, q>1.

    Exact sign classification and analytic initial root brackets require no
    transcendental backend. ``max_steps=0`` keeps those brackets. Refinement
    uses rigorous mpmath-backed interval exponentials; it raises RuntimeError
    when that backend is absent or conditional libm history was used. It
    never resets fallback history, and shared strict sealing still applies.

    Input type/shape/domain errors raise. Unsupported model count or q<=1,
    exhausted construction/output budgets and nonfinite output boxes return
    None. A stopped refinement can still return a valid coarse certificate;
    check ``termination`` and bounds before relying on a desired accuracy.
    The result does not claim a unique inverse of the velocity map, a smooth
    optimizer across the tie hyperplane, or a solution for arbitrary mixtures.
    """
    if not isinstance(model, GaussianHeatMap):
        raise TypeError("model must be a GaussianHeatMap")
    _validate_vector(target, model.dimension, "target")
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, Fraction)):
        raise TypeError("tolerance must be exact int or Fraction, excluding bool")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    if type(max_steps) is not int:
        raise TypeError("max_steps must be an integer, excluding bool")
    if max_steps < 0:
        raise ValueError("max_steps must be nonnegative")
    if model.mode_count != 2 or model.contraction_bound <= 1:
        return None
    arithmetic = _Arithmetic(model.budget)
    dimension = model.dimension
    try:
        arithmetic._entries(64*(dimension+1)**2+1024)
        arithmetic._work(128*(dimension+1)**2+1024)
        arithmetic.exact(max_steps)
        goal = tuple(arithmetic.exact(value) for value in target)
        tol = arithmetic.exact(tolerance)
        direction = tuple(arithmetic._coefficient(arithmetic.add(plus, -minus)/2)
                          for minus, plus in zip(model.centers[0], model.centers[1], strict=True))
        alpha = _dot(direction, direction, arithmetic)
        inverse_q = arithmetic._coefficient(arithmetic.mul(arithmetic.mul(Fraction(2), model.viscosity), model.time)/alpha)
        if not 0 < inverse_q < 1:
            return None
        midpoint = tuple(arithmetic.add(origin, delta) for origin, delta in zip(model.centers[0], direction, strict=True))
        bias = arithmetic.add(model.biases[1], -model.biases[0])
        selection = arithmetic.add(_dot(direction, goal, arithmetic), arithmetic.mul(model.viscosity, bias))
        field = arithmetic._coefficient(arithmetic.mul(model.time, selection)/alpha)
        signs = (-1, 1) if field == 0 else (1,) if field > 0 else (-1,)
        projector = tuple(tuple(arithmetic.add(Fraction(int(i == j)),
                               -arithmetic._coefficient(arithmetic.mul(direction[i], direction[j])/alpha))
                               for j in range(dimension)) for i in range(dimension))
        transverse = _linear(projector, goal, arithmetic)
        scale = tuple(arithmetic.mul(inverse_q, value) for value in direction)
        half_bias = arithmetic._coefficient(bias/2)
        anchor = tuple(arithmetic.add(arithmetic.add(center, arithmetic.mul(model.time, normal)),
                                      -arithmetic.mul(factor, half_bias))
                       for center, normal, factor in zip(midpoint, transverse, scale, strict=True))
        bounds, iterations, termination, low_proof, high_proof, used = _refine(
            inverse_q, abs(field), tol, max_steps, dimension, arithmetic,
        )
        boxes: list[Box] = []
        for sign in signs:
            rows: list[Interval] = []
            for value, factor in zip(anchor, scale, strict=True):
                signed = arithmetic.mul(Fraction(sign), factor)
                endpoints = tuple(arithmetic.add(value, arithmetic.mul(signed, bound)) for bound in bounds)
                rows.append(Interval(_rational(min(endpoints)).lo, _rational(max(endpoints)).hi))
            boxes.append(tuple(rows))
        from omnibias.core.proof.certificate import encode_interval, make_certificate

        certificate = make_certificate(
            claim="All global minimizers of the original target-tilted nonconvex two-center Gaussian potential.",
            payload={
                "type": "gaussian_two_center_minimum",
                "centers": [[_pair(value) for value in row] for row in model.centers],
                "biases": [_pair(value) for value in model.biases], "viscosity": _pair(model.viscosity), "time": _pair(model.time),
                "target": [_pair(value) for value in goal], "midpoint": [_pair(value) for value in midpoint],
                "half_difference": [_pair(value) for value in direction], "squared_half_distance": _pair(alpha),
                "orthogonal_projector": [[_pair(value) for value in row] for row in projector],
                "bias_contrast": _pair(bias), "inverse_q": _pair(inverse_q),
                "selection_value": _pair(selection), "field": _pair(field), "branch_signs": list(signs),
                "positive_root_bounds": [_pair(value) for value in bounds],
                "minimizer_boxes": [[encode_interval(value) for value in box] for box in boxes],
                "tolerance": _pair(tol), "max_steps": max_steps, "iterations": iterations, "termination": termination,
                "lower_sign_evidence": low_proof, "upper_sign_evidence": high_proof,
                "identities": ["lambda=1/q=2*nu*s/alpha; 0<lambda<1", "P=I-a*a^T/alpha",
                               "w=s*(a.target+nu*delta)/alpha", "lambda*y-tanh(y)=w",
                               "X=c+s*P*target+lambda*a*(y-delta/2)"],
                "argument": "The original target-tilted Gaussian potential equals |P*(x-c)-s*P*target|²/(2*s) + 2*nu*(lambda*y²/2-log(cosh(y))-w*y) plus a constant, where y=q*a.(x-c)/alpha+delta/2. The scalar potential is coercive. For w>0, reflection strictly favors every positive y over its negative, and y=0 is not stationary. On y>0, F'(y)=lambda-sech²(y) strictly increases from negative to positive, so F(y)=w has exactly one positive root. It is the unique global minimum. Symmetry gives w<0. At w=0 the origin is a strict local maximum and exactly two opposite outer roots are equal global minima. For W=|w|, l=1-lambda and h=(W+1)/lambda bound the positive winning root: tanh(l)>l-l³/3 implies F(l)-W<0, while F(h)-W=1-tanh(h)>0. Each retained midpoint sign is proved either by 0<tanh(y)<1 or by comparing exp(-2*y) with (1-L)/(1+L), L=lambda*y-W. The exact inequality exp(-2*y)<1/(1+2*y) can prove a negative sign before numerical evaluation. The exact affine lift gives all original minimizers. The exact sign of a.target+nu*delta selects their branch independently of heat age within q>1.",
                "scope": "Exactly two original centers and q>1, unconstrained real domain, fixed model and exact target. All global minimizers are enclosed, but other stationary points may exist. Precision, iteration or refinement-budget termination retains valid bounds without asserting tolerance completion. No differentiable selection across the two-minimum tie, arbitrary-mixture, or formal theorem-prover verification claim.",
            }, honesty={"unproven_claim": False, "unconditional_transcendentals": used, "continuum_pde_claim": False},
            meta={"transcend_backend": "mpmath" if used else "not_used"},
        )
        result = object.__new__(GaussianTwoCenterMinimum)
        for name, stored in (("model", model), ("target", goal), ("midpoint", midpoint), ("half_difference", direction),
                            ("squared_half_distance", alpha), ("orthogonal_projector", projector), ("bias_contrast", bias),
                            ("inverse_q", inverse_q), ("selection_value", selection), ("field", field), ("branch_signs", signs),
                            ("positive_root_bounds", bounds), ("minimizer_boxes", tuple(boxes)), ("tolerance", tol),
                            ("max_steps", max_steps), ("iterations", iterations), ("termination", termination), ("certificate", certificate)):
            object.__setattr__(result, name, stored)
        return result
    except (ArithmeticError, ValueError):
        return None


__all__ = ["GaussianTwoCenterMinimum", "certify_gaussian_two_center_minimum"]
