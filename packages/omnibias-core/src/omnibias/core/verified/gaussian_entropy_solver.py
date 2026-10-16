# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Projected posterior-mean proposals with certified global residual bounds."""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING, Literal

from omnibias.core.gaussian_jet import _Arithmetic
from omnibias.core.verified.gaussian_affine import _validate_vector
from omnibias.core.verified.gaussian_entropy_residual import (
    GaussianEntropyResidual,
    _residual_source_slots,
    _residual_work_bound,
    _ResidualOracle,
    _seal_residual,
)
from omnibias.core.verified.gaussian_entropy_unique import GaussianEntropyUnique, _unique_intact
from omnibias.core.verified.gaussian_projection_minimum import _combined_budget
from omnibias.core.verified.inverse_gaussian_heat import _pair
from omnibias.core.verified.mean_map import _strict
from omnibias.core.verified.transcend import libm_fallback_used

if TYPE_CHECKING:
    from collections.abc import Sequence

    from omnibias.core.proof.certificate import Cert
    from omnibias.core.verified.interval import Interval

Rational = int | Fraction
Vector = tuple[Fraction, ...]
GaussianEntropySolveStatus = Literal["tolerance", "iteration_budget", "stagnation"]


def _posterior_proposal(posterior: tuple[Interval, ...], arithmetic: _Arithmetic) -> Vector:
    """Normalize exact interval midpoints; these are proposals, not true weights."""
    values = tuple(arithmetic._coefficient(arithmetic.add(arithmetic.exact(Fraction(value.lo)),
                                                        arithmetic.exact(Fraction(value.hi)))/2)
                   for value in posterior)
    total = arithmetic.sum(values)
    if total <= 0 or any(value < 0 for value in values):
        raise ArithmeticError("posterior enclosures cannot produce a simplex proposal")
    return tuple(arithmetic._coefficient(value/total) for value in values)


@dataclass(frozen=True, init=False)
class GaussianEntropySolve:
    r"""A bounded projected iteration and one rigorous GLOBAL optimizer enclosure.

    ``residual`` certifies the best evaluated candidate, selected by its exact
    residual squared bound. ``selected_iteration`` is its zero-based proposal
    update count. ``iterations`` counts completed updates and ``evaluations``
    counts their evaluations, including the initial candidate. ``error_bounds``
    records the certified spatial bound at each evaluated candidate in order;
    these diagnostics need not be monotone.

    ``status='tolerance'`` means residual.error_bound <= tolerance. The other
    statuses still retain a rigorous global location and value certificate,
    but do not assert the requested spatial accuracy. Tolerance concerns the
    Euclidean distance of the exact rational point to the optimizer, not the
    width of its binary64 coordinate enclosure or the objective interval.

    The exact fixed-point map is a contraction on the source physical box.
    This implementation uses rational posterior proposals and certifies their
    errors separately; it does not assume finite-precision convergence or claim
    its rational weights equal the transcendental original posterior.
    """

    unique: GaussianEntropyUnique
    residual: GaussianEntropyResidual
    iterations: int
    evaluations: int
    selected_iteration: int
    error_bounds: tuple[float, ...]
    status: GaussianEntropySolveStatus
    tolerance: float
    max_iterations: int
    certificate: Cert

    def __init__(self) -> None:
        raise TypeError("use solve_gaussian_entropy")


def solve_gaussian_entropy(
    unique: GaussianEntropyUnique, *, initial_probabilities: Sequence[Rational] | None = None,
    tolerance: float = 1e-8, max_iterations: int = 32,
) -> GaussianEntropySolve | None:
    r"""Iterate projected posterior means, certifying each candidate globally.

    Initial probabilities must form an exact rational simplex vector; boundary
    weights and weights outside the source posterior polytope are valid. The
    default is uniform. Every candidate is an exact feasible projection of
    s*target plus its weighted center mean, and hence belongs to the certified
    physical box. True posterior interval midpoints, normalized exactly, supply
    the next proposal. No logarithm of a proposed probability is needed.

    At most max_iterations updates plus one initial evaluation are attempted.
    A strictly validated source covariance bound gives global distance/value
    bounds at every candidate. The best bound is retained. Exhaustion of the
    requested update count or unchanged proposal is a reported termination,
    whereas arithmetic/resource/proof failure returns None, not a partial run.
    All requested work is preflighted against both source budgets and metered
    cumulatively. Invalid controls or simplex metadata raise. Missing rigorous
    backend or prior conditional-libm history raises RuntimeError.
    """
    if not isinstance(unique, GaussianEntropyUnique):
        raise TypeError("unique must be a GaussianEntropyUnique")
    if type(max_iterations) is not int:
        raise TypeError("max_iterations must be an integer excluding bool")
    if max_iterations < 0:
        raise ValueError("max_iterations must be nonnegative")
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float, Fraction)):
        raise TypeError("tolerance must be a finite nonnegative real")
    try:
        tolerance_value = float(tolerance)
    except OverflowError as exc:
        raise ValueError("tolerance must be finite and nonnegative") from exc
    if not math.isfinite(tolerance_value) or tolerance_value < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    model, projector = unique.domain.model, unique.domain.projector
    if initial_probabilities is not None:
        _validate_vector(initial_probabilities, model.mode_count, "initial_probabilities")
        if any(value < 0 for value in initial_probabilities):
            raise ValueError("initial_probabilities must be nonnegative and sum to one")
    a = _Arithmetic(_combined_budget(model.budget, projector.budget))
    try:
        source = 2*_residual_source_slots(unique)+4096
        evaluations_limit = max_iterations+1
        a._entries(source+128*evaluations_limit+128*(model.mode_count+1)*(model.dimension+1))
        a.limits._work(4*source+evaluations_limit*(_residual_work_bound(unique)+128*(model.mode_count+1))+4096)
        a._work(4*source)
        a.exact(max_iterations)
        if initial_probabilities is None:
            probabilities = (a._coefficient(Fraction(1, model.mode_count)),)*model.mode_count
        else:
            probabilities = tuple(a.exact(value) for value in initial_probabilities)
        total = a.sum(probabilities)
    except (ArithmeticError, ValueError):
        return None
    if total != 1:
        raise ValueError("initial_probabilities must be nonnegative and sum to one")
    initial = probabilities
    try:
        if not _unique_intact(unique):
            return None
        from omnibias.core.proof.certificate import certificate_is_unconditional, make_certificate

        if not certificate_is_unconditional(unique.certificate):
            return None
        if libm_fallback_used():
            raise RuntimeError("Gaussian entropy solver refuses prior conditional libm fallback history")
        with _strict():
            oracle = _ResidualOracle(unique, a)
            best = None
            selected_iteration = 0
            errors: list[float] = []
            status: GaussianEntropySolveStatus
            for iteration in range(evaluations_limit):
                data = oracle.evaluate(probabilities)
                errors.append(data.error_bound)
                if best is None or data.residual_squared_bound < best.residual_squared_bound:
                    best, selected_iteration = data, iteration
                if best.error_bound <= tolerance_value:
                    status = "tolerance"
                    break
                if iteration == max_iterations:
                    status = "iteration_budget"
                    break
                proposed = _posterior_proposal(data.posterior, a)
                if proposed == probabilities:
                    status = "stagnation"
                    break
                probabilities = proposed
            else:  # The finite loop always breaks at max_iterations.
                raise ArithmeticError("projected iteration did not terminate")
            if best is None:
                raise ArithmeticError("projected iteration produced no candidate")
            residual = _seal_residual(unique, best)
            certificate = make_certificate(
                claim="A projected posterior-mean iteration with certified global Gaussian optimizer distance and value bounds.",
                payload={
                    "type": "gaussian_entropy_solve", "residual_certificate": deepcopy(residual.certificate),
                    "initial_probabilities": [_pair(value) for value in initial],
                    "iterations": iteration, "evaluations": len(errors), "selected_iteration": selected_iteration,
                    "error_bounds": [value.hex() for value in errors], "status": status,
                    "tolerance": tolerance_value.hex(), "max_iterations": max_iterations,
                    "argument": "The source exact physical box K contains P_C(s*v+conv(centers)), and the source posterior bounds enclose the true posterior throughout K. Its covariance bound gives rho=B/(2nu*s)<1, so T(x)=P_C(s*v+E[a](x)) is a contraction on K and T(R^d) is contained in K. Exact iteration therefore converges to the unique constrained stationary point/global optimizer. Computational proposals instead use exact rational normalized midpoints of the rigorous posterior intervals. They need not equal the true posterior or belong to the source posterior polytope. Their exact projected points lie in K and are feasible. Every evaluated candidate has a separately certified true normal-cone residual and hence global distance/value bounds; proposal convergence is not assumed. The selected candidate has the smallest exact residual squared bound among those evaluated. Tolerance is asserted only when its outward spatial error bound is at most the requested tolerance; iteration budget and proposal stagnation retain a rigorous certificate without asserting that accuracy.",
                    "scope": "Fixed source Gaussian model, constraints and target. Tolerance controls Euclidean point error, not objective-interval or coordinate-box width. No finite-precision convergence, requested accuracy at arbitrary resources, global physical convexity, selected-optimizer differentiability, universal speed or external formal theorem-prover verification is claimed.",
                }, honesty={"unproven_claim": False, "unconditional_transcendentals": True, "continuum_pde_claim": False},
                meta={"transcend_backend": "mpmath"},
            )
        result = object.__new__(GaussianEntropySolve)
        for name, value in (("unique", unique), ("residual", residual), ("iterations", iteration),
                            ("evaluations", len(errors)), ("selected_iteration", selected_iteration),
                            ("error_bounds", tuple(errors)), ("status", status), ("tolerance", tolerance_value),
                            ("max_iterations", max_iterations), ("certificate", certificate)):
            object.__setattr__(result, name, value)
        return result
    except (ArithmeticError, ValueError):
        return None


__all__ = ["GaussianEntropySolve", "solve_gaussian_entropy"]
