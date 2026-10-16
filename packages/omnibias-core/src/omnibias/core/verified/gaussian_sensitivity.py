# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Certify proposed target gradients of a Gaussian heat inverse decision."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

from omnibias.core.verified.gaussian_heat import GaussianHeatMap, _finite, _float, _rational
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.inverse_gaussian_heat import (
    GaussianInverseCertificate,
    _inputs,
    _pair,
    _sqrt_upper,
    certify_gaussian_inverse,
)
from omnibias.core.verified.transcend import backend_name

if TYPE_CHECKING:
    from omnibias.core.proof.certificate import Cert


@dataclass(frozen=True)
class GaussianTargetVJPCertificate:
    """Enclosure of ``(D_target x*)^T @ upstream`` for the exact inverse.

    ``proposal`` is a binary64 vector, not an asserted exact gradient. The
    Euclidean ``error_bound`` accounts for both the root location and the
    residual of the proposed adjoint solve. The matrix is symmetric, so a
    target JVP obeys the same statement. No tensor backend is imported.
    """

    root_certificate: GaussianInverseCertificate
    proposal: tuple[float, ...]
    enclosure: tuple[Interval, ...]
    error_bound: float
    residual_bound: float
    jacobian_lipschitz_bound: float
    certificate: Cert


def _norm_squared(model: GaussianHeatMap, values: Sequence[Interval]) -> Fraction:
    result = Fraction(0)
    for value in values:
        magnitude = max(abs(Fraction(value.lo)), abs(Fraction(value.hi)))
        square = model.budget._coefficient(magnitude * magnitude)
        result = model.budget._coefficient(result + square)
    return result


def certify_gaussian_target_vjp(
    model: GaussianHeatMap, target: Sequence[int | Fraction], scaled_point: Sequence[float],
    upstream: Sequence[float], proposal: Sequence[float],
) -> GaussianTargetVJPCertificate | None:
    r"""Certify any proposed target VJP using root and adjoint residuals.

    Inputs ``scaled_point``, ``upstream``, and ``proposal`` are finite binary64
    vectors (exactly representable integers also accepted); target/model data
    are exact rationals. The root proposal uses y=x/s, and the gradient proposal
    is for the physical x. The upstream vector is held constant in this VJP.

    Let H=I-Cov(a)/(2*nu*s), q=D^2/(8*nu*s)<1, and epsilon be the certified
    physical root error. A rigorous bound eta on ``H(s*y) @ proposal-s*upstream``
    proves error at most ``(eta+L*epsilon*||proposal||_2)/(1-q)``, with
    ``L=D^3/(16*nu^2*s^2)``. All norms and constants round outward; a zero or
    poor proposal merely produces a looser bound. No dense Jacobian is formed.

    Returns None for an inapplicable contraction bound, arithmetic failure or
    exhausted exact budgets. Invalid inputs raise. A missing rigorous backend
    raises rather than issuing a conditional certificate. This is a numerical
    certificate backed by the stated analytic proof, not a Lean-kernel result.
    """
    frozen_target, point = _inputs(model, target, scaled_point)
    if not isinstance(upstream, Sequence) or not isinstance(proposal, Sequence):
        raise TypeError("upstream and proposal must be finite sequences")
    if len(upstream) != model.dimension or len(proposal) != model.dimension:
        raise ValueError("upstream and proposal dimensions must match the model")
    direction = tuple(_float(value) for value in upstream)
    proposed = tuple(_float(value) for value in proposal)
    if model.contraction_bound >= 1:
        return None
    checked = model.budget._coefficient
    try:
        # One root proof, one matrix-free action, and scalar exact norm/bound
        # arithmetic. Constructor budgets have already bounded stored metadata.
        model.budget._work(model._evaluation_work() + 64 * (model.mode_count + 1) * (model.dimension + 1)
                           + 256 * model.dimension + 256)
        root = certify_gaussian_inverse(model, frozen_target, point)
        if root is None:
            return None
        proposal_box = tuple(Interval.point(value) for value in proposed)
        action = model.scaled_jacobian_action(tuple(Interval.point(value) for value in point), proposal_box)
        residual = tuple(_finite(value - _rational(checked(model.time * Fraction(component))))
                         for value, component in zip(action, direction, strict=True))
        residual_squared = _norm_squared(model, residual)
        proposal_squared = _norm_squared(model, proposal_box)
        eta = _sqrt_upper(residual_squared, model.budget)
        proposal_norm = _sqrt_upper(proposal_squared, model.budget)
        lipschitz_squared = Fraction(0)
        if model.diameter_squared:
            diameter_fourth = checked(model.diameter_squared * model.diameter_squared)
            diameter_sixth = checked(diameter_fourth * model.diameter_squared)
            nu_squared, time_squared = checked(model.viscosity * model.viscosity), checked(model.time * model.time)
            denominator = checked(checked(256 * checked(nu_squared * nu_squared)) * checked(time_squared * time_squared))
            lipschitz_squared = checked(diameter_sixth / denominator)
        lipschitz = _sqrt_upper(lipschitz_squared, model.budget)
        root_term = checked(checked(Fraction(lipschitz) * Fraction(root.error_bound)) * Fraction(proposal_norm))
        margin = checked(1 - model.contraction_bound)
        exact_bound = checked(checked(Fraction(eta) + root_term) / margin)
        error = _rational(exact_bound).hi
        enclosure = tuple(_finite(value + Interval(-error, error)) for value in proposal_box)
        from omnibias.core.proof.certificate import encode_interval, make_certificate

        certificate = make_certificate(
            claim="Enclosure of the target vector-Jacobian product of the unique global Gaussian heat inverse.",
            payload={
                "type": "gaussian_heat_target_vjp", "root_certificate": root.certificate,
                "upstream": list(direction), "proposal": list(proposed),
                "jacobian_action": [encode_interval(value) for value in action],
                "adjoint_residual": [encode_interval(value) for value in residual],
                "residual_norm_squared_bound": _pair(residual_squared),
                "proposal_norm_squared": _pair(proposal_squared),
                "jacobian_lipschitz_squared_bound": _pair(lipschitz_squared),
                "residual_bound": eta, "proposal_norm_bound": proposal_norm,
                "jacobian_lipschitz_bound": lipschitz, "error_bound": error,
                "exact_error_bound": _pair(exact_bound),
                "enclosure": [encode_interval(value) for value in enclosure],
                "global_argument": "H has eigenvalues >=1-q. Its spatial derivative has norm <=D^3/(16*nu^2*s^2). The gradient error satisfies H(root)*(proposal-true_gradient)=adjoint_residual+(H(root)-H(center))*proposal.",
            },
            honesty={"unproven_claim": False, "unconditional_transcendentals": True, "continuum_pde_claim": False},
            meta={"transcend_backend": backend_name()},
        )
        return GaussianTargetVJPCertificate(root, proposed, enclosure, error, eta, lipschitz, certificate)
    except (ArithmeticError, ValueError):
        return None


__all__ = ["GaussianTargetVJPCertificate", "certify_gaussian_target_vjp"]
