# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Joint log-weight and target derivatives of certified global Gaussian decisions."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

from omnibias.core.gaussian_jet import _Arithmetic
from omnibias.core.verified._gaussian_bias_response import (
    _bias_covariance,
    _interval_response,
    _projected_mean,
)
from omnibias.core.verified._gaussian_moments import _add, _multiply, _zero
from omnibias.core.verified.gaussian_affine import _validate_vector
from omnibias.core.verified.gaussian_entropy_sensitivity import (
    GaussianEntropySensitivity,
    _matrix_payload,
    _response,
    _sensitivity_intact,
)
from omnibias.core.verified.gaussian_entropy_sensitivity import (
    _source_slots as _sensitivity_source_slots,
)
from omnibias.core.verified.gaussian_heat import _rational
from omnibias.core.verified.gaussian_projection_minimum import _combined_budget
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.inverse_gaussian_affine import _intact, _linear
from omnibias.core.verified.inverse_gaussian_heat import _pair
from omnibias.core.verified.mean_map import _strict
from omnibias.core.verified.transcend import libm_fallback_used

if TYPE_CHECKING:
    from collections.abc import Sequence

    from omnibias.core.proof.certificate import Cert

Rational = int | Fraction
Vector = tuple[Fraction, ...]
Matrix = tuple[Vector, ...]
Box = tuple[Interval, ...]
IntervalMatrix = tuple[Box, ...]
_ZERO = Interval.point(0.0)


def _source_slots(sensitivity: GaussianEntropySensitivity) -> int:
    """Reserve the bound proof chain, its snapshot and new bias block scratch."""
    model = sensitivity.face.residual.unique.domain.model
    return (_sensitivity_source_slots(sensitivity.face)
            +512*(model.mode_count+1)*(model.dimension+1)+256*(model.dimension+1)**2+8192)


def _bias_intact(bias: GaussianEntropyBias) -> bool:
    from omnibias.core.proof.certificate import certificate_is_unconditional, encode_interval

    if (not _intact(bias.certificate) or not _sensitivity_intact(bias.sensitivity)
            or not certificate_is_unconditional(bias.certificate)):
        return False
    payload = bias.certificate.get("payload")
    return (isinstance(payload, dict) and payload.get("sensitivity_certificate") == bias.sensitivity.certificate
            and payload.get("projected_centers") == [[_pair(value) for value in row] for row in bias.projected_centers]
            and payload.get("projected_mean") == [encode_interval(value) for value in bias.projected_mean]
            and payload.get("bias_forcing") == _matrix_payload(bias.bias_forcing)
            and payload.get("bias_jacobian") == _matrix_payload(bias.bias_jacobian))


@dataclass(frozen=True, init=False)
class GaussianEntropyBias:
    r"""Bias and target derivatives of the exact stable global decision.

    Biases are the original Gaussian log mixture weights. Centers, age,
    viscosity and polyhedral geometry stay fixed. The source sensitivity's
    face and positive curvature give a smooth global decision branch in some
    joint bias/target neighborhood. No numerical radius is asserted.

    ``projected_centers`` are exact R*(a_j-a_0); ``projected_mean`` encloses
    their original posterior mean. The dimension-by-mode ``bias_forcing``
    encloses pi_j*(R*a_j-R*mu)/s and ``bias_jacobian`` encloses J times these
    columns, where J is the source target Jacobian. The same true posterior
    underlies every factor; interval products safely drop its dependencies.

    Query directions and cotangents are exact integers/Fractions. Common
    bias directions and normal upstream cotangents cancel before conversion.
    Cached queries require no new solve, transcendental backend or seal and
    preserve strict/backend history. They still validate the entire bound
    proof and enforce the combined source resource budgets.
    """

    sensitivity: GaussianEntropySensitivity
    projected_centers: Matrix
    projected_mean: Box
    bias_forcing: IntervalMatrix
    bias_jacobian: IntervalMatrix
    certificate: Cert

    def __init__(self) -> None:
        raise TypeError("use certify_gaussian_entropy_bias")

    def _arithmetic(self) -> _Arithmetic:
        domain = self.sensitivity.face.residual.unique.domain
        model = domain.model
        a = _Arithmetic(_combined_budget(model.budget, domain.projector.budget))
        slots = _source_slots(self.sensitivity)
        a._entries(slots)
        a._work(6*slots+1024*(model.mode_count+1)*(model.dimension+1)
                +1024*(model.dimension+1)**2+4096)
        if not _bias_intact(self):
            raise ValueError("Gaussian entropy bias proof or nested dependency is damaged or conditional")
        return a

    def jvp(self, target_direction: Sequence[Rational], bias_direction: Sequence[Rational]) -> Box:
        r"""Enclose J*(xi+Cov(a,beta)/s) for exact joint parameter directions.

        The bias direction is centered exactly before interval covariance;
        common shifts give exactly zero forcing. Original centers are already
        projected exactly, preserving structural tangent cancellations. Work
        after proof scans is O(m*d+d²). Invalid metadata, damaged proofs and
        exhausted resources raise, including on the zero-response branch.
        """
        source = self.sensitivity
        face, model = source.face, source.face.residual.unique.domain.model
        _validate_vector(target_direction, model.dimension, "target_direction")
        _validate_vector(bias_direction, model.mode_count, "bias_direction")
        a = self._arithmetic()
        xi = tuple(a.exact(value) for value in target_direction)
        beta = tuple(a.exact(value) for value in bias_direction)
        covariance = _bias_covariance(face.posterior, self.projected_centers, beta, a)
        if all(_zero(value) for value in covariance):
            return _response(face.tangent_projector, source.polynomial, model.time,
                             source.tail_norm_bound, xi, a)
        projected_target = _linear(face.tangent_projector, xi, a)
        inverse_time = _rational(a._coefficient(1/model.time))
        forcing = tuple(_add(_rational(value), _multiply(inverse_time, correction))
                        for value, correction in zip(projected_target, covariance, strict=True))
        return _interval_response(face.tangent_projector, source.polynomial, model.time,
                                  source.tail_norm_bound, forcing, a)

    def vjp(self, cotangent: Sequence[Rational]) -> tuple[Box, Box]:
        r"""Return (target_gradient, bias_gradient) for a fixed exact cotangent.

        With z=J*g, bias_gradient[j]=pi_j*(a_j-mu).z/s. Normal g gives
        exact zero in both blocks. Bias gradients mathematically sum to zero;
        independent component enclosures need not sum to a point interval.
        Use jvp for exact common-bias cancellation. An upstream gradient with
        its own uncertainty/dependence requires separate treatment.
        """
        source = self.sensitivity
        face, model = source.face, source.face.residual.unique.domain.model
        _validate_vector(cotangent, model.dimension, "cotangent")
        a = self._arithmetic()
        vector = tuple(a.exact(value) for value in cotangent)
        adjoint = _response(face.tangent_projector, source.polynomial, model.time,
                            source.tail_norm_bound, vector, a)
        if all(_zero(value) for value in adjoint):
            return adjoint, (_ZERO,)*model.mode_count
        result = []
        for j in range(model.mode_count):
            total = _ZERO
            for i, value in enumerate(adjoint):
                total = _add(total, _multiply(self.bias_forcing[i][j], value))
            result.append(total)
        return adjoint, tuple(result)


def certify_gaussian_entropy_bias(sensitivity: GaussianEntropySensitivity) -> GaussianEntropyBias | None:
    r"""Certify the original Gaussian log-weight Jacobian on a stable face.

    For U=(x-E_pi[a])/s, D_b U[beta]=-Cov_pi(a,beta)/s. Differentiating
    stable-face stationarity gives dx=J*(dv+Cov_pi(a,db)/s). Exact projected
    relative centers and the source posterior bound enclose every forcing
    column. The source matrix polynomial plus its rigorous tangent remainder
    then encloses the original bias Jacobian without a new inverse solve.

    The local implicit branch remains the unique global minimizer for small
    joint target/log-weight perturbations by uniform coercivity and base
    uniqueness. Only existence of a neighborhood is asserted, not its radius.
    No nearby posterior domains, finite solver derivatives, geometry gradients
    or derivatives at unresolved switches are asserted.

    Invalid source types raise; damaged/conditional dependencies, exhausted
    combined budgets and failed finite arithmetic return None. A new seal
    requires a rigorous backend and untainted history. Cached queries have no
    such backend requirement and preserve global strict/history state.
    """
    if not isinstance(sensitivity, GaussianEntropySensitivity):
        raise TypeError("sensitivity must be a GaussianEntropySensitivity")
    face = sensitivity.face
    model, projector = face.residual.unique.domain.model, face.residual.unique.domain.projector
    a = _Arithmetic(_combined_budget(model.budget, projector.budget))
    dimension, count = model.dimension, model.mode_count
    try:
        slots = _source_slots(sensitivity)
        a._entries(slots)
        a.limits._work(6*slots+4096*(count+1)*(dimension+1)**2+8192)
        a._work(6*slots)
        if not _sensitivity_intact(sensitivity):
            return None
        if libm_fallback_used():
            raise RuntimeError("Gaussian entropy bias refuses prior conditional libm fallback history")
        a._work(1024*(count+1)*(dimension+1)**2)
        relative = tuple(tuple(a.add(value, -origin) for value, origin in zip(row, model.centers[0], strict=True))
                         for row in model.centers)
        projected = tuple(_linear(face.tangent_projector, row, a) for row in relative)
        from omnibias.core.proof.certificate import encode_interval, make_certificate

        with _strict():
            mean = _projected_mean(face.posterior, projected, a)
            inverse_time = _rational(a._coefficient(1/model.time))
            columns = tuple(tuple(_multiply(inverse_time, _multiply(probability, _add(_rational(value), -average)))
                                  for value, average in zip(row, mean, strict=True))
                            for row, probability in zip(projected, face.posterior, strict=True))
            forcing = tuple(tuple(column[i] for column in columns) for i in range(dimension))
            responses = tuple(_interval_response(face.tangent_projector, sensitivity.polynomial,
                                                model.time, sensitivity.tail_norm_bound, column, a)
                              for column in columns)
            jacobian = tuple(tuple(column[i] for column in responses) for i in range(dimension))
            certificate = make_certificate(
                claim="Joint target and original-log-weight derivatives of the unique global Gaussian decision on a stable face.",
                payload={
                    "type": "gaussian_entropy_bias", "sensitivity_certificate": deepcopy(sensitivity.certificate),
                    "projected_centers": [[_pair(value) for value in row] for row in projected],
                    "projected_mean": [encode_interval(value) for value in mean],
                    "bias_forcing": _matrix_payload(forcing), "bias_jacobian": _matrix_payload(jacobian),
                    "argument": "The source proves a stable face with exact orthogonal tangent projector R, positive curvature at the exact globally unique decision, and a rigorously enclosed symmetric target Jacobian J. Original biases are log mixture weights: partial pi_i/partial b_j=pi_i*(delta_ij-pi_j), hence D_b U[beta]=-Cov_pi(a,beta)/s. Differentiating tangent stationarity gives dx=J*(dv+Cov_pi(a,db)/s). The IFT extends the local branch to joint target/bias perturbations. Fixed Gaussian centers, positive age/viscosity and bounded nearby targets/biases give uniform coercivity; base global uniqueness therefore forces every nearby global minimizer into the branch neighborhood. Strict projection margins persist, so the branch is the unique global decision there. No nearby global posterior-domain re-certification is needed for this existential neighborhood argument. Exact projected relative centers u_j=R*(a_j-a_0) and the original posterior box enclose forcing columns pi_j*(u_j-Eu)/s. Each true column is tangent even when its interval box contains nontangent vectors. The source finite matrix polynomial and its operator remainder, using exact endpoint squared norms and exact-comparison square roots, enclose J times each column. For joint queries center beta exactly, form Cov_pi(u,beta), and use the same tangent response. A constant beta gives exact zero covariance. For a fixed exact cotangent g, z=J*g and the bias pullback is the forcing transpose times z; exact normal g gives zero before numerical conversion. Interval products enclose correlated factors without claiming independence or a target/bias perturbation radius.",
                    "scope": "Only original log mixture weights and the target vary. Centers, age, viscosity and the polyhedron stay fixed. These are derivatives of the exact global decision in some open joint parameter neighborhood, not of a finite solver run. Bias component gradients sum to zero mathematically; their separate interval bounds need not sum to a point. No arbitrary activation invariance, higher parameter derivatives, geometry sensitivity, finite-step error bound, universal PDE solver, novelty theorem or external formal verification is claimed.",
                }, honesty={"unproven_claim": False, "unconditional_transcendentals": True, "continuum_pde_claim": False},
                meta={"transcend_backend": "mpmath"},
            )
        result = object.__new__(GaussianEntropyBias)
        for name, value in (("sensitivity", sensitivity), ("projected_centers", projected),
                            ("projected_mean", mean), ("bias_forcing", forcing),
                            ("bias_jacobian", jacobian), ("certificate", certificate)):
            object.__setattr__(result, name, value)
        return result
    except (ArithmeticError, ValueError):
        return None


__all__ = ["GaussianEntropyBias", "certify_gaussian_entropy_bias"]
