# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Certified joint bias/target directional calculus at Gaussian switches."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

from omnibias.core.gaussian_jet import _Arithmetic
from omnibias.core.verified._gaussian_moments import _add, _multiply
from omnibias.core.verified.gaussian_affine import _validate_vector
from omnibias.core.verified.gaussian_affine_sensitivity import _binary_vector
from omnibias.core.verified.gaussian_grouped_directional import (
    GaussianGroupedCriticalFace,
    _box_payload,
    _critical_response,
    _equilibrium_intact,
    _exact_plus_box,
    _matrix_payload,
)
from omnibias.core.verified.gaussian_heat import _rational
from omnibias.core.verified.gaussian_projected_equilibrium import GaussianProjectedEquilibrium
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.inverse_gaussian_affine import _intact, _linear
from omnibias.core.verified.inverse_gaussian_heat import _pair

if TYPE_CHECKING:
    from collections.abc import Sequence

    from omnibias.core.proof.certificate import Cert

Vector = tuple[Fraction, ...]
Matrix = tuple[Vector, ...]
Box = tuple[Interval, ...]
_ZERO = Interval.point(0.0)


def _split_mixture(rows: Matrix, probabilities: Box, arithmetic: _Arithmetic) -> tuple[Vector, Box]:
    """Keep a normalized mixture's first exact row outside interval arithmetic."""
    reference = rows[0]
    remainder = [_ZERO]*len(reference)
    for row, probability in zip(rows[1:], probabilities[1:], strict=True):
        for i, (component, base) in enumerate(zip(row, reference, strict=True)):
            difference = arithmetic.add(component, -base)
            if difference:
                remainder[i] = _add(remainder[i], _multiply(_rational(difference), probability))
    return reference, tuple(remainder)


def _group_covariance(offsets: Matrix, means: Vector, probabilities: Box, arithmetic: _Arithmetic) -> Box:
    """Enclose Cov_pi(offset_g, mean_g), after exact removal of both gauges."""
    relative_means = tuple(arithmetic.add(component, -means[0]) for component in means)
    if not any(relative_means) or not any(any(row) for row in offsets):
        # Either random variable is exactly constant. Do not convert the
        # other one just to multiply it by zero: it may exceed binary64.
        return (_ZERO,)*len(offsets[0])
    # The first centroid offset is exactly zero. Projecting offsets by an
    # exact matrix preserves it and any identically zero normal-row entries.
    center_reference, center_remainder = _split_mixture(offsets, probabilities, arithmetic)
    mean_reference, mean_remainder = _split_mixture(tuple((component,) for component in relative_means), probabilities, arithmetic)
    centers = _exact_plus_box(center_reference, center_remainder, arithmetic)
    mean = _exact_plus_box(mean_reference, mean_remainder, arithmetic)[0]
    products = tuple(tuple(arithmetic.mul(component, beta) for component in row)
                     for row, beta in zip(offsets, relative_means, strict=True))
    reference, remainder = _split_mixture(products, probabilities, arithmetic)
    second = _exact_plus_box(reference, remainder, arithmetic)
    result = []
    for product, center in zip(second, centers, strict=True):
        correction = _multiply(center, mean)
        result.append(_add(product, Interval(-correction.hi, -correction.lo)))
    return tuple(result)


def _add_scaled(remainder: Box, correction: Box, scale: Fraction) -> Box:
    factor = _rational(scale)
    return tuple(_add(value, _multiply(factor, extra)) for value, extra in zip(remainder, correction, strict=True))


@dataclass(frozen=True, init=False)
class GaussianJointDirectional:
    r"""Enclose a one-sided joint original-bias/target decision derivative.

    ``target_direction`` and ``bias_direction`` are the exact directions xi
    and beta in X(b+t*beta,v+t*xi). Original centers, time, viscosity and all
    constraints stay fixed. The unique directional response solves the base
    critical-cone QP with forcing zeta=xi+Cov_p(a,beta)/time.

    ``relative_bias_direction`` subtracts beta[0] exactly. Group bias means
    use that relative direction; ``group_bias_covariances`` stores exact
    vectors mean_group[(a-c_g)*(beta-mean_group(beta))]. Groups refer only to
    the base equilibrium and need not stay equal-logit after perturbation.

    The three forcing boxes enclose zeta, P*zeta and L*zeta respectively.
    ``zero_projected_forcing`` records an exact groupwise zero identity,
    never an interval merely containing zero. A successful zero-forcing
    shortcut returns value=0 regardless of the supplied proposal. Otherwise
    value=P*proposal, and ``error_bound`` bounds its Euclidean distance from
    the true directional response. Coordinate and residual bounds have the
    same meaning as the target-only certificate.

    Multipliers are for the linearized critical QP in compiled E row order.
    With redundant tight rows they need not define a unique derivative of a
    chosen primal multiplier map. Weak arrays follow original inequalities;
    only weak-row entries have QP meaning. Zero derivative asserts no finite
    step constancy, inactivity, or persistence of a selected active face.
    """

    face: GaussianGroupedCriticalFace
    target_direction: Vector
    bias_direction: Vector
    relative_bias_direction: Vector
    group_bias_means: Vector
    group_bias_covariances: Matrix
    forcing_enclosure: Box
    projected_forcing_enclosure: Box
    normal_forcing_enclosure: Box
    zero_projected_forcing: bool
    proposal: tuple[float, ...]
    projected_proposal: Vector
    value: Vector
    enclosure: Box
    error_bound: Fraction
    coordinate_error_bounds: Vector
    residual_enclosure: Box
    residual_norm_bound: Fraction
    equality_multipliers: Box
    weak_multipliers: Box
    weak_slacks: Box
    certificate: Cert

    def __init__(self) -> None:
        raise TypeError("use certify_gaussian_joint_directional")


def certify_gaussian_joint_directional(
    face: GaussianGroupedCriticalFace, target_direction: Sequence[int | Fraction],
    bias_direction: Sequence[int | Fraction], proposal: Sequence[float],
) -> GaussianJointDirectional | None:
    r"""Certify a joint bias/target response using cached weights and residuals.

    D_b U[beta]=-Cov_p(a,beta)/time. Hence the fixed-geometry critical QP
    has the same Hessian and cone as before, with forcing
    zeta=xi+Cov_p(a,beta)/time. For each original base logit group let
    k_g=mean_group[(a-c_g)*(beta-mean_group(beta))]. Total covariance is
    sum pi_g*k_g+Cov_pi(c_g,mean_group(beta)). The second term vanishes
    after any compiled critical projector P, but generally survives L.
    Exact centroid differences, within-group bias deviations and transforms
    precede interval conversion, preserving common-bias gauges and zeros.

    Per-group projected forcing is P*xi+P*k_g/time. Its exact first row
    plus an interval remainder encloses P*zeta without rounding the declared
    direction. If every group row is zero, the equality-QP solution is zero;
    weak signs still must pass. Otherwise the projected Hessian residual
    divided by (1-q_base)/time bounds response error. Full normal forcing
    includes the between-group term, so certified QP multipliers remain valid.

    No solve, factorization, nonlinear iteration, active-set search or new
    transcendental evaluation occurs. All directions are exact integers or
    Fractions and proposals are finite binary64; malformed inputs raise even
    on shortcuts. Damaged dependencies, exhausted resources, unrepresentable
    enclosures or inconclusive signs return None. Source budget preflights
    include O((m+k)*d²+d²) work, transformed group moments and retained
    evidence. Shared strict sealing policy and sticky fallback state persist.
    No ordinary Jacobian/VJP, finite-step remainder, higher derivative or
    moving-geometry derivative is claimed. A digest is not formal proof.
    """
    if not isinstance(face, GaussianGroupedCriticalFace):
        raise TypeError("face must be a GaussianGroupedCriticalFace")
    equilibrium = face.equilibrium
    source = equilibrium.face.base_restriction.source_model
    dimension, count, inequalities = source.dimension, source.mode_count, len(equilibrium.face.inequalities)
    _validate_vector(target_direction, dimension, "target_direction")
    _validate_vector(bias_direction, count, "bias_direction")
    binary_proposal = _binary_vector(proposal, dimension, "proposal")
    try:
        arithmetic = _Arithmetic(source.budget)
        arithmetic._entries(2048*(count+inequalities+1)*(dimension+1)
                            + 512*(count+1)*(dimension+1)**2 + 16384)
        arithmetic._work(8192*(count+inequalities+dimension+1)*(dimension+1)**2
                         + 256*(count+1)*(dimension+1) + 32768)
        if not _intact(face.certificate) or not _equilibrium_intact(equilibrium):
            return None
        target = tuple(arithmetic.exact(component) for component in target_direction)
        bias = tuple(arithmetic.exact(component) for component in bias_direction)
        relative = tuple(arithmetic.add(component, -bias[0]) for component in bias)
        groups = equilibrium.groups
        means = tuple(arithmetic._coefficient(arithmetic.sum(tuple(relative[index] for index in group))/len(group))
                      for group in groups)
        within = []
        for group, mean in zip(groups, means, strict=True):
            deviations = tuple(arithmetic.add(relative[index], -mean) for index in group)
            within.append(tuple(arithmetic._coefficient(arithmetic.sum(tuple(
                arithmetic.mul(equilibrium.centered_centers[index][axis], deviation)
                for index, deviation in zip(group, deviations, strict=True)))/len(group)) for axis in range(dimension)))
        covariances = tuple(within)
        inverse_time = arithmetic._coefficient(1/source.time)
        projected_target = _linear(face.projector, target, arithmetic)
        normal_target = _linear(face.left_inverse, target, arithmetic)
        full_groups, projected_groups, normal_groups = [], [], []
        for row in covariances:
            full_groups.append(tuple(arithmetic.add(a, arithmetic.mul(inverse_time, b)) for a, b in zip(target, row, strict=True)))
            projected_groups.append(tuple(arithmetic.add(a, arithmetic.mul(inverse_time, b))
                                          for a, b in zip(projected_target, _linear(face.projector, row, arithmetic), strict=True)))
            normal_groups.append(tuple(arithmetic.add(a, arithmetic.mul(inverse_time, b))
                                       for a, b in zip(normal_target, _linear(face.left_inverse, row, arithmetic), strict=True)))
        full_reference, full_remainder = _split_mixture(tuple(full_groups), face.group_probabilities, arithmetic)
        projected_reference, projected_remainder = _split_mixture(tuple(projected_groups), face.group_probabilities, arithmetic)
        normal_reference, normal_remainder = _split_mixture(tuple(normal_groups), face.group_probabilities, arithmetic)
        if isinstance(equilibrium, GaussianProjectedEquilibrium):
            offsets = equilibrium.group_centroid_offsets
            full_between = _group_covariance(offsets, means, face.group_probabilities, arithmetic)
            normal_offsets = tuple(_linear(face.left_inverse, row, arithmetic) for row in offsets)
            normal_between = _group_covariance(normal_offsets, means, face.group_probabilities, arithmetic)
            full_remainder = _add_scaled(full_remainder, full_between, inverse_time)
            normal_remainder = _add_scaled(normal_remainder, normal_between, inverse_time)
        forcing = _exact_plus_box(full_reference, full_remainder, arithmetic)
        projected_forcing = _exact_plus_box(projected_reference, projected_remainder, arithmetic)
        normal_forcing = _exact_plus_box(normal_reference, normal_remainder, arithmetic)
        zero_projected = not any(any(row) for row in projected_groups)
        projected = _linear(face.projector, tuple(arithmetic._coefficient(Fraction(component)) for component in binary_proposal), arithmetic)
        response = _critical_response(face, projected, projected_reference, normal_reference, arithmetic,
                                      forcing_remainder=projected_remainder, normal_remainder=normal_remainder)
        if response is None:
            return None
        from omnibias.core.proof.certificate import make_certificate

        certificate = make_certificate(
            claim="Rigorous one-sided joint original-bias/target derivative of a constrained Gaussian decision.",
            payload={
                "type": "gaussian_joint_directional", "critical_face_certificate": deepcopy(face.certificate),
                "target_direction": [_pair(component) for component in target],
                "bias_direction": [_pair(component) for component in bias],
                "relative_bias_direction": [_pair(component) for component in relative],
                "group_bias_means": [_pair(component) for component in means],
                "group_bias_covariances": _matrix_payload(covariances),
                "forcing_enclosure": _box_payload(forcing), "projected_forcing_enclosure": _box_payload(projected_forcing),
                "normal_forcing_enclosure": _box_payload(normal_forcing), "zero_projected_forcing": zero_projected,
                "projected_forcing_reference": [_pair(component) for component in projected_reference],
                "projected_forcing_remainder": _box_payload(projected_remainder),
                "normal_forcing_reference": [_pair(component) for component in normal_reference],
                "normal_forcing_remainder": _box_payload(normal_remainder),
                "proposal": list(binary_proposal), "projected_proposal": [_pair(component) for component in projected],
                "value": [_pair(component) for component in response.value], "enclosure": _box_payload(response.enclosure),
                "error_bound": _pair(response.error_bound),
                "coordinate_error_bounds": [_pair(component) for component in response.coordinate_error_bounds],
                "residual_enclosure": _box_payload(response.residual_enclosure), "residual_norm_bound": _pair(response.residual_norm_bound),
                "equality_multipliers": _box_payload(response.equality_multipliers),
                "weak_multipliers": _box_payload(response.weak_multipliers), "weak_slacks": _box_payload(response.weak_slacks),
                "argument": "The softmax derivative gives D_b U[beta]=-Cov_p(a,beta)/time. Uniform base-tangent strong convexity persists under bias changes because its diameter bound is independent of biases. Parametric second-order variational expansion therefore gives the unique joint Hadamard directional response as the base critical-cone QP minimizer with forcing zeta=xi+Cov_p(a,beta)/time. Group conditional weights are uniform at the base point, even when the proposed perturbation splits those groups later. Exact within-group moments plus centroid/bias-mean covariance enclose the full forcing. The between-group term vanishes in P*zeta by the compiled projected-centroid identity, but is retained in L*zeta. Exact common-bias offsets and first group references are removed before interval conversion. The shared residual/coercivity bound encloses the equality-QP response; weak multiplier/slack signs prove it is the critical-cone minimizer. Exact groupwise zero projected forcing permits a zero-response shortcut, still checked against the weak signs.",
                "scope": "Only original Gaussian biases and optimization target vary in the declared exact directions; centers, age, viscosity and all geometry are fixed. One-sided Hadamard derivative enclosure, not an ordinary Jacobian/VJP, finite-step approximation, switching radius, active-face persistence, higher derivative or moving-geometry derivative. QP multipliers need not be unique derivatives of a globally selected primal multiplier map. Only weak-row entries of weak arrays have QP meaning. No query solve or transcendental evaluation.",
            }, honesty={"unproven_claim": False, "continuum_pde_claim": False},
            meta={"transcend_backend": face.certificate["meta"]["transcend_backend"]},
        )
        result = object.__new__(GaussianJointDirectional)
        for name, stored in (("face", face), ("target_direction", target), ("bias_direction", bias),
                             ("relative_bias_direction", relative), ("group_bias_means", means),
                             ("group_bias_covariances", covariances), ("forcing_enclosure", forcing),
                             ("projected_forcing_enclosure", projected_forcing), ("normal_forcing_enclosure", normal_forcing),
                             ("zero_projected_forcing", zero_projected), ("proposal", binary_proposal),
                             ("projected_proposal", projected), ("value", response.value), ("enclosure", response.enclosure),
                             ("error_bound", response.error_bound), ("coordinate_error_bounds", response.coordinate_error_bounds),
                             ("residual_enclosure", response.residual_enclosure), ("residual_norm_bound", response.residual_norm_bound),
                             ("equality_multipliers", response.equality_multipliers), ("weak_multipliers", response.weak_multipliers),
                             ("weak_slacks", response.weak_slacks), ("certificate", certificate)):
            object.__setattr__(result, name, stored)
        return result
    except (ArithmeticError, ValueError):
        return None


__all__ = ["GaussianJointDirectional", "certify_gaussian_joint_directional"]
