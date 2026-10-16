# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Exact analytic face paths and eventual Gaussian KKT sign certificates."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING, Literal

from omnibias.core.gaussian_bias_jet import GaussianBiasInverseSeries, _build_equal_logits
from omnibias.core.gaussian_jet import _Arithmetic
from omnibias.core.verified.gaussian_affine import _dot, _validate_vector
from omnibias.core.verified.gaussian_polyhedral_directional import _equilibrium_intact
from omnibias.core.verified.gaussian_polyhedral_equilibrium import GaussianPolyhedralEquilibrium
from omnibias.core.verified.inverse_gaussian_affine import _linear
from omnibias.core.verified.inverse_gaussian_heat import _pair

if TYPE_CHECKING:
    from omnibias.core.proof.certificate import Cert

Rational = int | Fraction
Vector = tuple[Fraction, ...]
Matrix = tuple[Vector, ...]
Status = Literal["valid", "invalid", "inconclusive"]
Verdict = Literal["positive", "negative", "zero", "inconclusive"]


@dataclass(frozen=True)
class GaussianPathSign:
    """One right-sided analytic sign test, indexed by original inequality.

    A first nonzero coefficient determines the sign for sufficiently small
    positive parameters. An all-zero finite prefix is inconclusive unless
    ``identity`` proves the entire analytic quantity identically zero.
    """

    quantity: Literal["slack", "multiplier"]
    index: int
    coefficients: Vector
    leading_order: int | None
    leading_coefficient: Fraction | None
    verdict: Verdict
    identity: str | None


@dataclass(frozen=True, init=False)
class GaussianPolyhedralPath:
    """Taylor coefficients and eventual signs on a proposed equality face.

    All coefficient matrices are order-major ordinary Taylor coefficients,
    including order zero. The target and original log-bias paths are the
    specified finite polynomials; inverse and multiplier series need not
    terminate. Constraints, centers, viscosity and physical time stay fixed.

    ``valid`` certifies the unique global polyhedral optimizer for some
    unspecified interval 0<t<delta. ``invalid`` rejects this selected KKT
    representation: a negative slack proves primal infeasibility, whereas
    a negative selected multiplier may admit another representation through
    omitted tight constraints. ``inconclusive`` proves neither sign nor
    failure for at least one all-zero finite prefix. No numerical radius or
    finite-step error is provided. ``eventual_active_indices`` is populated
    only for a valid path and includes structurally tight omitted rows.
    """

    equilibrium: GaussianPolyhedralEquilibrium
    series: GaussianBiasInverseSeries
    target_coefficients: Matrix
    bias_coefficients: Matrix
    coefficients: Matrix
    posterior_coefficients: Matrix
    equality_multiplier_coefficients: Matrix
    inequality_multiplier_coefficients: Matrix
    slack_coefficients: Matrix
    sign_tests: tuple[GaussianPathSign, ...]
    status: Status
    eventual_active_indices: tuple[int, ...] | None
    certificate: Cert

    def __init__(self) -> None:
        raise TypeError("use compile_gaussian_polyhedral_path")

    @property
    def order(self) -> int:
        return len(self.coefficients)-1


def _sign(
    quantity: Literal["slack", "multiplier"], index: int, coefficients: Vector, identity: str | None,
) -> GaussianPathSign:
    for order, value in enumerate(coefficients):
        if value:
            return GaussianPathSign(quantity, index, coefficients, order, value,
                                    "positive" if value > 0 else "negative", None)
    return GaussianPathSign(quantity, index, coefficients, None, None,
                            "zero" if identity is not None else "inconclusive", identity)


def compile_gaussian_polyhedral_path(
    equilibrium: GaussianPolyhedralEquilibrium,
    higher_target_coefficients: Sequence[Sequence[Rational]],
    higher_bias_coefficients: Sequence[Sequence[Rational]],
) -> GaussianPolyhedralPath | None:
    r"""Compile a joint target/bias polynomial path on one proposed face.

    Supply equally many rows v_1,...,v_N and b_1,...,b_N, including explicit
    zero rows. The base record binds the ORIGINAL equal-logit Gaussian KKT
    equilibrium and q_base<1. Restrict to its independent selected face and
    reuse one rational LDLT throughout the implicit inverse recurrence.
    The same posterior series yields original moments and multipliers;
    replay normalization, all equality coefficients, and original KKT.

    Test selected inequality multipliers and omitted slacks by their first
    nonzero coefficient. Zero prefixes are inconclusive except identities:
    an omitted slack with P_face*G_i=0 is constant; constant target and
    relative-bias paths leave the whole equilibrium fixed; a selected
    multiplier is constant when its L row annihilates every center contrast
    and every higher target coefficient. No floating point, root iteration
    or transcendental evaluation is used. Exact budgets are shared across
    construction and replay. Invalid input raises; damaged dependencies,
    exhausted budgets and failed identities return None. A digest does not
    establish formal verification by an external proof checker.
    """
    if not isinstance(equilibrium, GaussianPolyhedralEquilibrium):
        raise TypeError("equilibrium must be a GaussianPolyhedralEquilibrium")
    for rows in (higher_target_coefficients, higher_bias_coefficients):
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            raise TypeError("higher coefficients must be finite exact matrices")
    if len(higher_target_coefficients) != len(higher_bias_coefficients):
        raise ValueError("target and bias paths must have the same number of higher coefficient rows")
    face = equilibrium.face
    source = face.base_restriction.source_model
    dimension, count = source.dimension, source.mode_count
    inequalities, base_rows = len(face.inequalities), len(face.base_restriction.constraints)
    order = len(higher_target_coefficients)
    if order > source.budget.max_order:
        return None
    arithmetic = _Arithmetic(source.budget)
    try:
        arithmetic._entries(128*(count+inequalities+1)*(dimension+1) + 64*(dimension+1)**2
                            + 32*(order+1)*(count+dimension+inequalities+1) + 1024)
        arithmetic._work(256*(count+inequalities+dimension+1)*(dimension+1)**2
                         + 64*(order+1)*(count+inequalities+dimension+1)*(dimension+1)**2 + 4096)
        if not _equilibrium_intact(equilibrium) or face.base_restriction.model.contraction_bound >= 1:
            return None
    except (ArithmeticError, ValueError):
        return None
    for row in higher_target_coefficients:
        _validate_vector(row, dimension, "target coefficient")
    for row in higher_bias_coefficients:
        _validate_vector(row, count, "bias coefficient")
    try:
        targets = (equilibrium.target,)+tuple(tuple(arithmetic.exact(value) for value in row)
                                             for row in higher_target_coefficients)
        biases = (source.biases,)+tuple(tuple(arithmetic.exact(value) for value in row)
                                       for row in higher_bias_coefficients)
        restriction = face.geometry.restriction
        projector, left_inverse = restriction.projector, face.geometry.left_inverse
        projected_targets = tuple(_linear(projector, row, arithmetic) for row in targets)
        series = _build_equal_logits(restriction.model, equilibrium.point, projected_targets[1:],
                                     biases[1:], source.budget, arithmetic)
        if series.target_coefficients != projected_targets or series.posterior_coefficients[0] != equilibrium.posterior:
            return None
        relative = tuple(tuple(arithmetic.add(value, -reference) for value, reference in
                               zip(center, source.centers[0], strict=True)) for center in source.centers)
        transpose = tuple(tuple(row[axis] for row in restriction.constraints) for axis in range(dimension))
        equalities: list[Vector] = []
        multipliers: list[Vector] = []
        slacks: list[Vector] = []
        for degree, (point, posterior, target) in enumerate(zip(
            series.coefficients, series.posterior_coefficients, targets, strict=True,
        )):
            if arithmetic.sum(posterior) != int(degree == 0):
                return None
            expected = restriction.rhs if degree == 0 else (Fraction(0),)*len(restriction.constraints)
            if _linear(restriction.constraints, point, arithmetic) != expected:
                return None
            mean = tuple(arithmetic.sum(tuple(arithmetic.mul(weight, center[axis])
                                             for weight, center in zip(posterior, relative, strict=True)))
                         for axis in range(dimension))
            position = point if degree else tuple(arithmetic.add(value, -reference) for value, reference in
                                                   zip(point, source.centers[0], strict=True))
            velocity = tuple(arithmetic._coefficient(arithmetic.add(value, -average)/source.time)
                             for value, average in zip(position, mean, strict=True))
            force = tuple(arithmetic.add(goal, -value) for goal, value in zip(target, velocity, strict=True))
            selected = _linear(left_inverse, force, arithmetic)
            if _linear(transpose, selected, arithmetic) != force:
                return None
            mu = [Fraction(0)]*inequalities
            for index, value in zip(face.active_indices, selected[base_rows:], strict=True):
                mu[index] = value
            slack = tuple(arithmetic.add(bound if degree == 0 else Fraction(0), -_dot(row, point, arithmetic))
                          for row, bound in zip(face.inequalities, face.upper_bounds, strict=True))
            if any(slack[index] for index in face.active_indices):
                return None
            equalities.append(selected[:base_rows])
            multipliers.append(tuple(mu))
            slacks.append(slack)
        if (equalities[0] != equilibrium.equality_multipliers or multipliers[0] != equilibrium.inequality_multipliers
                or slacks[0] != equilibrium.slacks):
            return None
        constant_parameters = not any(any(row) for row in targets[1:]) and not any(
            any(row) for row in series.relative_bias_coefficients[1:])
        signs: list[GaussianPathSign] = []
        for selected_index, index in enumerate(face.active_indices):
            left_row = left_inverse[base_rows+selected_index]
            normal_constant = not any(_dot(left_row, row, arithmetic) for row in relative) and not any(
                _dot(left_row, row, arithmetic) for row in targets[1:])
            identity = ("constant target and relative-bias paths" if constant_parameters else
                        "normal center contrasts and target increments vanish" if normal_constant else None)
            signs.append(_sign("multiplier", index, tuple(row[index] for row in multipliers), identity))
        selected_indices = set(face.active_indices)
        for index in range(inequalities):
            if index in selected_indices:
                continue
            identity = ("slack is constant on the selected affine face" if not any(face.slack_directions[index]) else
                        "constant target and relative-bias paths" if constant_parameters else None)
            signs.append(_sign("slack", index, tuple(row[index] for row in slacks), identity))
        status: Status = ("invalid" if any(sign.verdict == "negative" for sign in signs) else
                          "inconclusive" if any(sign.verdict == "inconclusive" for sign in signs) else "valid")
        active = tuple(sorted(selected_indices | {sign.index for sign in signs
                                                  if sign.quantity == "slack" and sign.verdict == "zero"}))
        eventual_active = active if status == "valid" else None
        from omnibias.core.proof.certificate import make_certificate

        certificate = make_certificate(
            claim="Exact analytic Gaussian face-path Taylor coefficients and right-sided KKT sign classification.",
            payload={
                "type": "gaussian_polyhedral_path", "equilibrium_certificate": deepcopy(equilibrium.certificate),
                "target_coefficients": [[_pair(value) for value in row] for row in targets],
                "bias_coefficients": [[_pair(value) for value in row] for row in biases],
                "coefficients": [[_pair(value) for value in row] for row in series.coefficients],
                "posterior_coefficients": [[_pair(value) for value in row] for row in series.posterior_coefficients],
                "equality_multiplier_coefficients": [[_pair(value) for value in row] for row in equalities],
                "inequality_multiplier_coefficients": [[_pair(value) for value in row] for row in multipliers],
                "slack_coefficients": [[_pair(value) for value in row] for row in slacks],
                "sign_tests": [{"quantity": sign.quantity, "index": sign.index,
                                "coefficients": [_pair(value) for value in sign.coefficients],
                                "leading_order": sign.leading_order,
                                "leading_coefficient": None if sign.leading_coefficient is None else _pair(sign.leading_coefficient),
                                "verdict": sign.verdict, "identity": sign.identity} for sign in signs],
                "status": status, "eventual_active_indices": None if eventual_active is None else list(eventual_active),
                "parameter_side": "positive", "radius": None,
                "identities": ["sum_j posterior_j[0]=1; sum_j posterior_j[n]=0 for n>0",
                               "E*x[0]=e; E*x[n]=0 for n>0", "U_b(x)-v+E^T*lambda=0 coefficientwise",
                               "selected slacks and omitted multipliers vanish identically"],
                "argument": "The fixed affine restriction preserves original posterior ratios under all supplied bias increments. Its q<1 gives an invertible analytic implicit equation and one positive definite rational Jacobian factor. Exact exponential recurrence and power-series division yield its Taylor coefficients. Original-center moments replay stationarity and equality coefficients. An analytic function has the sign of its first nonzero Taylor coefficient for sufficiently small positive parameters. Zero finite prefixes remain inconclusive unless an exact structural identity makes the full quantity zero. Finitely many positive or identically zero KKT quantities imply one common positive neighborhood. Under q_base<1, a valid representation then proves the unique global polyhedral optimum throughout that neighborhood.",
                "scope": "Fixed centers, constraints, viscosity and physical time; polynomial target and log-bias paths about the bound equal-logit equilibrium. Valid is an existential local optimizer certificate with no numerical radius, remainder or finite-step evaluation bound. Invalid rejects only this selected KKT representation; a negative multiplier does not exclude a different representation using omitted tight constraints. A negative slack does establish eventual primal infeasibility. Inconclusive is not failure or an identity claim. No arbitrary nonuniform posterior, exhaustive active-set search, continuum PDE claim or formal-checker verification is asserted.",
            },
            honesty={"unproven_claim": False, "continuum_pde_claim": False},
            meta={"transcend_backend": "not_used"},
        )
        result = object.__new__(GaussianPolyhedralPath)
        for name, stored in (("equilibrium", equilibrium), ("series", series), ("target_coefficients", targets),
                            ("bias_coefficients", biases), ("coefficients", series.coefficients),
                            ("posterior_coefficients", series.posterior_coefficients),
                            ("equality_multiplier_coefficients", tuple(equalities)),
                            ("inequality_multiplier_coefficients", tuple(multipliers)),
                            ("slack_coefficients", tuple(slacks)), ("sign_tests", tuple(signs)), ("status", status),
                            ("eventual_active_indices", eventual_active), ("certificate", certificate)):
            object.__setattr__(result, name, stored)
        return result
    except (ArithmeticError, ValueError):
        return None


__all__ = ["GaussianPathSign", "GaussianPolyhedralPath", "compile_gaussian_polyhedral_path"]
