#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Exact, bounded symbolic checks for a local quadratic-vector-field slice.

Dependency: SymPy. This checks polynomial and rational identities over Q(C).
The analytic inverse-function theorem and its neighborhood conclusion are
mathematical arguments in the accompanying note, not formalized by this script.
"""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path

import sympy as sp
from omnibias.core.verified.interval import Interval
from omnibias.dynamics.quadratic_graphic import (
    certify_outer_schwarzian,
    outer_first_variation_ray,
    outer_splitting_asymptotics,
)
from omnibias.dynamics.quadratic_unfolding import certify_unfolding_nonoscillation


def require_zero(expression: sp.Expr, label: str) -> None:
    """Reject a claimed identity unless exact rational simplification gives zero."""
    residual = sp.cancel(expression)
    if residual != 0:
        raise ArithmeticError(f"{label}: nonzero exact residual {residual}")


def symbolic_audit() -> dict[str, object]:
    x, y, C, epsilon, w = sp.symbols("x y C epsilon w", real=True)
    A, mu1, mu2, mu3 = sp.symbols("A mu1 mu2 mu3", real=True)
    position = sp.Matrix([x, y])
    family = sp.Matrix(
        [
            A * x - y + x**2 + (mu2 + mu3) * x * y + mu1 * y**2,
            C * x + x**2 + x * y + mu3 * y**2,
        ]
    )
    base_parameters = {A: 1, mu1: 0, mu2: 0, mu3: 0}
    field = family.subs(base_parameters)
    derivative = field.jacobian(position)
    generator_names = ["(1,0)", "(0,1)", "(x,0)", "(y,0)", "(0,x)", "(0,y)"]
    generators = [
        sp.Matrix(vector)
        for vector in [(1, 0), (0, 1), (x, 0), (y, 0), (0, x), (0, y)]
    ]

    affine_tangents = []
    for name, generator in zip(generator_names, generators, strict=True):
        # Pullback action: (D Phi)^(-1) f(Phi(p)), Phi(p)=p+epsilon*g(p).
        # Therefore its first derivative is Df*g - Dg*f, with this sign.
        phi = position + epsilon * generator
        linear_part = sp.eye(2) + epsilon * generator.jacobian(position)
        transformed = linear_part.inv() * field.subs(
            {x: phi[0], y: phi[1]}, simultaneous=True
        )
        direct_tangent = transformed.diff(epsilon).subs(epsilon, 0)
        tangent = derivative * generator - generator.jacobian(position) * field
        for component in range(2):
            require_zero(direct_tangent[component] - tangent[component], f"affine {name}")
        affine_tangents.append(tangent)

    # Positive constant rescaling e^t of a vector field has tangent f at t=0.
    time_tangent = (sp.exp(epsilon) * field).diff(epsilon).subs(epsilon, 0)
    for component in range(2):
        require_zero(time_tangent[component] - field[component], "positive time tangent")

    parameters = [A, C, mu1, mu2, mu3]
    family_tangents = [family.diff(parameter).subs(base_parameters) for parameter in parameters]
    expected_family_tangents = [
        sp.Matrix(vector)
        for vector in [(x, 0), (0, x), (y**2, 0), (x * y, 0), (x * y, y**2)]
    ]
    for parameter, actual, expected in zip(
        parameters, family_tangents, expected_family_tangents, strict=True
    ):
        for component in range(2):
            require_zero(actual[component] - expected[component], f"family {parameter}")

    monomials = [sp.Integer(1), x, y, x**2, x * y, y**2]
    columns = affine_tangents + [time_tangent] + family_tangents
    matrix = sp.Matrix.hstack(
        *[
            sp.Matrix(
                [
                    sp.Poly(column[component], x, y).coeff_monomial(monomial)
                    for component in range(2)
                    for monomial in monomials
                ]
            )
            for column in columns
        ]
    )
    if matrix.shape != (12, 12):
        raise ArithmeticError(f"Wrong dimension: {matrix.shape}")
    determinant = sp.factor(matrix.det(method="bareiss"))
    require_zero(determinant - 3 * C, "slice determinant")
    reduced_matrix = sp.Matrix([[1, 1, -1, -1], [1, -1, 0, 1], [2, 0, -1, 1], [1, 2, 0, 1]])
    require_zero(reduced_matrix.det() - 3, "reduced determinant")
    if matrix.subs(C, 1).rank() != 12:
        raise ArithmeticError("Slice is unexpectedly singular at C=1")

    # Base-field coordinate change, with C constant along trajectories.
    q = (x**2 + 2 * x + C) / 2
    h = y - x**2 / 2 + C / 2
    h_time_derivative = sp.diff(h, x) * field[0] + sp.diff(h, y) * field[1]
    require_zero(field[0] - (q - h), "xdot=q-h")
    require_zero(h_time_derivative - 2 * x * h, "hdot=2*x*h")
    y_from_w = w + x**2 / 2 - C / 2
    transformed_field = sp.Matrix([field[0], h_time_derivative]).subs(y, y_from_w)
    require_zero(transformed_field[0] - (q - w), "xdot=q-w")
    require_zero(transformed_field[1] - 2 * x * w, "wdot=2*x*w")

    # Rational identity in x-time: requires q-w != 0 and q+w != 0.
    w_x = 2 * x * w / (q - w)
    J = w / (q + w) ** 2
    total_J_x = sp.diff(J, x) + sp.diff(J, w) * w_x
    require_zero(total_J_x + 2 * J / (q + w), "J_x=-2*J/(q+w)")

    # Exact overlap with the published positive-y phase chart.
    radius, z, t = sp.symbols("R z t", real=True)
    for sign in (-1, 1):
        section_y = (radius**2*(1 + z) + 2*sign*radius*z + C*(z - 1)) / 2
        phase_squared = radius**2 / section_y
        recovered_t = ((1 + sign/radius)*phase_squared - 1) / (
            1 + (sign/radius + C/radius**2)*phase_squared
        )
        require_zero(recovered_t - (1 - z)/(1 + z), f"phase overlap {sign}")
    z_from_t = (1 - t)/(1 + t)
    require_zero(z_from_t/(1 + z_from_t)**2 - (1 - t**2)/4, "H in t coordinate")

    # First parameter sources in w-time along the invariant base parabola.
    full_w_time = sp.diff(h, x)*family[0] + sp.diff(h, y)*family[1]
    expected_sources = {
        A: -x**2,
        mu1: -x*(x**2 - C)**2/4,
        mu2: -x**2*(x**2 - C)/2,
        mu3: (C**2 - x**4)/4,
    }
    for parameter, source in expected_sources.items():
        actual = full_w_time.diff(parameter).subs(base_parameters).subs(y, (x**2 - C)/2)
        require_zero(actual - source, f"unfolding source {parameter}")

    primitive_alpha = -(x**4 + 8*x**3 + 2*(C + 12)*x**2 + C*(C + 12))/(16*(C + 3))
    primitive_polynomials = {
        A: primitive_alpha,
        mu2: (C + 6)*primitive_alpha + x**3 + 3*x**2 + 3*C/2,
        mu3: 3*primitive_alpha + x**3/2 + 3*x**2/2 + C*x/2 + C,
    }
    for parameter, primitive in primitive_polynomials.items():
        require_zero(
            q*sp.diff(primitive, x) - 2*x*primitive - expected_sources[parameter],
            f"exact first-variation antiderivative {parameter}",
        )

    i0 = sp.symbols("I0")
    moments = [i0, 0, C*i0/3, 2*C*i0/3, C*(C + 4)*i0]
    for degree in range(4):
        polynomial = sp.expand(degree*x**(degree - 1)*q - 2*x**(degree + 1))
        moment_residual = sum(sp.Poly(polynomial, x).coeff_monomial(x**j)*moments[j] for j in range(5))
        require_zero(moment_residual, f"integration-by-parts moment {degree}")
    boundary_polynomial = sp.expand(4*x**3*q - 2*x**5)
    boundary_moment = sum(sp.Poly(boundary_polynomial, x).coeff_monomial(x**j)*moments[j] for j in range(5))
    require_zero(boundary_moment - sp.Rational(16, 3)*C*(C + 3)*i0, "fourth boundary moment")

    output = {
        "status": "all exact symbolic checks passed",
        "sympy_version": sp.__version__,
        "dimension": {"affine": 6, "positive_time_scale": 1, "family": 5, "coefficient_space": 12},
        "action": "exp(t) * B^(-1) * f_theta(B*p+b)",
        "affine_tangent_sign": "Df*g-Dg*f (pullback action)",
        "column_order": generator_names + ["time"] + [str(parameter) for parameter in parameters],
        "row_order": [f"f{component+1}:{monomial}" for component in range(2) for monomial in monomials],
        "coefficient_matrix": [[str(entry) for entry in row] for row in matrix.tolist()],
        "determinant": str(determinant),
        "local_ift_condition": "C != 0; affine B near I; exp(t)>0",
        "rank_at_C_1": 12,
        "coordinate_identities": ["xdot=q-w", "wdot=2*x*w", "w_x=2*x*w/(q-w)", "J_x=-2*J/(q+w)"],
        "chart_overlap_identities": ["finite-R t versus signed phase-square (both signs)", "H(z)=(1-t^2)/4"],
        "unfolding_source_identities": {str(parameter): str(source) for parameter, source in expected_sources.items()},
        "first_variation_antiderivative_identities": {
            str(parameter): str(primitive) for parameter, primitive in primitive_polynomials.items()
        },
        "moment_identities": ["I1=0", "I2=C*I0/3", "I3=2*C*I0/3", "I4=C*(C+4)*I0"],
        "outer_identity_domain": "q-w != 0 and q+w != 0; in particular C>1 and 0<=w/q<1",
        "scope": "Exact symbolic algebra; no analytic IFT, asymptotic convergence, or cyclicity theorem is machine-verified here.",
    }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional JSON artifact path")
    args = parser.parse_args()
    algebra = symbolic_audit()
    checks = []
    c_range = Interval(2.0, Interval.from_rational(Fraction(201, 100)).hi)
    for parameter, radius in ((Interval.point(2.0), 1e6), (c_range, 1e9)):
        result = certify_outer_schwarzian(parameter, radius)
        if not result.certified_for_larger_cutoffs:
            raise ArithmeticError("regular-passage negative-Schwarzian check did not close")
        upper = max(bound.hi for bound in result.schwarzian_bounds if bound is not None)
        checks.append({
            "C": [parameter.lo, parameter.hi],
            "R_min": radius,
            "all_R_at_least_R_min": result.certified_for_larger_cutoffs,
            "normalized_section": [-1/32, 1/32],
            "cells": result.cells,
            "schwarzian_upper": upper,
            "derivative_error_bounds": list(result.comparison.derivative_errors),
            "unresolved_cells": len(result.unresolved_cells),
        })
    data = {
        "symbolic_algebra": algebra,
        "regular_passage_checks": checks,
        "full_hilbert16_solved": False,
        "scope": "Regular passage, base first variations, and a full-family chart bound; written analytic arguments in the Hilbert16 package notes. No complete return-map or cyclicity theorem.",
        "formal_analytic_proof_run": False,
    }
    splitting = outer_splitting_asymptotics(c_range)
    ray = outer_first_variation_ray(c_range, 10000.0)
    if any(bound.hi >= 0 for bound in (ray.alpha, ray.mu2, ray.mu3)):
        raise ArithmeticError("first-variation signs did not close")
    data["first_parameter_variations"] = {
        "C": [c_range.lo, c_range.hi],
        "at": "A=1, mu=0, z_in=0; derivatives only, no nonlinear perturbation bound",
        "all_R_at_least": ray.cutoff_min,
        "derivative_over_R_squared": {
            "alpha": [ray.alpha.lo, ray.alpha.hi],
            "mu2": [ray.mu2.lo, ray.mu2.hi],
            "mu3": [ray.mu3.lo, ray.mu3.hi],
        },
        "mu1_derivative_over_R_squared_logR_limit": [splitting.mu1_log.lo, splitting.mu1_log.hi],
    }
    unfolding = certify_unfolding_nonoscillation()
    if not unfolding.certified_at_most_one_height_extremum or unfolding.drift_slope is None:
        raise ArithmeticError("full-family chart nonoscillation premises did not close")
    data["full_family_chart_nonoscillation"] = {
        "parameter_box": {
            name: [getattr(unfolding.parameters, name).lo, getattr(unfolding.parameters, name).hi]
            for name in ("nu", "a", "c", "m1", "m2", "m3")
        },
        "open_phase_box": {"v": [-2, 2], "z": [0, 4]},
        "ell_lower": unfolding.geometry.normal_v.lo,
        "turning_determinant_lower": unfolding.geometry.determinant_factor.lo,
        "nullcline_drift_slope_upper": unfolding.drift_slope.hi,
        "max_height_extrema_per_nonconstant_segment_in_box": 1,
        "passage_existence_proved": False,
        "return_map_zero_bound_proved": False,
    }
    output = json.dumps(data, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    print(output, end="")


if __name__ == "__main__":
    main()
