# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Quantitative intervals for Gaussian polyhedral path signs and optimizers."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING, Literal

from omnibias.core.gaussian_jet import _Arithmetic
from omnibias.core.verified.gaussian_affine import _dot
from omnibias.core.verified.gaussian_affine_sensitivity import _norm_upper
from omnibias.core.verified.gaussian_bias_analytic import (
    GaussianBiasSeriesCertificate,
    _certify_bias_series,
    _enclose,
    _tail_bound,
)
from omnibias.core.verified.gaussian_polyhedral_directional import _equilibrium_intact
from omnibias.core.verified.gaussian_polyhedral_path import GaussianPathSign, GaussianPolyhedralPath
from omnibias.core.verified.inverse_gaussian_affine import _intact
from omnibias.core.verified.inverse_gaussian_heat import _pair

if TYPE_CHECKING:
    from omnibias.core.proof.certificate import Cert
    from omnibias.core.verified.interval import Interval

Rational = int | Fraction
Vector = tuple[Fraction, ...]


def _parameter(value: Rational, name: str) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, Fraction)):
        raise TypeError(f"{name} must be an exact int or Fraction, excluding bool")
    return Fraction(value)


def _preflight(path: GaussianPolyhedralPath, arithmetic: _Arithmetic) -> None:
    source = path.equilibrium.face.base_restriction.source_model
    d, m, k, n = source.dimension, source.mode_count, len(path.equilibrium.face.inequalities), path.order
    arithmetic._entries(256*(m+k+1)*(d+1) + 128*(d+1)**2 + 96*(n+1)*(m+k+d+1) + 4096)
    arithmetic._work(512*(m+k+d+1)*(d+1)**2 + 256*(n+1)*(m+k+d+1)*(d+1) + 8192)


def _path_intact(path: GaussianPolyhedralPath) -> bool:
    return _intact(path.certificate) and _equilibrium_intact(path.equilibrium)


@dataclass(frozen=True)
class GaussianPathSignBound:
    """A scalar Cauchy tail compared with its first nonzero coefficient.

    ``change_bound`` bounds |f(t)-f(0)| on the analytic disk. The polynomial
    remainder and tail are divided by the leading power t^k. ``certified``
    means the full quantity has its recorded sign for every 0<t<=radius.
    Structural zeros have all bounds zero. Unknown zero prefixes have None
    remainder/tail/margin and are never promoted to a sign or identity.
    """

    sign: GaussianPathSign
    change_bound: Fraction
    polynomial_remainder_bound: Fraction | None
    scaled_tail_bound: Fraction | None
    margin: Fraction | None
    certified: bool


@dataclass(frozen=True, init=False)
class GaussianPolyhedralPathRadius:
    """Certified path behavior on the closed real interval [0, radius].

    For ``valid``, the face path is the unique global polyhedral optimizer
    throughout this interval. Its ``eventual_active_indices`` describes
    0<t<=radius; the active set at t=0 can be larger. For ``invalid``, at
    least one recorded negative sign holds throughout 0<t<=radius. This
    rejects the chosen KKT representation, with the same redundant-row
    caveat as the original path certificate. In that case other entries in
    ``sign_bounds`` may remain uncertified. The analytic face path still
    exists, and ``enclose`` bounds that face point without optimality claims.
    """

    path: GaussianPolyhedralPath
    analytic: GaussianBiasSeriesCertificate
    radius: Fraction
    sign_bounds: tuple[GaussianPathSignBound, ...]
    status: Literal["valid", "invalid"]
    eventual_active_indices: tuple[int, ...] | None
    certificate: Cert

    def __init__(self) -> None:
        raise TypeError("use certify_gaussian_polyhedral_path_radius")

    def _query(self, parameter: Rational) -> tuple[Fraction, _Arithmetic]:
        value = _parameter(parameter, "parameter")
        if value < 0 or value > self.radius:
            raise ValueError("parameter must lie in the certified interval [0, radius]")
        arithmetic = _Arithmetic(self.path.series.budget)
        _preflight(self.path, arithmetic)
        if not (_intact(self.certificate) and _intact(self.analytic.certificate) and _path_intact(self.path)):
            raise ValueError("damaged Gaussian path-radius certificate or dependency")
        arithmetic.exact(value)
        return value, arithmetic

    def tail_bound(self, parameter: Rational) -> Fraction:
        """Euclidean error of the finite point polynomial on 0<=t<=radius."""
        value, arithmetic = self._query(parameter)
        return _tail_bound(self.analytic, value, arithmetic)

    def enclose(self, parameter: Rational) -> tuple[Interval, ...]:
        """Enclose the true face point; it is the optimizer when status=valid."""
        value, arithmetic = self._query(parameter)
        return _enclose(self.analytic, value, arithmetic)


def _powers(value: Fraction, order: int, arithmetic: _Arithmetic) -> Vector:
    values = [Fraction(1)]
    for _ in range(order):
        values.append(arithmetic.mul(values[-1], value))
    return tuple(values)


def certify_gaussian_polyhedral_path_radius(
    path: GaussianPolyhedralPath, *, parameter_radius: Rational, spatial_radius: Rational, radius: Rational,
) -> GaussianPolyhedralPathRadius | None:
    r"""Quantify analytic face-path signs, including joint bias effects.

    R=parameter_radius and r=spatial_radius prove a complex inverse disk;
    rho=radius is the positive real endpoint. Require 0<rho<R and r>0.
    The bias-aware complex contraction certificate is built using one shared
    arithmetic budget. For each omitted slack use M=||P*G_i||*r. For a
    selected multiplier with row ell of L_E, use ORIGINAL centers and
    M=sum_n |ell*v_n|*R^n+range_j(ell*a_j)*eta/[2*s*(1-eta²)]. This remains
    necessary when projected centers coincide but original normal means vary.

    Cauchy's estimate bounds the tail after order N by
    M*(t/R)^(N+1)/(1-t/R). Divide by the first nonzero power t^k and bound
    all remaining coefficients on 0<t<=rho. A strict positive margin proves
    the recorded scalar sign throughout that interval. Structural identities
    need no tail test. A valid path needs every sign certified. An invalid
    path needs one certified negative sign; other quantities remain explicit
    diagnostics. An inconclusive path cannot acquire a sign from this bound.

    Invalid metadata raises. Failed sufficient bounds, inconclusive signs,
    damaged evidence or exhausted budgets return None. No radius search,
    exponentials, new series factorization or nonlinear solve is performed.
    The guarantee concerns the analytic face solution, not evaluation of the
    finite Taylor polynomial without its remainder. The certificate digest
    is not formal verification by an external theorem prover.
    """
    if not isinstance(path, GaussianPolyhedralPath):
        raise TypeError("path must be a GaussianPolyhedralPath")
    outer = _parameter(parameter_radius, "parameter_radius")
    spatial = _parameter(spatial_radius, "spatial_radius")
    inner = _parameter(radius, "radius")
    if outer <= 0 or spatial <= 0 or inner <= 0:
        raise ValueError("all radii must be strictly positive")
    if inner >= outer:
        raise ValueError("radius must lie strictly inside parameter_radius")
    arithmetic = _Arithmetic(path.series.budget)
    try:
        _preflight(path, arithmetic)
        if not _path_intact(path) or path.status == "inconclusive":
            return None
        outer, spatial, inner = (arithmetic.exact(value) for value in (outer, spatial, inner))
        analytic = _certify_bias_series(path.series, outer, spatial, arithmetic)
        if analytic is None:
            return None
        face = path.equilibrium.face
        source = face.base_restriction.source_model
        base_rows = len(face.base_restriction.constraints)
        left_inverse = face.geometry.left_inverse
        selected_rows = {index: left_inverse[base_rows+j] for j, index in enumerate(face.active_indices)}
        relative = tuple(tuple(arithmetic.add(value, -reference) for value, reference in
                               zip(center, source.centers[0], strict=True)) for center in source.centers)
        outer_powers = _powers(outer, path.order, arithmetic)
        inner_powers = _powers(inner, path.order+1, arithmetic)
        ratio = arithmetic._coefficient(inner/outer)
        ratio_power = _powers(ratio, path.order+1, arithmetic)[-1]
        cauchy_scale = arithmetic._coefficient(ratio_power/arithmetic.add(Fraction(1), -ratio))
        eta = analytic.phase_half_width_bound
        phase_margin = arithmetic.add(Fraction(1), -arithmetic.mul(eta, eta))
        mean_scale = arithmetic._coefficient(eta/arithmetic.mul(arithmetic.mul(Fraction(2), source.time), phase_margin))
        bounds: list[GaussianPathSignBound] = []
        for sign in path.sign_tests:
            if sign.verdict == "zero":
                bounds.append(GaussianPathSignBound(sign, Fraction(0), Fraction(0), Fraction(0), Fraction(0), True))
                continue
            if sign.quantity == "slack":
                change = arithmetic.mul(_norm_upper(face.slack_directions[sign.index], arithmetic), spatial)
            else:
                row = selected_rows[sign.index]
                target_growth = arithmetic.sum(tuple(arithmetic.mul(abs(_dot(row, vector, arithmetic)), power)
                                                     for vector, power in zip(path.target_coefficients[1:], outer_powers[1:], strict=True)))
                normal_centers = tuple(_dot(row, center, arithmetic) for center in relative)
                normal_range = arithmetic.add(max(normal_centers), -min(normal_centers))
                change = arithmetic.add(target_growth, arithmetic.mul(normal_range, mean_scale))
            if sign.leading_order is None or sign.leading_coefficient is None:
                bounds.append(GaussianPathSignBound(sign, change, None, None, None, False))
                continue
            leading = sign.leading_order
            polynomial = arithmetic.sum(tuple(arithmetic.mul(abs(coefficient), inner_powers[degree-leading])
                                              for degree, coefficient in enumerate(sign.coefficients)
                                              if degree > leading))
            tail = arithmetic._coefficient(arithmetic.mul(change, cauchy_scale)/inner_powers[leading])
            margin = arithmetic.add(arithmetic.add(abs(sign.leading_coefficient), -polynomial), -tail)
            bounds.append(GaussianPathSignBound(sign, change, polynomial, tail, margin, margin > 0))
        if path.status == "valid":
            if not all(bound.certified for bound in bounds):
                return None
            status: Literal["valid", "invalid"] = "valid"
        else:
            if not any(bound.certified and bound.sign.verdict == "negative" for bound in bounds):
                return None
            status = "invalid"
        from omnibias.core.proof.certificate import make_certificate

        certificate = make_certificate(
            claim="Quantitative right-sided interval for Gaussian analytic face-path KKT signs and Taylor enclosures.",
            payload={
                "type": "gaussian_polyhedral_path_radius", "path_certificate": deepcopy(path.certificate),
                "analytic_certificate": deepcopy(analytic.certificate), "radius": _pair(inner), "status": status,
                "eventual_active_indices": None if path.eventual_active_indices is None else list(path.eventual_active_indices),
                "sign_bounds": [{"quantity": bound.sign.quantity, "index": bound.sign.index,
                                 "verdict": bound.sign.verdict, "change_bound": _pair(bound.change_bound),
                                 "polynomial_remainder_bound": None if bound.polynomial_remainder_bound is None else _pair(bound.polynomial_remainder_bound),
                                 "scaled_tail_bound": None if bound.scaled_tail_bound is None else _pair(bound.scaled_tail_bound),
                                 "margin": None if bound.margin is None else _pair(bound.margin),
                                 "certified": bound.certified} for bound in bounds],
                "sign_interval": "0<t<=radius", "enclosure_interval": "0<=t<=radius",
                "argument": "The nested bias-aware analytic contraction proves a holomorphic face branch on the parameter disk. Slacks vary by at most ||P*G_i||*spatial_radius. The original normal multiplier formula uses L_E*x constant and bounds its mean change by mixed complex covariances of original normal centers with projected spatial and original bias increments. Cauchy's coefficient estimate for each scalar change gives an exact rational Taylor tail. After dividing by the leading t^k, all remainder terms increase with positive t, so a strict dominance inequality at radius proves the scalar sign throughout the punctured interval. Structural zeros retain their exact identities. If every KKT sign is certified nonnegative, q_base<1 proves the unique global constrained optimizer, including its certified base point. One certified negative sign suffices to reject the chosen KKT representation on the punctured interval.",
                "scope": "Fixed centers, polyhedron, viscosity and physical time; the complete bound polynomial target and log-bias paths. Valid certifies the true optimizer on the closed interval; the eventual active set is only for positive parameters. Invalid rejects the selected KKT representation: negative slack proves infeasibility, but negative multiplier need not exclude alternate multipliers on redundant tight rows. Uncertified scalar diagnostics on an invalid record carry no interval sign claim. The enclosed analytic face point need not be the constrained optimizer for an invalid record. No largest-radius, arbitrary nonuniform-base or formal-checker verification claim.",
            },
            honesty={"unproven_claim": False, "continuum_pde_claim": False},
            meta={"transcend_backend": "not_used"},
        )
        result = object.__new__(GaussianPolyhedralPathRadius)
        for name, stored in (("path", path), ("analytic", analytic), ("radius", inner), ("sign_bounds", tuple(bounds)),
                            ("status", status), ("eventual_active_indices", path.eventual_active_indices), ("certificate", certificate)):
            object.__setattr__(result, name, stored)
        return result
    except (ArithmeticError, ValueError):
        return None


__all__ = ["GaussianPathSignBound", "GaussianPolyhedralPathRadius", "certify_gaussian_polyhedral_path_radius"]
