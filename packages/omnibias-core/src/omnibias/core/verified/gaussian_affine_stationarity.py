# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Original Gaussian KKT multipliers from an affine root certificate."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

from omnibias.core.gaussian_jet import _Arithmetic
from omnibias.core.verified._gaussian_moments import _add, _multiply
from omnibias.core.verified.gaussian_affine_adjoint import GaussianAffineAdjointGeometry
from omnibias.core.verified.gaussian_affine_parameters import _action, _mean, _shift
from omnibias.core.verified.gaussian_affine_sensitivity import _binary_vector
from omnibias.core.verified.gaussian_heat import _rational
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.inverse_gaussian_affine import (
    GaussianAffineInverseCertificate,
    _intact,
    _linear,
    certify_gaussian_affine_inverse,
)
from omnibias.core.verified.inverse_gaussian_heat import _pair
from omnibias.core.verified.mean_map import _probabilities, _strict
from omnibias.core.verified.transcend import backend_name

if TYPE_CHECKING:
    from omnibias.core.proof.certificate import Cert

Vector = tuple[Fraction, ...]
Box = tuple[Interval, ...]


@dataclass(frozen=True, init=False)
class GaussianAffineStationarity:
    r"""Unique affine optimizer and its original primal KKT multipliers.

    ``root_enclosure`` contains X and ``primal_multiplier`` contains lambda
    in U_source(X)-target+C^T lambda=0, C X=rhs. The constraint rows are the
    declared full-row-rank representation. The multipliers do not depend on
    an external loss or adjoint. Their signs are not asserted by this record.

    ``root_certificate`` always contains the composed affine inverse proof;
    its underlying numerical root is absent for an exact affine inverse.
    ``posterior`` encloses the original Gaussian probabilities at X. It is
    empty only when there are no constraint rows and no multipliers to form.
    The root box need not consist entirely of feasible points. No component
    accuracy tolerance or derivative enclosure is claimed.
    """

    geometry: GaussianAffineAdjointGeometry
    target: Vector
    root_certificate: GaussianAffineInverseCertificate
    root_enclosure: Box
    posterior: Box
    primal_multiplier: Box
    certificate: Cert

    def __init__(self) -> None:
        raise TypeError("use certify_gaussian_affine_stationarity")


def certify_gaussian_affine_stationarity(
    geometry: GaussianAffineAdjointGeometry,
    target: Sequence[int | Fraction],
    scaled_point: Sequence[float],
) -> GaussianAffineStationarity | None:
    r"""Certify an untrusted root proposal and the original primal multipliers.

    Target data are exact int/Fraction; scaled_point is y=x/time with finite
    binary64 entries (exactly representable Python integers are accepted).
    Require the projected Gaussian contraction bound q_T<1. Neither the
    original ambient contraction bound nor feasibility of the proposal is
    required. Full row rank C is checked by the geometry compiler.

    For L=(C C^T)^(-1) C, the original stationarity equation gives
    lambda=L[target+(mu-anchor)/time]. Exact original center differences and
    the transformed posterior on the certified root box enclose this mean.
    Transformed and original probabilities agree at the feasible root. Even
    when the feasible set is a singleton or q_T=0, the original multipliers
    normally require these posterior weights. No loss upstream is involved.

    Aggregate work/storage are checked before composing root certificates.
    The query uses no inverse iteration, factorization or covariance matrix.
    Invalid types/shapes raise; failed contraction, damaged dependencies,
    exhausted budgets and nonfinite arithmetic return None. Missing rigorous
    support raises when posterior or root evaluation is required. Exact
    branches retain the shared strict sealing policy. Certificate digests
    are not theorem-prover verification.
    """
    if not isinstance(geometry, GaussianAffineAdjointGeometry):
        raise TypeError("geometry must be a GaussianAffineAdjointGeometry")
    restriction = geometry.restriction
    model, source = restriction.model, restriction.source_model
    dimension, count, rows = model.dimension, model.mode_count, len(restriction.constraints)
    if not isinstance(target, Sequence):
        raise TypeError("target must be a finite sequence of exact integers or Fractions")
    if len(target) != dimension:
        raise ValueError("target dimension must match the restriction")
    if any(isinstance(value, bool) or not isinstance(value, (int, Fraction)) for value in target):
        raise TypeError("target entries must be exact integers or Fractions, excluding bool")
    point = _binary_vector(scaled_point, dimension, "scaled_point")
    try:
        arithmetic = _Arithmetic(model.budget)
        arithmetic._entries(256*(count+rows+1)*(dimension+1) + 256*(dimension+1)**2 + 1024)
        # Include the affine root wrapper's conservative construction cost,
        # its residual evaluation, posterior work and all dependency copies.
        arithmetic._work(2048*(count+rows+1)*(dimension+1)**2 + 4*model._evaluation_work() + 4096)
        if not all(_intact(certificate) for certificate in (geometry.certificate, restriction.certificate)):
            return None
        frozen = tuple(arithmetic.exact(value) for value in target)
        if model.contraction_bound >= 1:
            return None
        root = certify_gaussian_affine_inverse(restriction, frozen, point)
        if root is None or not _intact(root.certificate):
            return None
        if root.root_certificate is not None and not _intact(root.root_certificate.certificate):
            return None
        posterior: Box = ()
        multiplier: Box = ()
        if rows:
            inverse_time = arithmetic._coefficient(1/model.time)
            relative = tuple(tuple(arithmetic.add(value, -reference) for value, reference in
                                   zip(center, source.centers[0], strict=True)) for center in source.centers)
            if model.contraction_bound == 0:
                coordinates = (Interval.point(0.0),)*dimension
            else:
                position = _shift(root.enclosure, model.centers[0], arithmetic)
                coordinates = tuple(_multiply(value, _rational(inverse_time)) for value in position)
            with _strict():
                posterior = _probabilities(model._mean_map._logits(coordinates))
            relative_mean = _mean(posterior, relative)
            base = tuple(arithmetic.add(value, arithmetic.mul(inverse_time, arithmetic.add(reference, -anchor)))
                         for value, reference, anchor in zip(frozen, source.centers[0], restriction.anchor, strict=True))
            constant = _linear(geometry.left_inverse, base, arithmetic)
            correction = _action(geometry.left_inverse, relative_mean)
            multiplier = tuple(_add(_rational(value), _multiply(_rational(inverse_time), component))
                               for value, component in zip(constant, correction, strict=True))
        from omnibias.core.proof.certificate import encode_interval, make_certificate

        certificate = make_certificate(
            claim="Original primal KKT multipliers at the unique affine-constrained Gaussian optimizer.",
            payload={
                "type": "gaussian_affine_stationarity",
                "geometry_certificate": deepcopy(geometry.certificate),
                "root_certificate": deepcopy(root.certificate),
                "target": [_pair(value) for value in frozen], "scaled_point": list(point),
                "root_enclosure": [encode_interval(value) for value in root.enclosure],
                "posterior": [encode_interval(value) for value in posterior],
                "primal_multiplier": [encode_interval(value) for value in multiplier],
                "equations": ["U_source(X)-target+C^T*lambda=0", "C*X=rhs"],
                "argument": "The intact affine inverse certificate encloses the unique constrained optimizer X under q_T<1. Original and transformed Gaussian posteriors agree on the affine set. With mu the original posterior mean, U_source(X)=(X-mu)/time and L*X=L*anchor. Applying the exact compiled L=(C*C^T)^(-1)*C to stationarity gives lambda=L*(target+(mu-anchor)/time). Relative original centers are formed exactly before interval mean evaluation, and outward posterior bounds on the root box enclose this expression. Full row rank makes these multipliers unique. Empty constraint rows need no multipliers or posterior.",
                "scope": "Original fixed-time Gaussian stationarity under affine equalities. No inequality feasibility, multiplier sign, loss adjoint, parameter derivative, higher derivative or component accuracy claim.",
            },
            honesty={"unproven_claim": False, "unconditional_transcendentals": bool(posterior) or root.root_certificate is not None,
                     "continuum_pde_claim": False},
            meta={"transcend_backend": backend_name() if posterior else root.certificate["meta"]["transcend_backend"]},
        )
        result = object.__new__(GaussianAffineStationarity)
        for name, value in (("geometry", geometry), ("target", frozen), ("root_certificate", root),
                            ("root_enclosure", root.enclosure), ("posterior", posterior),
                            ("primal_multiplier", multiplier), ("certificate", certificate)):
            object.__setattr__(result, name, value)
        return result
    except (ArithmeticError, ValueError):
        return None


__all__ = ["GaussianAffineStationarity", "certify_gaussian_affine_stationarity"]
