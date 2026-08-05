# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Proof-carrying optimization: a certified subspace trust-region step.

This module fuses the two halves of omnibias's training story into a single
primitive that neither has alone -- a **proof-carrying optimization trajectory**.

* The *differentiable* register (``omnibias.torch.optim.JetSubspaceTensor``)
  takes an exact degree-3 Taylor model of the loss restricted to a small
  ``k``-dimensional Krylov subspace ``Q`` and steps inside a trust region.  It is
  fast but says nothing rigorous about how faithful that model is.
* The *rigorous* register (:mod:`omnibias.verify._core.param_loss`) certifies a
  trained ``theta*`` is a strict local minimum, but statically -- it does not
  follow the optimizer.

Here we certify the optimizer's *step*.  Given a base point ``theta0``, an
orthonormal subspace basis ``Q`` (``P x k``) and a trust radius ``r``, define the
restriction

.. math::

    \psi(a) = L(\theta_0 + Q a), \qquad a \in [-r, r]^k .

We enclose ``psi`` as an order-3 :class:`~omnibias.core.verified.taylor_model_mv.TaylorModelMV`
over the box.  Its polynomial part is the *exact* reduced Taylor model
``m(a) = psi(0) + c.a + 1/2 a^T H a + 1/6 T[a,a,a]`` (the same object the torch
``taylor_subspace_model`` returns) and its scalar :attr:`.remainder` ``R`` is a
**rigorous bound on the model-vs-truth error** ``psi(a) - m(a)`` valid for *every*
``a`` in the box.  The activation towers are composed onto the Taylor model in
closed form (Makino-Berz: the exact ``sigma^(n)`` from
:func:`~omnibias.core.verified.sigma.sigma_tower_interval` plus a Lagrange
remainder for the truncated series), so every term of total degree ``> 3`` lands
soundly in ``R``.

For a candidate step ``a*`` (the Cauchy point, or one ingested from the torch
``solve_subspace_trust_region``) with predicted decrease ``pred = m(0) - m(a*)``
and ``w = R.width``, the true decrease is rigorously enclosed:

.. math::

    \Delta = \psi(0) - \psi(a^*) \in [\,\mathrm{pred} - w,\ \mathrm{pred} + w\,],

because ``psi(0) in m(0) + R`` and ``psi(a*) in m(a*) + R`` share the same uniform
remainder ``R`` at two independent points (so ``R - R = [-w, w]``).  The step is
**certified to strictly decrease the true loss** iff ``pred - w > 0``.  The
remainder scales like ``O(r^4)`` while ``pred`` scales like ``O(r)``, so for a
small enough radius the margin is positive -- the same "small box" mechanism that
makes the strict-local-min certificate work.

The margin is sealed as a one-signed interval certificate whose ``> 0`` sign
obligation the Lean kernel can re-check (``enclosed_quantity_pos``), alongside a
companion certificate recording the model-vs-truth remainder ``R``.

Honest scope: small networks, fixed data, ``tanh`` / ``sigmoid`` activations, a
small subspace and a small trust radius.  Each certificate is a rigorous *local*
proof about one step; :func:`certify_trajectory` chains them into a proof-carrying
descent.  This is not a global-optimality or open-problem claim.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from omnibias.core.proof.certificate import Cert, interval_certificate, verify_certificate_digest
from omnibias.core.proof.lean_check import LeanCheckResult, check_certificate
from omnibias.core.verified.interval import Interval
from omnibias.verify._core.param_jet import Data, param_jet
from omnibias.verify._core.param_loss import MLPArchitecture, ParamSpaceLoss

_ORDER = 3  # the certified subspace Taylor model is degree 3 (exact c, H, T + remainder)


@dataclass(frozen=True)
class SubspaceModel:
    r"""The exact degree-3 subspace Taylor model plus its rigorous remainder.

    ``m(a) = constant + grad.a + 1/2 a^T hessian a + 1/6 third[a,a,a]`` is the
    exact Taylor model of ``psi(a) = L(theta0 + Q a)`` about ``a = 0`` (the reduced
    gradient / Hessian / third-derivative are :class:`Interval` enclosures, tight up
    to outward rounding), and :attr:`remainder` rigorously bounds ``psi(a) - m(a)``
    over the whole trust box ``[-radius, radius]^k``.
    """

    radius: float
    k: int
    basis: tuple[tuple[float, ...], ...]
    constant: Interval
    grad: tuple[Interval, ...]
    hessian: tuple[tuple[Interval, ...], ...]
    third: tuple[tuple[tuple[Interval, ...], ...], ...]
    remainder: Interval


def enclose_subspace_model(
    arch: MLPArchitecture,
    data: Data,
    theta0: Sequence[float],
    basis: Sequence[Sequence[float]],
    *,
    radius: float,
    l2: float = 0.0,
) -> SubspaceModel:
    r"""Rigorously enclose the degree-3 subspace Taylor model of the training loss.

    Builds the order-3 :class:`TaylorModelMV` of ``psi(a) = J(theta0 + Q a)`` over
    the box ``[-radius, radius]^k`` -- ``J = L`` for ``l2 = 0`` and the
    L2-regularised objective ``L + l2 ||theta||^2`` otherwise -- and reads off the
    exact reduced coefficients (``constant``, ``grad``, ``hessian``, ``third``) and
    the rigorous model-vs-truth ``remainder``.

    ``basis`` is ``P x k`` (``P = arch.n_params``); the parameter map
    ``theta_p = theta0_p + sum_j basis[p][j] a_j`` is exact and affine in ``a``, so
    the ``k`` subspace variables are shared across all ``P`` parameters -- the
    dependency structure Taylor-model arithmetic exploits to keep the enclosure
    tight.

    This is the order-3 subspace special case of the general
    :func:`~omnibias.verify._core.param_jet.param_jet`; the reduced ``constant`` /
    ``grad`` / ``hessian`` / ``third`` are its order 0..3 derivative readouts.
    """
    if radius <= 0.0:
        raise ValueError(f"radius must be > 0, got {radius}")
    if l2 < 0.0:
        raise ValueError(f"l2 (weight decay) must be >= 0, got {l2}")
    theta = [float(t) for t in theta0]
    p_dim = arch.n_params
    if len(theta) != p_dim:
        raise ValueError(f"theta0 has {len(theta)} entries but the model has {p_dim} parameters")
    if len(basis) != p_dim:
        raise ValueError(f"basis has {len(basis)} rows but the model has {p_dim} parameters")

    jet = param_jet(arch, data, theta, order=_ORDER, radius=radius, l2=l2, basis=basis)
    return SubspaceModel(
        radius=float(radius),
        k=jet.dim,
        basis=tuple(tuple(float(v) for v in row) for row in basis),
        constant=jet.value(),
        grad=jet.grad(),
        hessian=jet.hessian(),
        third=jet.tensor(_ORDER),
        remainder=jet.remainder,
    )


def _poly_value_at(model: SubspaceModel, a: Sequence[float]) -> Interval:
    r"""Rigorous enclosure of the polynomial model ``m(a)`` at a point ``a``."""
    pts = [Interval.point(float(ai)) for ai in a]
    acc = model.constant
    for i in range(model.k):
        acc = acc + model.grad[i] * pts[i]
    half = Interval.from_rational(Fraction(1, 2))
    for i in range(model.k):
        for j in range(model.k):
            acc = acc + half * model.hessian[i][j] * pts[i] * pts[j]
    sixth = Interval.from_rational(Fraction(1, 6))
    for i in range(model.k):
        for j in range(model.k):
            for el in range(model.k):
                acc = acc + sixth * model.third[i][j][el] * pts[i] * pts[j] * pts[el]
    return acc


# --------------------------------------------------------------------------- #
# Subspace basis and trust-region step (pure Python; no torch).
# --------------------------------------------------------------------------- #
def krylov_basis(
    problem: ParamSpaceLoss, theta0: Sequence[float], k: int
) -> tuple[tuple[float, ...], ...]:
    r"""Orthonormal Krylov basis ``span{g, H g, H^2 g, ...}`` at ``theta0``.

    Builds the subspace the exact-Hessian methods favour -- an Arnoldi/Lanczos
    sweep on the *point* gradient and Hessian midpoints of ``problem`` at
    ``theta0`` -- entirely in pure Python, so the certifier needs no differentiable
    backend.  Returns a ``P x k`` basis ``basis[p][j]`` (fewer than ``k`` columns if
    the Krylov space is exhausted first).
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    p_dim = problem.arch.n_params
    theta = [float(t) for t in theta0]
    if len(theta) != p_dim:
        raise ValueError(f"theta0 has {len(theta)} entries but the model has {p_dim} parameters")
    point_box = [Interval.point(t) for t in theta]
    g_iv = problem.grad(point_box)
    h_iv = problem.hessian(point_box)
    g = [iv.mid for iv in g_iv]
    hmat = [[h_iv[i][j].mid for j in range(p_dim)] for i in range(p_dim)]

    cols: list[list[float]] = []
    v = g
    for _ in range(min(k, p_dim)):
        for q in cols:  # modified Gram-Schmidt against the existing columns
            dot = math.fsum(v[i] * q[i] for i in range(p_dim))
            v = [v[i] - dot * q[i] for i in range(p_dim)]
        norm = math.sqrt(math.fsum(vi * vi for vi in v))
        if norm <= 1e-14:
            break
        q_new = [vi / norm for vi in v]
        cols.append(q_new)
        v = [math.fsum(hmat[i][j] * q_new[j] for j in range(p_dim)) for i in range(p_dim)]
    if not cols:
        raise ValueError("the gradient is (numerically) zero at theta0; no descent subspace")
    kk = len(cols)
    return tuple(tuple(cols[j][p] for j in range(kk)) for p in range(p_dim))


def cauchy_step(grad_reduced: Sequence[float], radius: float) -> tuple[float, ...]:
    r"""Steepest-descent step to the trust-region boundary: ``-radius * c / ||c||``."""
    norm = math.sqrt(math.fsum(g * g for g in grad_reduced))
    if norm <= 0.0:
        return tuple(0.0 for _ in grad_reduced)
    scale = radius / norm
    return tuple(-scale * float(g) for g in grad_reduced)


# --------------------------------------------------------------------------- #
# The certified step.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SubspaceStepCertificate:
    r"""A sealed, tamper-evident proof that a subspace trust-region step descends.

    Bundles the trust-region step ``a*`` (reduced coordinates), the rigorous
    model-vs-truth :attr:`remainder` ``rho``, the certified true-loss-decrease
    enclosure ``[pred - w, pred + w]``, and the two sealed v1
    :class:`~omnibias.core.proof.certificate.Cert`\s: :attr:`model_certificate`
    (the remainder ``rho``) and :attr:`descent_certificate` (the one-signed
    decrease enclosure, whose ``> 0`` obligation the Lean kernel can re-check).
    """

    theta0: tuple[float, ...]
    basis: tuple[tuple[float, ...], ...]
    radius: float
    step: tuple[float, ...]
    predicted_decrease: float
    remainder: Interval
    decrease_enclosure: Interval
    model_certificate: Cert
    descent_certificate: Cert
    lean: LeanCheckResult | None = None

    @property
    def verified(self) -> bool:
        """``True`` iff both sealed certificates' digests match their bodies (untampered)."""
        return verify_certificate_digest(self.model_certificate) and verify_certificate_digest(
            self.descent_certificate
        )

    @property
    def certified(self) -> bool:
        """``True`` iff the step is proven to strictly decrease the true loss (``margin > 0``)."""
        return self.decrease_enclosure.lo > 0.0

    @property
    def theorem_prover_verified(self) -> bool:
        """``True`` only when the Lean kernel genuinely re-checked the descent obligation."""
        return self.lean is not None and self.lean.verified


def certify_subspace_step(
    arch: MLPArchitecture,
    data: Data,
    theta0: Sequence[float],
    basis: Sequence[Sequence[float]],
    *,
    radius: float,
    l2: float = 0.0,
    step: Sequence[float] | None = None,
    lean: bool = False,
    provenance: dict[str, Any] | None = None,
) -> SubspaceStepCertificate:
    r"""Certify that a subspace trust-region step strictly decreases the true loss.

    Over the trust box ``[-radius, radius]^k`` this:

    1. encloses the exact degree-3 subspace Taylor model ``m(a)`` of
       ``psi(a) = J(theta0 + Q a)`` and its rigorous model-vs-truth remainder ``R``
       (:func:`enclose_subspace_model`);
    2. takes the step ``a*`` -- the given ``step``, else the Cauchy point
       ``-radius * c / ||c||`` -- and forms the predicted decrease
       ``pred = m(0) - m(a*)``;
    3. **certifies the true decrease** ``psi(0) - psi(a*)`` is enclosed by
       ``[pred - R.width, pred + R.width]`` (rigorous, because both evaluations
       share the uniform remainder ``R``); the step is certified iff the lower end
       is ``> 0``;
    4. **seals** two v1 certificates -- the remainder ``R`` and the one-signed
       decrease enclosure (so the ``> 0`` obligation is Lean-checkable) -- and
       optionally runs the Lean kernel (``lean=True``, only on a positive margin);
       :attr:`~SubspaceStepCertificate.theorem_prover_verified` is set only on a
       genuine kernel pass and degrades to ``False`` when no toolchain is present.

    Honest scope: a rigorous *local* proof about one step (a small net, a small
    subspace and a small trust radius); not a global-optimality claim.
    """
    if radius <= 0.0:
        raise ValueError(f"radius must be > 0, got {radius}")
    if l2 < 0.0:
        raise ValueError(f"l2 (weight decay) must be >= 0, got {l2}")
    theta = [float(t) for t in theta0]
    p_dim = arch.n_params
    if len(theta) != p_dim:
        raise ValueError(f"theta0 has {len(theta)} entries but the model has {p_dim} parameters")
    if len(basis) != p_dim:
        raise ValueError(f"basis has {len(basis)} rows but the model has {p_dim} parameters")
    k = len(basis[0])

    model = enclose_subspace_model(arch, data, theta, basis, radius=radius, l2=l2)

    if step is None:
        a_star = cauchy_step(tuple(g.mid for g in model.grad), radius)
    else:
        a_star = tuple(float(s) for s in step)
        if len(a_star) != k:
            raise ValueError(f"step has {len(a_star)} entries but the subspace has {k} dims")
        if any(abs(s) > radius * (1.0 + 1e-9) for s in a_star):
            raise ValueError("step lies outside the trust box [-radius, radius]^k")

    m_at_step = _poly_value_at(model, a_star)
    pred = model.constant - m_at_step
    # psi(0) in m(0) + R and psi(a*) in m(a*) + R at two independent points, so the
    # true decrease Delta = psi(0) - psi(a*) lies in pred + (R - R) = pred + [-w, w]
    # (outward-rounded via interval arithmetic).
    decrease_enclosure = pred + model.remainder - model.remainder
    predicted_decrease = pred.mid

    dims = list(arch.dims)
    model_claim = (
        "the exact degree-3 subspace Taylor model m(a) of the training loss agrees with the true "
        "loss psi(a) = L(theta0 + Q a) to within this interval for every ||a|| <= radius "
        "(the model-vs-truth remainder)"
    )
    model_meta: dict[str, Any] = {
        "kind": "subspace_model_remainder",
        "dims": dims,
        "activation": arch.activation,
        "subspace_dim": k,
        "radius": float(radius),
        "l2": float(l2),
        "remainder": [model.remainder.lo, model.remainder.hi],
    }
    model_cert = interval_certificate(model_claim, model.remainder, meta=model_meta)

    descent_claim = (
        "certified descent: the true training-loss decrease L(theta0) - L(theta0 + Q a*) over this "
        "subspace trust-region step is enclosed by the interval (a positive lower bound proves the "
        "step strictly decreases the true loss)"
    )
    descent_meta: dict[str, Any] = {
        "kind": "certified_subspace_step",
        "dims": dims,
        "activation": arch.activation,
        "n_params": p_dim,
        "subspace_dim": k,
        "radius": float(radius),
        "l2": float(l2),
        "theta0": list(theta),
        "step": list(a_star),
        "predicted_decrease": float(predicted_decrease),
        "remainder": [model.remainder.lo, model.remainder.hi],
        "decrease_enclosure": [decrease_enclosure.lo, decrease_enclosure.hi],
        "model_cert_digest": model_cert["digest"],
        "provenance": dict(provenance) if provenance is not None else {},
    }
    certified = decrease_enclosure.lo > 0.0
    honesty = {"unproven_claim": False, "certified_descent": bool(certified)}
    descent_cert = interval_certificate(descent_claim, decrease_enclosure, honesty=honesty, meta=descent_meta)

    lean_res = check_certificate(descent_cert) if (lean and certified) else None

    return SubspaceStepCertificate(
        theta0=tuple(theta),
        basis=tuple(tuple(float(v) for v in row) for row in basis),
        radius=float(radius),
        step=a_star,
        predicted_decrease=float(predicted_decrease),
        remainder=model.remainder,
        decrease_enclosure=decrease_enclosure,
        model_certificate=model_cert,
        descent_certificate=descent_cert,
        lean=lean_res,
    )


def certify_trajectory(
    arch: MLPArchitecture,
    data: Data,
    theta0: Sequence[float],
    *,
    radius: float,
    k: int,
    steps: int,
    l2: float = 0.0,
    lean: bool = False,
    shrink: float = 0.5,
    min_radius: float = 1e-9,
) -> list[SubspaceStepCertificate]:
    r"""A proof-carrying descent: a sequence of certified subspace trust-region steps.

    From ``theta0`` this repeatedly (i) builds a fresh ``k``-dimensional Krylov
    basis at the current parameters, (ii) certifies a subspace trust-region step,
    and (iii) *applies it only if certified* -- shrinking the radius by ``shrink``
    and retrying otherwise (a rigorous trust-region reduction).  Returns the list
    of accepted step certificates (length ``<= steps``); every entry satisfies
    ``cert.certified`` by construction, so the whole chain is a rigorously
    monotone-decreasing trajectory.
    """
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    if not 0.0 < shrink < 1.0:
        raise ValueError(f"shrink must be in (0, 1), got {shrink}")
    problem = ParamSpaceLoss(arch, data, l2=l2)
    p_dim = arch.n_params
    theta = [float(t) for t in theta0]
    if len(theta) != p_dim:
        raise ValueError(f"theta0 has {len(theta)} entries but the model has {p_dim} parameters")
    certs: list[SubspaceStepCertificate] = []
    r = float(radius)
    while len(certs) < steps and r >= min_radius:
        try:
            basis = krylov_basis(problem, theta, k)
        except ValueError:
            break  # (numerically) stationary: no descent subspace remains
        cert = certify_subspace_step(arch, data, theta, basis, radius=r, l2=l2, lean=lean)
        if cert.certified:
            kk = len(cert.step)
            theta = [
                theta[p] + math.fsum(basis[p][j] * cert.step[j] for j in range(kk))
                for p in range(p_dim)
            ]
            certs.append(cert)
        else:
            r *= shrink
    return certs


__all__ = [
    "SubspaceModel",
    "SubspaceStepCertificate",
    "cauchy_step",
    "certify_subspace_step",
    "certify_trajectory",
    "enclose_subspace_model",
    "krylov_basis",
]
