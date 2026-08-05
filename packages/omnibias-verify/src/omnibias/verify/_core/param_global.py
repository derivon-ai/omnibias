# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Proof-carrying training, globally: certify a *global* minimum of the training loss.

:func:`omnibias.verify.certify_trained_min` proves a trained ``theta*`` is a *local* strict
minimum (a ball around one point). This module asks the harder, global question over a whole
**parameter search box**: what is a rigorous enclosure of ``min_theta J(theta)`` -- and is a located
point the certified global minimizer to a tolerance?

It is the parameter-space twin of the input-space :func:`certify_trained_network`. The interval
branch-and-bound in :mod:`omnibias.verify._core.global_opt` already consumes exactly the interval
``value`` / ``grad`` / ``hessian`` enclosures that :class:`~omnibias.verify._core.param_loss.ParamSpaceLoss`
exposes, so the objective, monotonicity test, mean-value / second-order lower bounds, and the
interval-Newton (Krawczyk) contractor all come for free. The result is a sealed, tamper-evident
certificate whose payload is the rigorous global-minimum enclosure ``[f_lower, f_upper]``.

**Honest scope.** Interval B&B is *sound for any dimension* but its cost is exponential in the number
of parameters in the worst case, and here each explored box triggers an ``O(P^2)`` hyper-dual sweep
for the gradient / Hessian. This is therefore for **tiny networks** (a handful of parameters --
``1-1-1``, ``1-2-1``) over a bounded weight box, exactly like the input-space network certificate; it
is *not* million-parameter training. Within a finite ``max_boxes`` budget the enclosure is always
sound (``f_lower <= min J <= f_upper``); ``converged`` reports honestly whether the certified gap
reached ``tol``. The Lean gate fires only when the certified global minimum is bounded away from zero
(``f_lower > 0``) -- a meaningful, kernel-checkable claim that the architecture *cannot* fit the data
exactly.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from omnibias.core.proof.certificate import (
    Cert,
    interval_certificate,
    positive_definite_certificate,
    verify_certificate_digest,
)
from omnibias.core.proof.lean_check import LeanCheckResult, check_certificate
from omnibias.core.verified.eig_operator import interval_ldlt_pivots
from omnibias.core.verified.interval import Interval
from omnibias.verify._core.global_opt import (
    GlobalMinResult,
    certified_minimize,
)
from omnibias.verify._core.param_loss import MLPArchitecture, ParamSpaceLoss

Data = Sequence[tuple[Sequence[float], Sequence[float]]]


@dataclass(frozen=True)
class GlobalTrainingCertificate:
    r"""A sealed, tamper-evident enclosure of the *global* minimum of the training objective.

    Bundles the rigorous :class:`~omnibias.verify._core.global_opt.GlobalMinResult` (the
    ``[f_lower, f_upper]`` enclosure of ``min_theta J(theta)`` over the parameter box), the optional
    strict-local-min flag at the located argmin, the sealed v1
    :class:`~omnibias.core.proof.certificate.Cert` (payload = the minimum enclosure, box / argmin /
    gap in ``meta``), and the optional :class:`~omnibias.core.proof.lean_check.LeanCheckResult`.

    When the strict-local-min upgrade succeeds, ``pd_certificate`` additionally seals the interval
    ``LDL^T`` **pivot vector** of the Hessian at the located argmin -- the kernel-checkable matrix
    positive-definiteness sub-claim (``allPivotsPos`` inertia obligation), the parameter-space twin
    of the strict-min payload sealed by :func:`~omnibias.verify.certify_trained_min`.
    """

    result: GlobalMinResult
    certificate: Cert
    l2: float
    strict_local_min: bool | None = None
    lean: LeanCheckResult | None = None
    pd_certificate: Cert | None = None

    @property
    def verified(self) -> bool:
        """``True`` iff every sealed certificate's digest matches its body (untampered)."""
        if not verify_certificate_digest(self.certificate):
            return False
        return self.pd_certificate is None or verify_certificate_digest(self.pd_certificate)

    @property
    def positive_definite(self) -> bool:
        """``True`` iff the argmin Hessian was certified PD and its pivot vector was sealed."""
        return self.pd_certificate is not None

    @property
    def converged(self) -> bool:
        """``True`` iff the certified optimality gap ``f_upper - f_lower`` reached ``tol``."""
        return self.result.converged

    @property
    def certified(self) -> bool:
        """``True`` iff the global minimum is enclosed to the requested tolerance."""
        return self.result.converged

    @property
    def theorem_prover_verified(self) -> bool:
        """``True`` only when the Lean kernel genuinely re-checked the sealed obligation."""
        return self.lean is not None and self.lean.verified


def certify_trained_global_min(
    arch: MLPArchitecture,
    data: Data,
    param_bounds: Sequence[tuple[float, float]],
    *,
    tol: float = 1e-4,
    l2: float = 0.0,
    max_boxes: int = 100_000,
    use_newton: bool = True,
    strict_local_min: bool = False,
    strict_radius: float = 1e-3,
    lean: bool = False,
    provenance: dict[str, Any] | None = None,
) -> GlobalTrainingCertificate:
    r"""Certify a rigorous enclosure of the *global* minimum of the training objective.

    Over the parameter search box ``param_bounds`` (one ``(lo, hi)`` per parameter) this runs the
    certified interval branch-and-bound (:func:`certified_minimize`) on the objective
    ``J(theta) = L(theta) + l2 * ||theta||^2`` -- ``L`` the mean-squared error, ``l2 = 0`` the bare
    loss -- using the :class:`ParamSpaceLoss` interval ``value`` / ``grad`` / ``hessian`` enclosures
    for the monotonicity test, mean-value and second-order lower bounds, and (``use_newton``) the
    interval-Newton contractor. It then:

    1. **seals** the global-minimum enclosure ``[f_lower, f_upper]`` as a v1 certificate whose ``meta``
       records the box, the argmin, the certified gap, and the honest ``converged`` flag;
    2. optionally certifies the located argmin is a **strict** local minimum (``strict_local_min``):
       the Hessian is positive definite on a small ball ``B(argmin, strict_radius)`` via the interval
       ``LDL^T`` inertia -- which, together with the global gap, upgrades the answer to a certified
       strict global minimizer, and seals the pivot vector as a kernel-checkable ``pd_certificate``;
    3. optionally runs the Lean kernel (``lean=True``) -- but only when the certified minimum is
       bounded away from zero (``f_lower > 0``), i.e. the honest claim "this architecture cannot fit
       this data exactly"; :attr:`~GlobalTrainingCertificate.theorem_prover_verified` is set only on a
       genuine kernel pass and degrades gracefully with no toolchain.

    Honest scope: **tiny** networks over a bounded weight box (interval B&B is exponential in the
    parameter count in the worst case); this is a rigorous *global* proof for a small net with fixed
    data, not a open-problem claim.
    """
    if tol <= 0.0:
        raise ValueError(f"tol must be positive, got {tol}")
    if l2 < 0.0:
        raise ValueError(f"l2 (weight decay) must be >= 0, got {l2}")
    if strict_radius <= 0.0:
        raise ValueError(f"strict_radius must be > 0, got {strict_radius}")
    p_dim = arch.n_params
    bounds = [(float(lo), float(hi)) for lo, hi in param_bounds]
    if len(bounds) != p_dim:
        raise ValueError(f"param_bounds has {len(bounds)} entries but the model has {p_dim} parameters")
    for lo, hi in bounds:
        if lo >= hi:
            raise ValueError(f"each parameter bound needs lo < hi, got ({lo}, {hi})")

    problem = ParamSpaceLoss(arch, data, l2=l2)
    box = [Interval(lo, hi) for lo, hi in bounds]

    result = certified_minimize(
        problem.value,
        box,
        tol=tol,
        max_boxes=max_boxes,
        grad=problem.grad,
        hess=problem.hessian,
        use_newton=use_newton,
    )

    slm: bool | None = None
    slm_pivots: tuple[Interval, ...] | None = None
    slm_hbox: list[list[Interval]] | None = None
    if strict_local_min:
        slm_box = [Interval(xi - strict_radius, xi + strict_radius) for xi in result.x]
        slm_hbox = [[Interval.from_value(hij) for hij in row] for row in problem.hessian(slm_box)]
        slm_pivots = interval_ldlt_pivots(slm_hbox)
        # PD iff the factorisation succeeds and every pivot's lower endpoint is positive
        # (equivalent to is_positive_definite, but retaining the pivots for the sealed payload).
        slm = bool(slm_pivots is not None and all(p.lo > 0.0 for p in slm_pivots))

    objective = "L(theta)" if l2 == 0.0 else f"L(theta) + {float(l2)!r} * ||theta||^2"
    claim = (
        f"the global minimum min_{{theta in box}} {objective} of the training objective over the "
        "parameter box is enclosed by the interval"
    )
    honesty = {"unproven_claim": False, "global_min_certified": bool(result.converged)}
    meta: dict[str, Any] = {
        "kind": "global_training_min",
        "dims": list(arch.dims),
        "activation": arch.activation,
        "n_params": p_dim,
        "n_data": len(problem.data),
        "l2": float(l2),
        "box": [[iv.lo, iv.hi] for iv in box],
        "argmin": list(result.x),
        "enclosure": [result.f_lower, result.f_upper],
        "gap": result.gap,
        "tol": result.tol,
        "converged": result.converged,
        "boxes_explored": result.boxes_explored,
        "boxes_remaining": result.boxes_remaining,
        "provenance": dict(provenance) if provenance is not None else {},
    }
    if slm is not None:
        meta["strict_local_min"] = slm

    cert = interval_certificate(claim, result.enclosure, honesty=honesty, meta=meta)
    # Only Lean-check a genuinely positive, finite minimum enclosure (f_lower > 0): a global min that
    # includes 0 carries no positive scalar obligation for the kernel.
    lean_res = check_certificate(cert) if (lean and result.enclosure.lo > 0.0) else None

    # Seal the strict-min sub-claim's PD pivot vector (kernel-checkable matrix positive-definiteness).
    pd_cert: Cert | None = None
    if slm and slm_pivots is not None:
        pd_claim = (
            f"the objective Hessian of {objective} at the located global argmin (over "
            "B(argmin, strict_radius)) is positive definite: every interval LDL^T pivot is strictly "
            "positive (zero negative inertia), so the argmin is a strict local -- hence strict "
            "global-in-box -- minimizer"
        )
        pd_cert = positive_definite_certificate(
            pd_claim, slm_pivots, matrix=slm_hbox, honesty=honesty, meta=meta
        )

    return GlobalTrainingCertificate(
        result=result,
        certificate=cert,
        l2=float(l2),
        strict_local_min=slm,
        lean=lean_res,
        pd_certificate=pd_cert,
    )


__all__ = [
    "GlobalTrainingCertificate",
    "certify_trained_global_min",
]
