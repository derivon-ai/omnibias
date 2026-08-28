# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Enclosure Collapse product facade (theory 01-14).

The founding bias collapse (``delta -> 0``) yields a derivative.
Temperature collapse (``beta -> inf``, feasibility) yields a 0/1 step.
Enclosure Collapse is the ``width -> 0`` limit of a sound enclosure:
a point plus a proof, or ``Inconclusive``. Not a derivative and not a
0/1 step. Do not conflate the three.

Thin wrappers only. Do not reimplement Krawczyk, B&B, Lohner, or
remainder training.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from omnibias.core.proof.certificate import Cert, verify_certificate_digest
from omnibias.core.remainder_train import RemainderTrainConfig, remainder_loss
from omnibias.core.verified.enclosure_collapse import (
    RecommendedAction,
    diagnose_width,
)
from omnibias.core.verified.enclosure_collapse import (
    honesty_payload as algebra_honesty,
)
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.jet_flow import ValidatedRun, WidthBudget, seal_run
from omnibias.core.verified.jet_mv import Layer
from omnibias.core.verified.kantorovich import (
    CONTINUUM_PDE_CLAIM_KEY,
    IntervalJac,
    IntervalMap,
    kantorovich_accept_step,
    krawczyk_certificate,
    radii_polynomial_certificate,
)
from omnibias.core.verified.pde_certificate import (
    LinearPDE,
    StabilityEstimate,
    adaptive_certified_interior_residual,
    aposteriori_error_certificate,
    certified_interior_residual,
)
from omnibias.verify._core.localization import (
    Inconclusive,
    LocalizationCertificate,
    ScanResponse,
    certify_peak,
)
from omnibias.verify._core.localization import (
    honesty_payload as peak_honesty,
)
from omnibias.verify._core.localization import (
    seal as seal_peak,
)
from omnibias.verify._core.param_global import (
    GlobalTrainingCertificate,
    certify_trained_global_min,
)
from omnibias.verify._core.param_loss import MLPArchitecture

SqueezeKind = Literal["peak", "residual", "identifiability", "existence", "remainder", "flow"]
SqueezeStatus = Literal["certified", "inconclusive", "rejected"]
Data = Sequence[tuple[Sequence[float], Sequence[float]]]
BoxLike = Sequence[tuple[float, float] | Interval]


@dataclass(frozen=True)
class SqueezeReport:
    """Unified squeeze result. Inner engines keep their own return types."""

    kind: SqueezeKind
    status: SqueezeStatus
    width: float | None
    budget: WidthBudget | None
    action: RecommendedAction | None
    scope: str
    honesty: dict[str, object]
    certificate: Cert | None
    inner: object


def _base_honesty() -> dict[str, object]:
    payload = dict(algebra_honesty())
    payload["theorem_prover_verified"] = False
    payload["mathlib_verified"] = False
    return payload


def squeeze_peak(
    response: ScanResponse,
    *,
    box: Interval,
    max_iter: int = 20,
) -> SqueezeReport:
    """Wrap ``certify_peak`` (03-08). Flat peaks stay ``Inconclusive``."""
    inner: LocalizationCertificate | Inconclusive = certify_peak(
        response, box=box, max_iter=max_iter
    )
    honesty = _base_honesty()
    honesty.update(peak_honesty())
    honesty["scope"] = "local_box"
    if isinstance(inner, Inconclusive):
        return SqueezeReport(
            "peak",
            "inconclusive",
            inner.box.width,
            None,
            None,
            "local_box",
            honesty,
            None,
            inner,
        )
    sealed = seal_peak(inner)
    return SqueezeReport(
        "peak",
        "certified",
        inner.offset_enclosure.width,
        None,
        None,
        "local_box",
        honesty,
        sealed,
        inner,
    )


def squeeze_residual(
    layers: Sequence[Layer],
    domain: BoxLike,
    pde: LinearPDE,
    *,
    until: Literal["ulps"] | None = None,
    width_cap: float | None = None,
    splits: int | Sequence[int] | None = None,
    max_splits: int = 64,
    aposteriori: bool = False,
    stability: StabilityEstimate | None = None,
) -> SqueezeReport:
    """Wrap certified interior residual; adapt splits until a width cap."""
    honesty = _base_honesty()
    honesty.update(
        {
            "scope": "model_problem",
            "continuum_navier_stokes_claim": False,
            "interval_verified": True,
            "fft_evidence": False,
        }
    )
    typed_layers = layers  # Layer format from pde_certificate
    if until is None and width_cap is None and splits is not None:
        residual = certified_interior_residual(typed_layers, domain, pde, splits=splits)
        inner: object = residual
        width = residual.width
        status: SqueezeStatus = "certified"
        cert: Cert | None = None
        if aposteriori:
            if stability is None:
                raise ValueError(
                    "aposteriori=True requires a provenance-carrying "
                    "StabilityEstimate"
                )
            result = aposteriori_error_certificate(
                typed_layers, domain, pde, splits=splits, stability=stability
            )
            inner = result
            width = result.interior_residual
            cert = result.certificate
        return SqueezeReport(
            "residual", status, width, None, None, "model_problem", honesty, cert, inner
        )

    target = width_cap
    if until == "ulps" and target is None:
        target = 8.0 * math.ulp(1.0)
    initial = 1 if splits is None else splits
    if isinstance(initial, Sequence) and not isinstance(initial, str | bytes):
        start: int | Sequence[int] = initial
    else:
        start = int(initial)
    diag = adaptive_certified_interior_residual(
        typed_layers,
        domain,
        pde,
        target=target,
        initial_splits=start,
        max_splits=int(max_splits),
    )
    status = "certified" if diag.reached_target else "inconclusive"
    return SqueezeReport(
        "residual",
        status,
        diag.residual.width,
        None,
        None,
        "model_problem",
        honesty,
        None,
        diag,
    )


def squeeze_identifiability(
    arch: MLPArchitecture,
    data: Data,
    param_bounds: Sequence[tuple[float, float]],
    *,
    tol: float = 1e-4,
    l2: float = 0.0,
    max_boxes: int = 100_000,
    use_newton: bool = True,
    lean: bool = False,
) -> SqueezeReport:
    """First-class product wrapping ``certify_trained_global_min``."""
    inner: GlobalTrainingCertificate = certify_trained_global_min(
        arch,
        data,
        param_bounds,
        tol=tol,
        l2=l2,
        max_boxes=max_boxes,
        use_newton=use_newton,
        lean=lean,
    )
    honesty = _base_honesty()
    honesty.update(
        {
            "scope": "parameter_box",
            "tiny_network": True,
            "p_vs_np_claim": False,
            "theorem_prover_verified": bool(inner.theorem_prover_verified),
        }
    )
    status: SqueezeStatus
    if inner.converged:
        status = "certified"
    else:
        status = "inconclusive"
    return SqueezeReport(
        "identifiability",
        status,
        float(inner.result.gap),
        None,
        None,
        "parameter_box",
        honesty,
        inner.certificate,
        inner,
    )


def squeeze_existence(
    *,
    y0: float | None = None,
    z0: float | None = None,
    z1: float | None = None,
    z2: float | None = None,
    func: IntervalMap | None = None,
    jacobian: IntervalJac | None = None,
    a_inv: Sequence[Sequence[float]] | None = None,
    trial_params: Sequence[float] | None = None,
    lipschitz_df: float | None = None,
    r_max: float = 1.0,
    x_bar: Sequence[float] | None = None,
    r: float | None = None,
) -> SqueezeReport:
    """Wrap radii / Krawczyk / Kantorovich accept. Empty ball is a reject."""
    honesty = _base_honesty()
    honesty.update(
        {
            "scope": "local_box",
            CONTINUUM_PDE_CLAIM_KEY: False,
            "continuum_pde_claim": False,
        }
    )
    if y0 is not None and z0 is not None and z1 is not None and z2 is not None:
        hit = radii_polynomial_certificate(y0, z0, z1, z2, r_max=r_max)
        if hit is None:
            return SqueezeReport(
                "existence", "rejected", None, None, None, "local_box", honesty, None, None
            )
        return SqueezeReport(
            "existence",
            "certified",
            float(hit.radius),
            None,
            None,
            "local_box",
            honesty,
            hit.certificate,
            hit,
        )
    if trial_params is not None:
        if func is None or jacobian is None or a_inv is None or lipschitz_df is None:
            raise ValueError("accept_step existence needs func, jacobian, a_inv, lipschitz_df")
        decision = kantorovich_accept_step(
            func,
            jacobian,
            a_inv,
            trial_params,
            lipschitz_df=lipschitz_df,
            r_max=r_max,
        )
        if not decision.accepted:
            return SqueezeReport(
                "existence",
                "rejected",
                None,
                None,
                None,
                "local_box",
                honesty,
                None,
                decision,
            )
        cert = None if decision.certificate is None else decision.certificate.certificate
        width = None if decision.certificate is None else float(decision.certificate.radius)
        return SqueezeReport(
            "existence", "certified", width, None, None, "local_box", honesty, cert, decision
        )
    if x_bar is not None and func is not None and jacobian is not None and a_inv is not None:
        radius = 0.1 if r is None else float(r)
        hit_k = krawczyk_certificate(func, jacobian, x_bar, a_inv, radius)
        if hit_k is None:
            return SqueezeReport(
                "existence", "rejected", None, None, None, "local_box", honesty, None, None
            )
        return SqueezeReport(
            "existence",
            "certified",
            float(hit_k.radius),
            None,
            None,
            "local_box",
            honesty,
            hit_k.certificate,
            hit_k,
        )
    raise ValueError("squeeze_existence needs radii bounds, accept_step args, or a Krawczyk box")


def squeeze_remainder(
    values: Sequence[float],
    jet: Sequence[float],
    xs: Sequence[float],
    x0: float = 0.0,
    *,
    config: RemainderTrainConfig | None = None,
) -> SqueezeReport:
    """Tag ``R_N`` as the truncation piece of a ``WidthBudget``."""
    inner = remainder_loss(values, jet, xs, x0, config=config)
    trunc = float(inner["max_abs"])
    budget = WidthBudget(truncation=trunc, jacobian=0.0, wrapping=0.0, rounding=0.0)
    honesty = _base_honesty()
    honesty.update(
        {
            "scope": "model_problem",
            "is_03_10": False,
            "is_03_13": False,
            "stretch_claim": False,
        }
    )
    return SqueezeReport(
        "remainder",
        "certified",
        trunc,
        budget,
        diagnose_width(budget),
        "model_problem",
        honesty,
        None,
        inner,
    )


def squeeze_flow(run: ValidatedRun) -> SqueezeReport:
    """Diagnose an existing Lohner ``WidthBudget``. Does not fork the type."""
    honesty = _base_honesty()
    honesty.update(
        {
            "scope": "finite_horizon",
            "continuum_existence_claim": False,
            "navier_stokes_regularity_claim": False,
        }
    )
    sealed = seal_run(run)
    return SqueezeReport(
        "flow",
        "certified",
        run.width,
        run.budget,
        diagnose_width(run.budget),
        "finite_horizon",
        honesty,
        sealed,
        run,
    )


def squeeze(kind: SqueezeKind, **kwargs: Any) -> SqueezeReport:
    """Dispatch to the named first-class ``squeeze_*`` path."""
    drivers: Mapping[SqueezeKind, Callable[..., SqueezeReport]] = {
        "peak": squeeze_peak,
        "residual": squeeze_residual,
        "identifiability": squeeze_identifiability,
        "existence": squeeze_existence,
        "remainder": squeeze_remainder,
        "flow": squeeze_flow,
    }
    if kind not in drivers:
        raise ValueError(f"unknown SqueezeKind {kind!r}")
    return drivers[kind](**kwargs)


def report_digest_ok(report: SqueezeReport) -> bool:
    if report.certificate is None:
        return True
    return verify_certificate_digest(report.certificate)


__all__ = [
    "SqueezeKind",
    "SqueezeReport",
    "SqueezeStatus",
    "report_digest_ok",
    "squeeze",
    "squeeze_existence",
    "squeeze_flow",
    "squeeze_identifiability",
    "squeeze_peak",
    "squeeze_remainder",
    "squeeze_residual",
]
