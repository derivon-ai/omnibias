# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Recommended 08-01 trainer stack on a one-layer closed-form loss jet.

Call order (theory 08-01 §4):

1. Gauss-Newton / Newton direction from the closed-form Hessian.
2. Exact jet line search (03-12) along that direction.
3. Kantorovich accept / reject (08-04) on the 1-D stationarity map
   ``F(s) = φ'(s)``.
4. Sharpness (08-06) as extra cubic damping from ``λ_max(H)``.

08-02 composed curvature is a two-layer slice rule and is **not**
invoked here (``used_composed`` stays false). Faà di Bruno remains the
chain rule that forms the jet. Not a global min of a deep nest, not a
full ``d h / d θ``, not CCF stretch. Founding bias collapse
(``delta -> 0``) supplies ``σ^(n)``. No temperature collapse.
``theorem_prover_verified`` is not asserted.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from omnibias.core.composed_curvature import eigh_symmetric
from omnibias.core.line_search import (
    JetLineSearchConfig,
    LineSearchResult,
    poly_derivative,
    run_model_line_search,
    taylor_coeffs_from_derivatives,
)
from omnibias.core.sharpness import SharpnessSchedule, scheduled_value
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.kantorovich import (
    KantorovichAccept,
    kantorovich_accept_step,
    select_accepted_params,
)
from omnibias.core.weight_loss_jet import (
    one_layer_forward,
    one_layer_loss,
    one_layer_loss_grad,
    one_layer_loss_hessian,
    one_layer_loss_jet,
    one_layer_newton_direction,
    one_layer_param_count,
)

DISCLAIMER = (
    "08-01 recommended stack on a one-layer closed-form loss jet; "
    "not a global min, not a skip of the chain rule, not CCF stretch"
)


def honesty_payload() -> dict[str, bool]:
    return {
        "closed_form": True,
        "one_layer_only": True,
        "used_composed_by_default": False,
        "skip_chain_rule": False,
        "full_parameter_jacobian": False,
        "global_min_claim": False,
        "stretch_claim": False,
        "continuum_claim": False,
        "theorem_prover_verified": False,
    }


@dataclass(frozen=True)
class TrainStackConfig:
    """Knobs for one recommended-stack step."""

    damping: float = 1e-4
    jet_order: int = 3
    activation: str = "tanh"
    use_line_search: bool = True
    use_kantorovich: bool = True
    use_sharpness: bool = True
    sharpness: SharpnessSchedule = SharpnessSchedule(c=1e-3, target="cubic_sigma")
    line_search: JetLineSearchConfig = JetLineSearchConfig(
        order=3,
        trust_radius=1.0,
        verify=True,
        max_step=1.0,
    )
    kantorovich_r_max: float = 1.0
    max_params: int = 256

    def __post_init__(self) -> None:
        if self.damping < 0.0 or not math.isfinite(self.damping):
            raise ValueError(f"damping must be a finite number >= 0, got {self.damping}")
        if int(self.jet_order) < 2:
            raise ValueError(f"jet_order must be >= 2, got {self.jet_order}")
        if self.kantorovich_r_max <= 0.0:
            raise ValueError(
                f"kantorovich_r_max must be > 0, got {self.kantorovich_r_max}"
            )
        if int(self.line_search.order) > int(self.jet_order):
            raise ValueError(
                f"line_search.order={self.line_search.order} exceeds "
                f"jet_order={self.jet_order}"
            )


@dataclass(frozen=True)
class TrainStackReport:
    """Diagnostics of one recommended-stack step."""

    accepted: bool
    reason: str
    step: float
    loss0: float
    loss1: float
    damping: float
    lambda_min: float
    lambda_max: float
    used_composed: bool
    used_line_search: bool
    used_kantorovich: bool
    used_sharpness: bool
    line_search: LineSearchResult | None
    kantorovich: KantorovichAccept | None


def _add(
    theta: Sequence[float],
    direction: Sequence[float],
    scale: float,
) -> list[float]:
    return [
        float(p) + scale * float(d)
        for p, d in zip(theta, direction, strict=True)
    ]


def _restriction_accept(
    derivatives: Sequence[float],
    trial_step: float,
    *,
    r_max: float,
) -> KantorovichAccept:
    """Kantorovich on the model ``F(s) = φ'(s)`` (1-D stationarity)."""
    coeffs = taylor_coeffs_from_derivatives(derivatives)
    d_coeffs = poly_derivative(coeffs)
    dd_coeffs = poly_derivative(d_coeffs)
    ddd_coeffs = poly_derivative(dd_coeffs)
    lip = 0.0
    for idx, coeff in enumerate(ddd_coeffs):
        lip += abs(float(coeff)) * (max(r_max, 1.0) ** idx)

    def func(xs: list[Interval]) -> list[Interval]:
        acc = Interval.point(0.0)
        for coeff in reversed(d_coeffs):
            acc = acc * xs[0] + Interval.point(float(coeff))
        return [acc]

    def jacobian(xs: list[Interval]) -> list[list[Interval]]:
        acc = Interval.point(0.0)
        for coeff in reversed(dd_coeffs):
            acc = acc * xs[0] + Interval.point(float(coeff))
        return [[acc]]

    second = 0.0
    power = 1.0
    for idx, coeff in enumerate(dd_coeffs):
        if idx:
            power *= trial_step
        second += float(coeff) * power
    if abs(second) < 1e-18:
        return KantorovichAccept(False, None, "bounds_failed")
    a_inv = ((1.0 / second,),)
    return kantorovich_accept_step(
        func,
        jacobian,
        a_inv,
        (trial_step,),
        lipschitz_df=lip,
        r_max=r_max,
    )


def recommended_stack_step(
    xs: Sequence[Sequence[float]],
    ys: Sequence[float],
    theta: Sequence[float],
    hidden: int,
    dim: int,
    *,
    config: TrainStackConfig | None = None,
) -> tuple[list[float], TrainStackReport]:
    """One 08-01 stack step on a one-layer Riccati MSE loss."""
    cfg = config if config is not None else TrainStackConfig()
    need = one_layer_param_count(hidden, dim)
    if len(theta) != need:
        raise ValueError(f"theta length {len(theta)} != P={need}")
    loss0 = one_layer_loss(xs, ys, theta, hidden, dim, cfg.activation)
    grad = one_layer_loss_grad(xs, ys, theta, hidden, dim, cfg.activation)
    hess = one_layer_loss_hessian(
        xs, ys, theta, hidden, dim, cfg.activation, max_params=cfg.max_params
    )
    evals, _vecs = eigh_symmetric(hess)
    lambda_min = float(evals[0])
    lambda_max = float(evals[-1])
    damping = float(cfg.damping)
    if cfg.use_sharpness:
        damping = max(damping, scheduled_value(lambda_max, cfg.sharpness))
    if lambda_min < 0.0:
        # Regularize an indefinite Hessian so Newton is a descent direction.
        damping = max(damping, -lambda_min + 1e-6)
    direction = one_layer_newton_direction(grad, hess, damping=damping)
    if sum(float(g) * float(d) for g, d in zip(grad, direction, strict=True)) >= 0.0:
        direction = [-float(g) for g in grad]

    step = 1.0
    ls_result: LineSearchResult | None = None
    if cfg.use_line_search:
        jet = one_layer_loss_jet(
            xs,
            ys,
            theta,
            direction,
            order=cfg.jet_order,
            hidden=hidden,
            dim=dim,
            activation=cfg.activation,
        )

        def actual(scale: float) -> float:
            return one_layer_loss(
                xs, ys, _add(theta, direction, scale), hidden, dim, cfg.activation
            )

        ls_result = run_model_line_search(
            jet,
            config=cfg.line_search,
            actual_fn=actual if cfg.line_search.verify else None,
        )
        step = float(ls_result.step)
    trial = _add(theta, direction, step)

    decision: KantorovichAccept | None = None
    accepted = True
    reason = "step"
    if cfg.use_kantorovich:
        jet_k = one_layer_loss_jet(
            xs,
            ys,
            theta,
            direction,
            order=cfg.jet_order,
            hidden=hidden,
            dim=dim,
            activation=cfg.activation,
        )
        decision = _restriction_accept(
            jet_k, step, r_max=cfg.kantorovich_r_max
        )
        chosen_s = select_accepted_params((0.0,), (step,), decision)[0]
        trial = _add(theta, direction, chosen_s)
        step = float(chosen_s)
        accepted = bool(decision.accepted)
        reason = decision.reason

    loss1 = one_layer_loss(xs, ys, trial, hidden, dim, cfg.activation)
    if accepted and loss1 > loss0:
        # Never-worse backstop even if Kantorovich accepted a model ball.
        trial = [float(v) for v in theta]
        step = 0.0
        accepted = False
        reason = "never_worse"
        loss1 = loss0
    report = TrainStackReport(
        accepted=accepted,
        reason=reason,
        step=step,
        loss0=loss0,
        loss1=loss1,
        damping=damping,
        lambda_min=lambda_min,
        lambda_max=lambda_max,
        used_composed=False,
        used_line_search=cfg.use_line_search,
        used_kantorovich=cfg.use_kantorovich,
        used_sharpness=cfg.use_sharpness,
        line_search=ls_result,
        kantorovich=decision,
    )
    return trial, report


def stack_minimize(
    xs: Sequence[Sequence[float]],
    ys: Sequence[float],
    theta: Sequence[float],
    hidden: int,
    dim: int,
    *,
    steps: int = 8,
    config: TrainStackConfig | None = None,
) -> tuple[list[float], list[TrainStackReport]]:
    """Apply :func:`recommended_stack_step` a fixed number of times."""
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    current = [float(v) for v in theta]
    reports: list[TrainStackReport] = []
    for _ in range(steps):
        current, report = recommended_stack_step(
            xs, ys, current, hidden, dim, config=config
        )
        reports.append(report)
    return current, reports


def gradient_descent_step(
    xs: Sequence[Sequence[float]],
    ys: Sequence[float],
    theta: Sequence[float],
    hidden: int,
    dim: int,
    *,
    lr: float,
    activation: str = "tanh",
) -> list[float]:
    """Named first-order baseline (not the stack)."""
    if lr <= 0.0:
        raise ValueError(f"lr must be > 0, got {lr}")
    grad = one_layer_loss_grad(xs, ys, theta, hidden, dim, activation)
    return [float(p) - lr * float(g) for p, g in zip(theta, grad, strict=True)]


def skill_vs_zero(
    loss: float,
    ys: Sequence[float],
) -> float:
    """``1 - L / mean(y^2)``. Positive means the fit beats the zero predictor."""
    denom = sum(float(y) * float(y) for y in ys) / float(len(ys))
    if denom <= 0.0:
        raise ValueError("zero-predictor energy must be > 0")
    return 1.0 - float(loss) / denom


def worked_example() -> dict[str, bool | float]:
    """Teacher/student identities for the cookbook snippet."""
    hidden, dim = 1, 1
    xs = ((0.4,), (-0.3,), (0.8,), (-0.6,))
    teacher = [0.0, 1.2, 0.1, 0.7]
    ys = tuple(one_layer_forward(x, teacher, hidden, dim, "tanh") for x in xs)
    student = [0.05, 0.9, 0.0, 0.4]
    cfg = TrainStackConfig(
        use_kantorovich=False,
        use_sharpness=True,
        line_search=JetLineSearchConfig(
            order=3, trust_radius=1.0, verify=True, max_step=1.0
        ),
    )
    _theta1, report = recommended_stack_step(
        xs, ys, student, hidden, dim, config=cfg
    )
    fitted, reports = stack_minimize(
        xs, ys, student, hidden, dim, steps=6, config=cfg
    )
    final = one_layer_loss(xs, ys, fitted, hidden, dim, "tanh")
    start = one_layer_loss(xs, ys, student, hidden, dim, "tanh")
    gd = [float(v) for v in student]
    for _ in range(6):
        gd = gradient_descent_step(xs, ys, gd, hidden, dim, lr=0.15)
    gd_loss = one_layer_loss(xs, ys, gd, hidden, dim, "tanh")
    return {
        "first_step_never_worse": report.loss1 <= report.loss0 + 1e-15,
        "used_composed": report.used_composed,
        "beat_start": final < start,
        "beat_gd": final <= gd_loss + 1e-12,
        "skill_positive": skill_vs_zero(final, ys) > 0.0,
        "start_loss": start,
        "final_loss": final,
        "gd_loss": gd_loss,
        "n_reports": float(len(reports)),
    }


__all__ = [
    "DISCLAIMER",
    "TrainStackConfig",
    "TrainStackReport",
    "gradient_descent_step",
    "honesty_payload",
    "recommended_stack_step",
    "skill_vs_zero",
    "stack_minimize",
    "worked_example",
]
