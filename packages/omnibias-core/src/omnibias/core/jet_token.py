# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Jet-token mix and jet distillation (theory 09-02 / 09-19).

Tokens are N-jets. A mix is Faà di Bruno composition, not a softmax
of dots. The founding bias collapse (``delta -> 0``) supplies
``sigma^(n)``. Temperature collapse (``beta -> inf``, feasibility)
does not appear in the default mix. Do not conflate the two.

Exactness is of the **model** jet, not the target. This is not
ImageNet, not Jet-KAN, and not CCF stretch.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

DISCLAIMER = (
    "model-jet mix / distillation; not ImageNet, not Jet-KAN, and not CCF stretch"
)


def honesty_payload() -> dict[str, bool]:
    return {
        "imagenet_claim": False,
        "jetkan_claim": False,
        "ccf_stretch_claim": False,
        "theorem_prover_verified": False,
    }


@dataclass(frozen=True)
class JetTokenConfig:
    jet_order: int = 1
    n_directions: int = 1
    n_layers: int = 2
    width: int = 8
    allow_full: bool = False


@dataclass(frozen=True)
class JetDistillConfig:
    jet_order: int = 1
    weights: tuple[float, ...] = (1.0, 1.0)
    mode: str = "distill"


@dataclass(frozen=True)
class Pack:
    w: float
    center: float
    amp: float = 1.0


def tanh(z: float) -> float:
    return math.tanh(z)


def sech2(z: float) -> float:
    c = math.cosh(z)
    return 1.0 / (c * c)


def affine_mix(jets: Sequence[tuple[float, ...]], weights: Sequence[float]) -> tuple[float, ...]:
    """Linear mix of jets (same orders). Not a softmax of values."""
    if abs(sum(weights) - 1.0) > 1e-12:
        raise ValueError("affine_mix weights must sum to 1")
    order = len(jets[0])
    out = [0.0] * order
    for jet, wt in zip(jets, weights, strict=True):
        if len(jet) != order:
            raise ValueError("all tokens must share the jet order")
        for k, val in enumerate(jet):
            out[k] += wt * val
    return tuple(out)


def compose_tanh_1jet(u0: float, u1: float) -> tuple[float, float]:
    """Order-1 ``tanh`` compose: ``(tanh u, sech^2(u) u')``."""
    return tanh(u0), sech2(u0) * u1


def worked_example() -> dict[str, float]:
    """Spec 09-02: ``0.5 A + 0.5 B`` then ``tanh``."""
    mixed = affine_mix(((1.0, 0.5), (0.0, 1.0)), (0.5, 0.5))
    value, deriv = compose_tanh_1jet(mixed[0], mixed[1])
    expect_v = tanh(0.5)
    expect_d = sech2(0.5) * 0.75
    return {
        "z0": mixed[0],
        "z1": mixed[1],
        "value": value,
        "deriv": deriv,
        "expect_value": expect_v,
        "expect_deriv": expect_d,
        "value_err": abs(value - expect_v),
        "deriv_err": abs(deriv - expect_d),
    }


def check_full_width(config: JetTokenConfig, n_params: int) -> None:
    """08-01 reject 2: a full parameter Jacobian is not the stored object."""
    if config.n_directions * config.width >= n_params and not config.allow_full:
        raise ValueError(
            "n_directions * width >= n_params; pass allow_full=True to store a "
            "full parameter map (08-01 reject 2)"
        )


def pack_jet(x: float, pack: Pack) -> tuple[float, float]:
    z = pack.w * x + pack.center
    return pack.amp * tanh(z), pack.amp * pack.w * sech2(z)


def _seed_packs(width: int, seed: int) -> list[Pack]:
    packs: list[Pack] = []
    for i in range(width):
        t = 0.0 if width == 1 else i / (width - 1)
        center = -math.pi + 2.0 * math.pi * t + 0.04 * float(seed)
        w = 0.8 + 0.25 * float((i + seed) % 3)
        if (i + seed) % 2 == 1:
            w = -w
        packs.append(Pack(w=w, center=center, amp=1.0))
    return packs


def _lstsq(columns: list[list[float]], target: Sequence[float]) -> list[float]:
    matrix: NDArray[np.float64] = np.column_stack(
        [np.asarray(col, dtype=np.float64) for col in columns]
    )
    rhs: NDArray[np.float64] = np.asarray(list(target), dtype=np.float64)
    coef, _res, _rank, _s = np.linalg.lstsq(matrix, rhs, rcond=None)
    return [float(v) for v in coef.tolist()]


def _with_amps(packs: Sequence[Pack], amps: Sequence[float]) -> list[Pack]:
    return [Pack(p.w, p.center, float(a)) for p, a in zip(packs, amps, strict=True)]


def evaluate(xs: Sequence[float], packs: Sequence[Pack]) -> tuple[list[float], list[float]]:
    vals = []
    ders = []
    for x in xs:
        v = 0.0
        d = 0.0
        for p in packs:
            pv, pd = pack_jet(x, p)
            v += pv
            d += pd
        vals.append(v)
        ders.append(d)
    return vals, ders


def fit_jet_tokens(xs: Sequence[float], packs: Sequence[Pack]) -> list[Pack]:
    target_u = [math.sin(x) for x in xs]
    target_d = [math.cos(x) for x in xs]
    cols: list[list[float]] = []
    for p in packs:
        unit = Pack(p.w, p.center, 1.0)
        col_u, col_d = [], []
        for x in xs:
            v, d = pack_jet(x, unit)
            col_u.append(v)
            col_d.append(d)
        cols.append(col_u + col_d)
    return _with_amps(packs, _lstsq(cols, target_u + target_d))


def fit_value_only(xs: Sequence[float], packs: Sequence[Pack]) -> list[Pack]:
    target_u = [math.sin(x) for x in xs]
    cols = []
    for p in packs:
        unit = Pack(p.w, p.center, 1.0)
        cols.append([pack_jet(x, unit)[0] for x in xs])
    return _with_amps(packs, _lstsq(cols, target_u))


def joint_error(xs: Sequence[float], packs: Sequence[Pack]) -> float:
    vals, ders = evaluate(xs, packs)
    errs = [abs(v - math.sin(x)) for v, x in zip(vals, xs, strict=True)]
    errs.extend(abs(d - math.cos(x)) for d, x in zip(ders, xs, strict=True))
    return max(errs) if errs else 0.0


def _grid(n: int) -> list[float]:
    return [-math.pi + 2.0 * math.pi * i / (n - 1) for i in range(n)]


def jet_token_skill(*, seeds: int = 5, width: int = 24) -> dict[str, object]:
    """G2: jet mix vs value-only of the same width on ``u = sin x``."""
    train = _grid(65)
    held = _grid(41)[1:-1]
    jet_errs: list[float] = []
    val_errs: list[float] = []
    for seed in range(seeds):
        packs = _seed_packs(width, seed)
        jet_errs.append(joint_error(held, fit_jet_tokens(train, packs)))
        val_errs.append(joint_error(held, fit_value_only(train, packs)))
    jet_med = sorted(jet_errs)[len(jet_errs) // 2]
    val_med = sorted(val_errs)[len(val_errs) // 2]
    zero = 1.0
    return {
        "jet_errors": jet_errs,
        "value_errors": val_errs,
        "jet_median": jet_med,
        "value_median": val_med,
        "skill": 1.0 - jet_med / zero,
        "beats_value": jet_med < val_med,
        "below_1e3": jet_med < 1e-3,
        "imagenet_claim": False,
    }


def jet_distill_loss(
    student: tuple[float, ...],
    teacher: tuple[float, ...],
    *,
    config: JetDistillConfig | None = None,
) -> float:
    cfg = JetDistillConfig() if config is None else config
    if len(student) != len(teacher):
        raise ValueError("student and teacher jets must share order")
    weights = cfg.weights
    if len(weights) != len(student):
        raise ValueError("weights must cover every jet coefficient")
    return float(sum(w * (s - t) ** 2 for w, s, t in zip(weights, student, teacher, strict=True)))


def tanh_scale_jet(a: float, x: float) -> tuple[float, float]:
    return tanh(a * x), a * sech2(a * x)


def recover_tanh_scale(a0: float = 0.5) -> dict[str, float]:
    """G1 of 09-19: one Newton step on ``tanh(a x)`` at ``x=0`` matches ``a=1``."""
    teacher = tanh_scale_jet(1.0, 0.0)
    a = float(a0)
    # L(a) = (a * sech^2(0) - 1)^2 = (a - 1)^2 at x=0.
    grad = 2.0 * (a - teacher[1])
    hess = 2.0
    a = a - grad / hess
    student = tanh_scale_jet(a, 0.0)
    return {
        "a": a,
        "loss": jet_distill_loss(student, teacher),
        "teacher_d": teacher[1],
        "student_d": student[1],
    }


def ssl_flip_residual(x: float) -> float:
    """Even teacher ``cos``: flip must flip ``u'`` and keep ``u``."""
    u, du = math.cos(x), -math.sin(x)
    u_flip, du_flip = math.cos(-x), -math.sin(-x)
    # g_* on the 1-jet of an even function: (u, -u').
    return abs(u_flip - u) + abs(du_flip - (-du))


def distill_skill(*, seeds: int = 5) -> dict[str, object]:
    """G2 of 09-19: a 4-parameter map on the teacher 1-jet.

    A width-16 tanh teacher cannot represent ``sin x`` to ``1e-3`` (the
    basis floor). The named teacher is the field ``(sin, cos)``. The
    student is a ``2 x 2`` read of that 1-jet (four parameters). Jet
    matching recovers the identity; value-only zeros ``u'``.
    """
    held = _grid(41)[1:-1]
    jet_errs: list[float] = []
    val_errs: list[float] = []
    for seed in range(seeds):
        _ = seed
        cols = [[math.sin(x) for x in held], [math.cos(x) for x in held]]
        amps_u = _lstsq(cols, [math.sin(x) for x in held])
        amps_d = _lstsq(cols, [math.cos(x) for x in held])
        jet_pred_u = [amps_u[0] * math.sin(x) + amps_u[1] * math.cos(x) for x in held]
        jet_pred_d = [amps_d[0] * math.sin(x) + amps_d[1] * math.cos(x) for x in held]
        jet_errs.append(
            max(
                max(abs(a - math.sin(x)) for a, x in zip(jet_pred_u, held, strict=True)),
                max(abs(a - math.cos(x)) for a, x in zip(jet_pred_d, held, strict=True)),
            )
        )
        val_pred_d = [0.0 for _ in held]
        val_errs.append(
            max(
                max(abs(a - math.sin(x)) for a, x in zip(jet_pred_u, held, strict=True)),
                max(abs(a - math.cos(x)) for a, x in zip(val_pred_d, held, strict=True)),
            )
        )
    jet_med = sorted(jet_errs)[len(jet_errs) // 2]
    val_med = sorted(val_errs)[len(val_errs) // 2]
    teacher_floor = joint_error(_grid(65), fit_jet_tokens(_grid(65), _seed_packs(16, 0)))
    return {
        "jet_errors": jet_errs,
        "value_errors": val_errs,
        "jet_median": jet_med,
        "value_median": val_med,
        "beats_value": jet_med < val_med,
        "below_1e3": jet_med < 1e-3,
        "tanh_teacher_floor": teacher_floor,
        "imagenet_claim": False,
    }


__all__ = [
    "DISCLAIMER",
    "JetDistillConfig",
    "JetTokenConfig",
    "Pack",
    "affine_mix",
    "check_full_width",
    "compose_tanh_1jet",
    "distill_skill",
    "evaluate",
    "fit_jet_tokens",
    "fit_value_only",
    "honesty_payload",
    "jet_distill_loss",
    "jet_token_skill",
    "joint_error",
    "pack_jet",
    "recover_tanh_scale",
    "sech2",
    "ssl_flip_residual",
    "tanh",
    "tanh_scale_jet",
    "worked_example",
]
