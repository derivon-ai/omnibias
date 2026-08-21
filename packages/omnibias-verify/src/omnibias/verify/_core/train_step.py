# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Certified parameter step (theory 08-09).

A trial ``theta'`` is legal only when a sound
:func:`~omnibias.verify._core.certificates.lipschitz_bound` or
:func:`~omnibias.verify._core.taylor.taylor_output_bounds` enclosure of
a named input-output property stays inside a declared cap. Empty or
exploding enclosures reject the step; that is not a robustness claim.

Distinct from 08-04 (unique zero of a residual map). Not a global min,
not CCF stretch, and not a continuum PDE claim. Lipschitz and
Taylor-model paths use the founding bias collapse (``delta -> 0``)
``sigma'`` tower. This module is pure Python.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from omnibias.core.verified.interval import Interval, IntervalLike
from omnibias.verify._core.certificates import lipschitz_bound
from omnibias.verify._core.network import LinearLayer, Network
from omnibias.verify._core.taylor import taylor_output_bounds

PropertyName = Literal["lipschitz", "output_box"]
StepReason = Literal["ok", "violates", "vacuous"]
CollapseName = Literal["bias_collapse", "temperature_collapse"]

HONESTY: dict[str, bool] = {
    "robust_without_enclosure": False,
    "navier_stokes_proof_claim": False,
    "ccf_stretch_cleared": False,
    "global_min_claim": False,
    "continuum_pde_claim": False,
}


class CertifiedStepForbidden(ValueError):
    """Raised when a result would forge a robustness or accept claim."""


@dataclass(frozen=True)
class CertifiedStepConfig:
    """I/O property, declared cap, and input cell for a trial step."""

    property: PropertyName = "lipschitz"
    p_max: float = 2.0
    cell: Sequence[IntervalLike] | None = None
    lipschitz_norm: str = "inf"
    taylor_order: int = 2
    collapse: CollapseName = "bias_collapse"

    def __post_init__(self) -> None:
        if self.property not in ("lipschitz", "output_box"):
            raise ValueError(
                f"property must be 'lipschitz' or 'output_box', got {self.property!r}"
            )
        if not math.isfinite(self.p_max) or self.p_max < 0.0:
            raise ValueError(f"p_max must be a finite number >= 0, got {self.p_max}")
        if self.taylor_order < 1:
            raise ValueError(f"taylor_order must be >= 1, got {self.taylor_order}")
        if self.collapse not in ("bias_collapse", "temperature_collapse"):
            raise ValueError(
                f"collapse must be 'bias_collapse' or 'temperature_collapse', "
                f"got {self.collapse!r}"
            )
        if self.collapse == "temperature_collapse":
            raise ValueError(
                "certified_accept Lipschitz / output_box uses the sigma' "
                "tower (bias collapse, delta -> 0). Temperature collapse "
                "is the mollified-soft-gate path and is not this policy"
            )


@dataclass(frozen=True)
class CertifiedStepResult:
    """Accept / reject of one trial, with a sealed upper bound."""

    accepted: bool
    bound_hi: float | None
    reason: StepReason
    p_max: float
    collapse: CollapseName = "bias_collapse"
    robust_without_enclosure: bool = False
    navier_stokes_proof_claim: bool = False
    ccf_stretch_cleared: bool = False
    global_min_claim: bool = False
    continuum_pde_claim: bool = False

    def __post_init__(self) -> None:
        if self.reason not in ("ok", "violates", "vacuous"):
            raise CertifiedStepForbidden(f"unknown reason {self.reason!r}")
        if self.robust_without_enclosure:
            raise CertifiedStepForbidden(
                "robust_without_enclosure cannot be True; an empty enclosure "
                "is a reject, not a robustness claim"
            )
        if self.navier_stokes_proof_claim or self.ccf_stretch_cleared:
            raise CertifiedStepForbidden("cannot forge NS regularity or CCF stretch")
        if self.global_min_claim or self.continuum_pde_claim:
            raise CertifiedStepForbidden("cannot forge a global-min or continuum claim")
        if self.accepted:
            if self.reason != "ok":
                raise CertifiedStepForbidden("accepted steps must have reason='ok'")
            if self.bound_hi is None or not math.isfinite(self.bound_hi):
                raise CertifiedStepForbidden("accepted steps need a finite bound_hi")
            if self.bound_hi > self.p_max:
                raise CertifiedStepForbidden(
                    f"accepted bound_hi={self.bound_hi} exceeds p_max={self.p_max}"
                )
        elif self.reason == "ok":
            raise CertifiedStepForbidden("reason='ok' requires accepted=True")


def honesty_payload() -> dict[str, bool]:
    """Sealed artifact keys. Values are copies."""
    return dict(HONESTY)


def apply_flat_theta(net: Network, theta: Sequence[float]) -> Network:
    """Replace every :class:`LinearLayer` weight and bias from a flat vector."""
    vals = [float(t) for t in theta]
    i = 0
    layers = []
    for layer in net.layers:
        if isinstance(layer, LinearLayer):
            rows: list[tuple[float, ...]] = []
            for row in layer.weight:
                n = len(row)
                if i + n > len(vals):
                    raise ValueError(
                        f"theta has length {len(vals)}; need more entries "
                        "for LinearLayer weights"
                    )
                rows.append(tuple(vals[i : i + n]))
                i += n
            nb = len(layer.bias)
            if i + nb > len(vals):
                raise ValueError(
                    f"theta has length {len(vals)}; need more entries "
                    "for LinearLayer bias"
                )
            bias = tuple(vals[i : i + nb])
            i += nb
            layers.append(LinearLayer(weight=tuple(rows), bias=bias))
        else:
            layers.append(layer)
    if i != len(vals):
        raise ValueError(
            f"theta has {len(vals)} entries but the network uses {i} linear parameters"
        )
    return Network(layers)


def select_certified_theta(
    current: Sequence[float],
    trial: Sequence[float],
    result: CertifiedStepResult,
) -> tuple[float, ...]:
    """Keep ``trial`` only when the enclosure accepted it."""
    chosen = trial if result.accepted else current
    return tuple(float(x) for x in chosen)


def _as_cell(cell: Sequence[IntervalLike] | None) -> tuple[Interval, ...] | None:
    if cell is None:
        return None
    out: list[Interval] = []
    for item in cell:
        try:
            if isinstance(item, Interval):
                iv = item
            elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
                seq = list(item)
                if len(seq) != 2:
                    return None
                lo, hi = float(seq[0]), float(seq[1])
                if (not math.isfinite(lo)) or (not math.isfinite(hi)) or lo > hi:
                    return None
                iv = Interval(lo, hi)
            else:
                iv = Interval.from_value(item)
        except (TypeError, ValueError):
            return None
        if (not math.isfinite(iv.lo)) or (not math.isfinite(iv.hi)):
            return None
        out.append(iv)
    if not out:
        return None
    return tuple(out)


def _is_vacuous_hi(hi: float | None) -> bool:
    return hi is None or (not math.isfinite(hi))


def _lipschitz_hi(net: Network, cell: tuple[Interval, ...], norm: str) -> float:
    return float(lipschitz_bound(net, cell, norm=norm))


def _output_box_hi(net: Network, cell: tuple[Interval, ...], order: int) -> float:
    boxes = taylor_output_bounds(net, cell, order=order)
    if not boxes:
        raise ValueError("empty output enclosure")
    hi = 0.0
    for iv in boxes:
        if (not math.isfinite(iv.lo)) or (not math.isfinite(iv.hi)):
            return math.inf
        hi = max(hi, abs(iv.lo), abs(iv.hi))
    return float(hi)


def _enclose(
    net: Network, config: CertifiedStepConfig, cell: tuple[Interval, ...]
) -> float:
    if config.property == "lipschitz":
        return _lipschitz_hi(net, cell, config.lipschitz_norm)
    return _output_box_hi(net, cell, config.taylor_order)


def certified_accept(
    net: Network,
    theta_trial: Sequence[float] | None = None,
    *,
    config: CertifiedStepConfig | None = None,
) -> CertifiedStepResult:
    """Accept ``theta_trial`` only if the sealed I/O bound is ``<= p_max``.

    ``theta_trial`` replaces linear weights when given; omit it to score
    ``net`` as already holding the trial. Empty or non-finite enclosures
    return ``reason='vacuous'``, never ``accepted=True``.
    """
    cfg = config if config is not None else CertifiedStepConfig()
    trial_net = apply_flat_theta(net, theta_trial) if theta_trial is not None else net
    cell = _as_cell(cfg.cell)
    if cell is None:
        return CertifiedStepResult(
            accepted=False,
            bound_hi=None,
            reason="vacuous",
            p_max=cfg.p_max,
            collapse=cfg.collapse,
        )
    try:
        hi = _enclose(trial_net, cfg, cell)
    except (ValueError, ZeroDivisionError, OverflowError):
        return CertifiedStepResult(
            accepted=False,
            bound_hi=None,
            reason="vacuous",
            p_max=cfg.p_max,
            collapse=cfg.collapse,
        )
    if _is_vacuous_hi(hi):
        return CertifiedStepResult(
            accepted=False,
            bound_hi=None,
            reason="vacuous",
            p_max=cfg.p_max,
            collapse=cfg.collapse,
        )
    if hi <= cfg.p_max:
        return CertifiedStepResult(
            accepted=True,
            bound_hi=hi,
            reason="ok",
            p_max=cfg.p_max,
            collapse=cfg.collapse,
        )
    return CertifiedStepResult(
        accepted=False,
        bound_hi=hi,
        reason="violates",
        p_max=cfg.p_max,
        collapse=cfg.collapse,
    )


__all__ = [
    "CertifiedStepConfig",
    "CertifiedStepForbidden",
    "CertifiedStepResult",
    "CollapseName",
    "HONESTY",
    "PropertyName",
    "StepReason",
    "apply_flat_theta",
    "certified_accept",
    "honesty_payload",
    "select_certified_theta",
]
