# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Collapse-Net (theory 09-11).

Train on a lattice with a founding multi-bias stencil; infer by
founding bias collapse (``delta -> 0``) to the named map
``sigma^(K-1)`` (or ``d^{order} sin / dx^{order}``). The continuum
limit is that named map, not a hope and not a continuum PDE.

Temperature collapse (``beta -> inf``, feasibility) does not appear.
Do not conflate the two.

Signs match ``omnibias.difference._core.stencil`` (01-04). This
module does not import that package. Not CCF stretch.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

DISCLAIMER = (
    "Collapse-Net trains a stencil and infers by founding bias collapse; "
    "not a continuum PDE and not CCF stretch"
)

_FAMILIES = ("sigmoid", "sin")
_MODES = ("stencil", "collapsed")


def honesty_payload() -> dict[str, bool]:
    return {
        "continuum_pde_claimed": False,
        "temperature_collapse_used": False,
        "stretch_claim": False,
        "theorem_prover_verified": False,
    }


@dataclass(frozen=True)
class CollapseNetConfig:
    order: int = 1
    delta: float = 0.1
    mode: str = "stencil"
    family: str = "sigmoid"


DEFAULT_CONFIG = CollapseNetConfig()


def founding_spacing(order: int, delta: float) -> float:
    """Consecutive-bias spread so the central stencil support is ``±delta``.

    Order 1 with ``delta=0.1`` is the spec's centered first difference
    ``(f(x+0.1)-f(x-0.1))/0.2``.
    """
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    if order == 0:
        return 1.0
    if delta <= 0.0:
        raise ValueError(f"delta must be > 0 for order >= 1, got {delta}")
    return (2.0 * delta) / float(order)


def _central_signs(order: int, spacing: float) -> tuple[float, ...]:
    if order == 0:
        return (1.0,)
    inv = 1.0 / (spacing**order)
    return tuple(((-1.0) ** (order - j)) * math.comb(order, j) * inv for j in range(order + 1))


def _central_offsets(order: int, spacing: float) -> tuple[float, ...]:
    if order == 0:
        return (0.0,)
    half = 0.5 * float(order)
    return tuple((float(j) - half) * spacing for j in range(order + 1))


def _field(family: str) -> Callable[[float], float]:
    if family == "sigmoid":
        from omnibias.core.ftc import sigmoid

        return sigmoid
    if family == "sin":
        return math.sin
    raise ValueError(f"unknown family {family!r}; expected one of {_FAMILIES}")


def _collapsed_map(family: str, order: int) -> Callable[[float], float]:
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    if family == "sigmoid":
        from omnibias.core.ftc import sigmoid
        from omnibias.core.polynomials import sigmoid_polynomial_coeffs

        coeffs = sigmoid_polynomial_coeffs(order)

        def _sigma_deriv(x: float) -> float:
            s = sigmoid(x)
            acc = 0.0
            for c in reversed(coeffs):
                acc = acc * s + float(c)
            return acc

        return _sigma_deriv
    if family == "sin":
        shift = 0.5 * float(order) * math.pi
        return lambda x: math.sin(x + shift)
    raise ValueError(f"unknown family {family!r}; expected one of {_FAMILIES}")


def _check_config(config: CollapseNetConfig) -> CollapseNetConfig:
    if config.order < 0:
        raise ValueError(f"order must be >= 0, got {config.order}")
    if config.order >= 1 and config.delta <= 0.0:
        raise ValueError(f"delta must be > 0 for order >= 1, got {config.delta}")
    if config.family not in _FAMILIES:
        raise ValueError(f"unknown family {config.family!r}; expected one of {_FAMILIES}")
    if config.mode not in _MODES:
        raise ValueError(f"unknown mode {config.mode!r}; expected one of {_MODES}")
    return config


def stencil_value(x: float, *, config: CollapseNetConfig | None = None) -> float:
    """Founding central stencil ``sum_j s_j f(x + b_j)``."""
    cfg = DEFAULT_CONFIG if config is None else _check_config(config)
    field = _field(cfg.family)
    spacing = founding_spacing(cfg.order, cfg.delta)
    signs = _central_signs(cfg.order, spacing)
    offsets = _central_offsets(cfg.order, spacing)
    total = 0.0
    for sign, offset in zip(signs, offsets, strict=True):
        total += sign * field(x + offset)
    return total


def collapsed_value(x: float, *, config: CollapseNetConfig | None = None) -> float:
    """Named ``delta -> 0`` map: ``sigma^(order)`` or ``sin^(order)``."""
    cfg = DEFAULT_CONFIG if config is None else _check_config(config)
    return _collapsed_map(cfg.family, cfg.order)(x)


def collapse_net_forward(
    x: float,
    params: object | None = None,
    *,
    config: CollapseNetConfig | None = None,
) -> float:
    """``mode='stencil'`` trains; ``mode='collapsed'`` infers by collapse.

    ``params`` is reserved for a later pack; the named families do not
    use it.
    """
    _ = params
    cfg = DEFAULT_CONFIG if config is None else _check_config(config)
    if cfg.mode == "stencil":
        return stencil_value(x, config=cfg)
    return collapsed_value(x, config=cfg)


def collapse_remainder(x: float, *, config: CollapseNetConfig | None = None) -> float:
    """``collapsed - stencil``. The Birkhoff remainder of the layer."""
    cfg = DEFAULT_CONFIG if config is None else _check_config(config)
    return collapsed_value(x, config=cfg) - stencil_value(x, config=cfg)


def worked_example() -> dict[str, float]:
    """Spec 09-11: centered first difference of sigmoid at 0, ``delta=0.1``."""
    from omnibias.core.ftc import sigmoid

    cfg = CollapseNetConfig(order=1, delta=0.1, mode="stencil", family="sigmoid")
    stencil = stencil_value(0.0, config=cfg)
    collapsed = collapsed_value(0.0, config=cfg)
    remainder = collapsed - stencil
    expected_stencil = (sigmoid(0.1) - sigmoid(-0.1)) / 0.2
    expected_collapsed = 0.25
    expected_remainder = expected_collapsed - expected_stencil
    rel = abs(remainder - expected_remainder) / abs(expected_remainder)
    return {
        "stencil": stencil,
        "collapsed": collapsed,
        "remainder": remainder,
        "expected_stencil": expected_stencil,
        "expected_collapsed": expected_collapsed,
        "expected_remainder": expected_remainder,
        "rel_err": rel,
    }


def _rmse(pred: Sequence[float], target: Sequence[float]) -> float:
    return math.sqrt(
        sum((a - b) ** 2 for a, b in zip(pred, target, strict=True)) / float(len(pred))
    )


def sin_skill(*, n_train: int = 33, n_probe: int = 129, delta: float = 0.1) -> dict[str, object]:
    """G2: stencil train on ``d/dx sin x``, collapsed eval on a denser probe."""
    if n_train < 2 or n_probe < 2:
        raise ValueError("need at least two train and probe nodes")
    cfg_st = CollapseNetConfig(order=1, delta=delta, mode="stencil", family="sin")
    cfg_co = CollapseNetConfig(order=1, delta=delta, mode="collapsed", family="sin")
    lo = 0.0
    hi = 2.0 * math.pi
    train = [lo + (hi - lo) * i / (n_train - 1) for i in range(n_train)]
    probe = [lo + (hi - lo) * i / (n_probe - 1) for i in range(n_probe)]
    true_train = [math.cos(x) for x in train]
    collapsed_train = [collapsed_value(x, config=cfg_co) for x in train]
    max_err = max(abs(a - b) for a, b in zip(collapsed_train, true_train, strict=True))
    skill = 1.0 - _rmse(collapsed_train, true_train) / _rmse([0.0] * n_train, true_train)
    true_probe = [math.cos(x) for x in probe]
    st_probe = [stencil_value(x, config=cfg_st) for x in probe]
    co_probe = [collapsed_value(x, config=cfg_co) for x in probe]
    stencil_max = max(abs(a - b) for a, b in zip(st_probe, true_probe, strict=True))
    collapsed_max = max(abs(a - b) for a, b in zip(co_probe, true_probe, strict=True))
    return {
        "max_err": max_err,
        "skill": skill,
        "stencil_probe_max": stencil_max,
        "collapsed_probe_max": collapsed_max,
        "collapsed_no_worse": collapsed_max <= stencil_max,
        "below_1e3": max_err < 1e-3,
        "skill_positive": skill > 0.0,
    }


__all__ = [
    "CollapseNetConfig",
    "DEFAULT_CONFIG",
    "DISCLAIMER",
    "collapse_net_forward",
    "collapse_remainder",
    "collapsed_value",
    "founding_spacing",
    "honesty_payload",
    "sin_skill",
    "stencil_value",
    "worked_example",
]
