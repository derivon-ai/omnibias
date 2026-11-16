# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Depth-causal local jet algebra (theory 08-03).

A layer may take a damped Gauss-Newton step on a *named* local residual
and push a compressed ``k``-direction jet to the next layer. Later
parameters are updated first so an earlier layer sees a corrected
downstream map (spec §5). The jet is Faà di Bruno / ``layer_jet`` --
the chain rule, not a skip of it.

This module is backend-free: config, the flood forbid, and the
strictly-monotone inverse used by invert-and-match. Tensor GN and
``layer_jet`` live in ``omnibias.{torch,jax}.train_local``.

Bias collapse (``delta -> 0``) supplies the tower. No temperature
collapse. Local GN is greedy, not a global min, and not CCF stretch.
``n_directions >= n_params`` raises :class:`LocalJetForbidden` unless
``allow_full=True``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

LocalJetVariant = Literal["readout", "invert", "predcode"]

INVERTIBLE_SIGMA: frozenset[str] = frozenset({"tanh", "sigmoid"})


class LocalJetForbidden(ValueError):
    """Raised when the caller requests a full ``d h / d theta`` flood."""


@dataclass(frozen=True)
class LocalJetConfig:
    """Direction budget, jet order, damping, and named local residual."""

    n_directions: int = 4
    jet_order: int = 1
    damping: float = 1e-4
    variant: LocalJetVariant = "readout"
    allow_full: bool = False
    refit_last: bool = True

    def __post_init__(self) -> None:
        if self.n_directions < 1:
            raise ValueError(f"n_directions must be >= 1, got {self.n_directions}")
        if self.jet_order < 0:
            raise ValueError(f"jet_order must be >= 0, got {self.jet_order}")
        if self.damping < 0.0 or not math.isfinite(self.damping):
            raise ValueError(
                f"damping must be a finite number >= 0, got {self.damping}"
            )
        if self.variant not in ("readout", "invert", "predcode"):
            raise ValueError(
                f"variant must be 'readout', 'invert', or 'predcode', "
                f"got {self.variant!r}"
            )


@dataclass(frozen=True)
class LocalJetReport:
    """Diagnostics of one depth-causal sweep.

    ``greedy_only_claimed_optimal`` is always ``False``: a local-only
    run is not written as optimal (G3).
    """

    residual_norms: tuple[float, ...]
    n_directions: int
    n_params: int
    variant: str
    jet_shape: tuple[int, ...]
    greedy_only_claimed_optimal: bool = False


def mlp_param_count(sizes: Sequence[tuple[int, int | None]]) -> int:
    """Total ``W`` + optional ``b`` entries across an MLP layer list."""
    total = 0
    if not sizes:
        raise ValueError("layers must be non-empty")
    for n_w, n_b in sizes:
        if n_w < 1:
            raise ValueError(f"layer weight size must be >= 1, got {n_w}")
        total += int(n_w)
        if n_b is not None:
            if n_b < 0:
                raise ValueError(f"layer bias size must be >= 0, got {n_b}")
            total += int(n_b)
    return total


def reject_local_jet_flood(
    n_directions: int, n_params: int, allow_full: bool
) -> None:
    """G1: refuse ``n_directions >= n_params`` unless ``allow_full``."""
    if n_directions < 1:
        raise ValueError(f"n_directions must be >= 1, got {n_directions}")
    if n_params < 1:
        raise ValueError(f"n_params must be >= 1, got {n_params}")
    if n_directions >= n_params and not allow_full:
        raise LocalJetForbidden(
            f"n_directions={n_directions} >= n_params={n_params}; "
            "pass allow_full=True to opt into a full-parameter Jacobian "
            "(the API rejects silent d h / d theta materialisation)"
        )


def require_invertible_sigma(name: str) -> None:
    """Invert-and-match is defined only for strictly monotone ``sigma``."""
    key = name.lower()
    if key not in INVERTIBLE_SIGMA:
        raise ValueError(
            f"invert variant requires strictly monotone sigma "
            f"(tanh or sigmoid); {name!r} is not invertible here"
        )


def invert_sigma(name: str, value: float) -> float:
    """``sigma^{-1}(y)`` for ``tanh`` (artanh) or ``sigmoid`` (logit)."""
    require_invertible_sigma(name)
    y = float(value)
    key = name.lower()
    if key == "tanh":
        if not -1.0 < y < 1.0:
            raise ValueError(f"artanh requires |y| < 1, got {y}")
        return math.atanh(y)
    if not 0.0 < y < 1.0:
        raise ValueError(f"logit requires y in (0, 1), got {y}")
    return math.log(y / (1.0 - y))


__all__ = [
    "INVERTIBLE_SIGMA",
    "LocalJetConfig",
    "LocalJetForbidden",
    "LocalJetReport",
    "LocalJetVariant",
    "invert_sigma",
    "mlp_param_count",
    "reject_local_jet_flood",
    "require_invertible_sigma",
]
