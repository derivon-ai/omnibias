# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Depth-causal residual algebra (theory 08-05).

After layer ``ell``, a declared decode of ``h_ell`` is a field. The same
local differential operator ``N`` used at the last readout forms
``r_ell = N[u_ell] - f``. A damped Gauss-Newton step updates only that
layer (and optionally its decode), then later layers see the corrected
activations. This is causal in **network depth**, not in physical time; the
time-marching drivers instead live in
``omnibias.pinn.train.torch.march`` and ``omnibias.pinn.train.jax.march``.
Distinct from 08-03 (a named *proxy* residual).

Backend-free: config, the unlock predicate, 1-D Dirichlet mask towers,
and the Leibniz product that applies a hard-BC factor to a derivative
tower. Tensor GN and ``mlp_jet`` / ``layer_jet`` live in
``omnibias.pinn.train.{torch,jax}.depth_residual``.

Bias collapse (``delta -> 0``) supplies the tower. No temperature
collapse. Local GN is greedy, not a global min, and not CCF stretch.
Hilbert / nonlocal operators are out of scope. Continuum Navier-Stokes
regularity is not a claim.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from omnibias.core.local_jet import LocalJetForbidden, reject_local_jet_flood

HardBCKind = Literal["unit_interval", "symmetric", "none"]
LastLayerScope = Literal["block", "decode"]

SEALED_HONESTY: dict[str, bool] = {
    "navier_stokes_proof_claim": False,
    "stretch_1e-13_cleared": False,
    "hilbert_not_in_scope": True,
    "greedy_only_claimed_optimal": False,
}

# Artifact key ``stretch_1e-13_cleared`` is not a valid identifier.


class DepthResidualForbidden(LocalJetForbidden):
    """Raised when the caller requests a full-parameter flood."""


@dataclass(frozen=True)
class DepthResidualConfig:
    """Direction budget, jet order, damping, and optional depth unlock.

    ``n_directions`` is the compressed input-jet budget from 08-03, not a
    full ``d h / d theta``. ``jet_order`` must be at least 2 for a
    second-order residual such as 1-D Poisson. ``unlock_ratio`` is an
    optional Wang-style gate on later *layers* (not time bins); the
    acceptance gate does not require it. ``last_layer_only`` spends the
    matched GN budget on the last layer and is the G1 control.
    """

    n_directions: int = 4
    jet_order: int = 2
    damping: float = 1e-4
    unlock_ratio: float | None = None
    update_decode: bool = True
    last_layer_only: bool = False
    last_layer_scope: LastLayerScope = "block"
    allow_full: bool = False
    steps_per_layer: int = 1
    hard_bc: HardBCKind = "unit_interval"

    def __post_init__(self) -> None:
        if self.n_directions < 1:
            raise ValueError(f"n_directions must be >= 1, got {self.n_directions}")
        if self.jet_order < 0:
            raise ValueError(f"jet_order must be >= 0, got {self.jet_order}")
        if self.damping < 0.0 or not math.isfinite(self.damping):
            raise ValueError(
                f"damping must be a finite number >= 0, got {self.damping}"
            )
        if self.unlock_ratio is not None:
            if self.unlock_ratio < 0.0 or not math.isfinite(self.unlock_ratio):
                raise ValueError(
                    f"unlock_ratio must be None or a finite number >= 0, "
                    f"got {self.unlock_ratio}"
                )
        if self.steps_per_layer < 1:
            raise ValueError(
                f"steps_per_layer must be >= 1, got {self.steps_per_layer}"
            )
        if self.hard_bc not in ("unit_interval", "symmetric", "none"):
            raise ValueError(
                f"hard_bc must be 'unit_interval', 'symmetric', or 'none', "
                f"got {self.hard_bc!r}"
            )
        if self.last_layer_scope not in ("block", "decode"):
            raise ValueError(
                f"last_layer_scope must be 'block' or 'decode', "
                f"got {self.last_layer_scope!r}"
            )


@dataclass(frozen=True)
class DepthResidualReport:
    """Diagnostics of one causal-in-depth residual sweep.

    Honesty fields are sealed: this trainer never claims Navier-Stokes
    regularity, CCF stretch, Hilbert, or a global min (G2 / G3).
    """

    residual_norms: tuple[float, ...]
    residual_norms_before: tuple[float, ...]
    unlocked: tuple[bool, ...]
    n_directions: int
    n_params: int
    n_gn_steps: int
    last_layer_only: bool
    greedy_only_claimed_optimal: bool = False
    navier_stokes_proof_claim: bool = False
    stretch_cleared: bool = False
    hilbert_not_in_scope: bool = True

    def __post_init__(self) -> None:
        if self.greedy_only_claimed_optimal:
            raise ValueError(
                "greedy_only_claimed_optimal is sealed False: a depth-causal "
                "sweep is not written as optimal"
            )
        if self.navier_stokes_proof_claim:
            raise ValueError(
                "navier_stokes_proof_claim is sealed False: finite "
                "collocation residual only"
            )
        if self.stretch_cleared:
            raise ValueError(
                "stretch_1e-13_cleared is sealed False: Hilbert is out of scope"
            )
        if not self.hilbert_not_in_scope:
            raise ValueError(
                "hilbert_not_in_scope is sealed True: this gate is a local PDE"
            )


def reject_depth_residual_flood(
    n_directions: int, n_params: int, allow_full: bool
) -> None:
    """Refuse ``n_directions >= n_params`` unless ``allow_full`` (08-03 G1)."""
    try:
        reject_local_jet_flood(n_directions, n_params, allow_full)
    except LocalJetForbidden as exc:
        raise DepthResidualForbidden(str(exc)) from exc


def layer_is_unlocked(
    layer_index: int,
    previous_residual_norm: float | None,
    reference_residual_norm: float | None,
    unlock_ratio: float | None,
) -> bool:
    """Unlock layer ``ell`` (Wang-style, on depth, not time).

    Layer 0 always runs. Later layers run when ``unlock_ratio`` is
    ``None``, or when the previous layer's residual is at most
    ``unlock_ratio`` times the first layer's pre-step residual.
    """
    if layer_index < 0:
        raise ValueError(f"layer_index must be >= 0, got {layer_index}")
    if unlock_ratio is None or layer_index == 0:
        return True
    if previous_residual_norm is None or reference_residual_norm is None:
        raise ValueError(
            "unlock comparison needs the previous residual and the "
            "first-layer reference residual"
        )
    if previous_residual_norm < 0.0 or reference_residual_norm < 0.0:
        raise ValueError("residual norms must be non-negative")
    return previous_residual_norm <= unlock_ratio * reference_residual_norm


def hard_bc_mask_tower(x: float, kind: HardBCKind, order: int) -> tuple[float, ...]:
    """Derivative tower of a 1-D Dirichlet factor, padded with zeros.

    ``unit_interval`` is ``s = x (1 - x)`` (spec §5, ``[0, 1]``).
    ``symmetric`` is ``s = 1 - x^2`` (the 08-03 ``[-1, 1]`` wrapper).
    ``none`` is the constant ``1``.
    """
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    z = float(x)
    rows = [0.0] * (order + 1)
    if kind == "none":
        rows[0] = 1.0
        return tuple(rows)
    if kind == "unit_interval":
        rows[0] = z * (1.0 - z)
        if order >= 1:
            rows[1] = 1.0 - 2.0 * z
        if order >= 2:
            rows[2] = -2.0
        return tuple(rows)
    rows[0] = 1.0 - z * z
    if order >= 1:
        rows[1] = -2.0 * z
    if order >= 2:
        rows[2] = -2.0
    return tuple(rows)


def leibniz_product_tower(
    left: tuple[float, ...] | list[float],
    right: tuple[float, ...] | list[float],
) -> tuple[float, ...]:
    """Derivative tower of a product via the Leibniz rule.

    ``(fg)^{(n)} = sum_k C(n, k) f^{(k)} g^{(n-k)}``. Used to wrap a
    decoded field by a hard-BC mask without finite differences.
    """
    a = tuple(float(v) for v in left)
    b = tuple(float(v) for v in right)
    n = min(len(a), len(b))
    if n == 0:
        raise ValueError("Leibniz towers must be non-empty")
    out: list[float] = []
    for m in range(n):
        acc = 0.0
        for k in range(m + 1):
            acc += float(math.comb(m, k)) * a[k] * b[m - k]
        out.append(acc)
    return tuple(out)


def apply_hard_bc_tower(
    field_tower: tuple[float, ...] | list[float],
    x: float,
    kind: HardBCKind,
) -> tuple[float, ...]:
    """Apply a named 1-D Dirichlet factor to a decoded-field tower."""
    order = len(tuple(field_tower)) - 1
    if order < 0:
        raise ValueError("field_tower must be non-empty")
    return leibniz_product_tower(hard_bc_mask_tower(x, kind, order), field_tower)


def honesty_payload() -> dict[str, bool]:
    """G2 keys for artifacts. Values are sealed copies."""
    return dict(SEALED_HONESTY)


__all__ = [
    "DepthResidualConfig",
    "DepthResidualForbidden",
    "DepthResidualReport",
    "HardBCKind",
    "LastLayerScope",
    "SEALED_HONESTY",
    "apply_hard_bc_tower",
    "hard_bc_mask_tower",
    "honesty_payload",
    "layer_is_unlocked",
    "leibniz_product_tower",
    "reject_depth_residual_flood",
]
