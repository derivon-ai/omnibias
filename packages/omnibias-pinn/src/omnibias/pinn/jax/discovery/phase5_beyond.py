# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Phase 5 beyond-DeepMind structure layer (gated behind CCF Rung-2).

5a Partition near/far self-similar patches via ``omnibias.pinn.partition``.
5b Tab soft-tree ansatz router on meta-features (scaffold until partition earned).
5c Logic MaxSAT finite-obligation planner (scaffold; never continuum literals).

These modules must not replace CubicGN / Martens–Grosse on the fluid residual.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Phase5EntryGate:
    """Phase 5 may run only after CCF whole-line certification."""

    rung2_earned: bool
    whole_line_certified: bool

    @property
    def allowed(self) -> bool:
        return bool(self.rung2_earned and self.whole_line_certified)


def phase5_entry_from_status(status: dict[str, Any]) -> Phase5EntryGate:
    gates = status.get("gates", {})
    rung2 = bool(gates.get("rung2_earned", False))
    wl = bool(status.get("rung2", {}).get("whole_line_certified", rung2))
    return Phase5EntryGate(rung2_earned=rung2, whole_line_certified=wl)


def partitioned_near_far_residual_report(
    *,
    single_patch_residual: float,
    partitioned_residual: float,
    residual_threshold: float,
) -> dict[str, Any]:
    """Absolute gate helper: partitioned must beat single-patch and clear thr."""
    skill = float(single_patch_residual) - float(partitioned_residual)
    cleared = bool(
        partitioned_residual <= residual_threshold and skill > 0.0
    )
    return {
        "single_patch_residual": float(single_patch_residual),
        "partitioned_residual": float(partitioned_residual),
        "residual_threshold": float(residual_threshold),
        "skill": skill,
        "earned": cleared,
        "honesty": {
            "navier_stokes_proof_claim": False,
            "requires_rung2": True,
            "layer": "phase5a_partition",
        },
    }


def ansatz_router_meta_features(
    *,
    family: str,
    order: int,
    residual_scale: float,
    n_scales: int,
    gauge_ok: bool,
) -> np.ndarray:
    """Meta-feature vector for omnibias-tab router (not the PDE field)."""
    fam = {"ccf": 0.0, "ipm": 1.0, "boussinesq": 2.0}.get(family, -1.0)
    return np.asarray(
        [
            fam,
            float(order),
            float(np.log10(max(residual_scale, 1e-30))),
            float(n_scales),
            1.0 if gauge_ok else 0.0,
        ],
        dtype=float,
    )


def router_skill_report(
    *,
    fixed_schedule_failed_ticks: int,
    router_failed_ticks: int,
) -> dict[str, Any]:
    """Absolute gate: router skill > 0 vs fixed hand schedule."""
    skill = float(fixed_schedule_failed_ticks - router_failed_ticks)
    return {
        "fixed_schedule_failed_ticks": int(fixed_schedule_failed_ticks),
        "router_failed_ticks": int(router_failed_ticks),
        "skill": skill,
        "earned": skill > 0.0,
        "honesty": {
            "navier_stokes_proof_claim": False,
            "meta_features_only": True,
            "layer": "phase5b_router",
        },
    }


# Finite CAP/Lean checklist atoms for MaxSAT planning (no continuum literals).
FINITE_OBLIGATION_ATOMS: tuple[str, ...] = (
    "residual_margin",
    "radii_polynomial",
    "interval_sign",
    "replay_match",
    "honesty_flags",
)


def obligation_planner_report(
    *,
    hand_ordered_calls: int,
    planned_calls: int,
    obligations_cleared: int,
    obligations_total: int,
) -> dict[str, Any]:
    """Absolute gate: clear same finite obligations with ≤ hand-ordered calls."""
    same = obligations_cleared >= obligations_total
    fewer_or_eq = planned_calls <= hand_ordered_calls
    return {
        "hand_ordered_calls": int(hand_ordered_calls),
        "planned_calls": int(planned_calls),
        "obligations_cleared": int(obligations_cleared),
        "obligations_total": int(obligations_total),
        "atoms": list(FINITE_OBLIGATION_ATOMS),
        "earned": bool(same and fewer_or_eq),
        "honesty": {
            "navier_stokes_proof_claim": False,
            "continuum_literals_forbidden": True,
            "rh_forbidden": True,
            "yang_mills_forbidden": True,
            "layer": "phase5c_obligations",
        },
    }


def blocked_phase5_bundle(reason: str) -> dict[str, Any]:
    return {
        "phase5": "BLOCKED",
        "reason": reason,
        "honesty": {
            "navier_stokes_proof_claim": False,
            "entry_requires_whole_line_certified": True,
        },
        "earned": False,
    }


__all__ = [
    "FINITE_OBLIGATION_ATOMS",
    "Phase5EntryGate",
    "ansatz_router_meta_features",
    "blocked_phase5_bundle",
    "obligation_planner_report",
    "partitioned_near_far_residual_report",
    "phase5_entry_from_status",
    "router_skill_report",
]
