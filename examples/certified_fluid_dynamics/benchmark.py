# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Deterministic proof-carrying fluid-dynamics smoke benchmark (generated data).

Certifies the residual of two exact periodic incompressible flows -- the 2-D
Taylor--Green vortex (laminar baseline) and the steady forced Kolmogorov base
state (forced / chaotic-facing) -- and adjudicates each through the proof
machine with schema, independent-replay and honesty gates.  A short Taylor--Green
"rollout" re-certifies several snapshots of the *exact analytic* decay to show
the residual stays at machine zero with no numerical drift.

Claim boundary
--------------
This is a finite-grid, finite-time residual certificate for known analytic model
flows.  It is **not** perfect weather, **not** a continuum Navier--Stokes proof,
and **not** high-Reynolds turbulence or long-horizon chaos tracking; the
certificates hard-wire those claims to ``False``.
"""

from __future__ import annotations

from typing import Any

from omnibias.core.proof import Conjecture
from omnibias.pinn.certified import (
    build_default_machine,
    certified_kolmogorov_residual,
    certified_taylor_green_residual,
)
from omnibias.pinn.certified.fluid_fixtures import (
    save_periodic_flow_sample,
    taylor_green_vortex,
)
from omnibias.symbolic.fluid import verify_periodic_flow_residual


def _adjudicate(machine: Any, label: str, cert: dict[str, Any]) -> dict[str, Any]:
    """Run a sealed certificate through the proof machine + independent replay."""
    verdict = machine.evaluate(
        Conjecture(label, "navier_stokes_periodic_residual", {"certificate": cert})
    )
    replay = verify_periodic_flow_residual(cert)
    return {
        "label": label,
        "fixture": cert["fixture"]["name"],
        "residual_sup": cert["residual_sup"],
        "momentum_residual_sup": cert["momentum_residual_sup"],
        "continuity_residual_sup": cert["continuity_residual_sup"],
        "pressure_poisson_residual_sup": cert["pressure_poisson_residual_sup"],
        "kinetic_energy": cert["kinetic_energy"],
        "enstrophy": cert["enstrophy"],
        "exact_solution_claim": cert["exact_solution_claim"],
        "verdict": verdict.status,
        "schema_ok": verdict.schema_ok,
        "replay_ok": verdict.replay_ok,
        "honesty_ok": verdict.honesty_ok,
        "replay_match": replay["replay_match"],
        "unproven_claim": cert["honesty"]["unproven_claim"],
        "chaotic_tracking_claim": cert["honesty"]["chaotic_tracking_claim"],
        "interval_verified": cert["honesty"]["interval_verified"],
    }


def evaluate_benchmark(
    *,
    n: int = 64,
    viscosity: float = 0.1,
    rollout_times: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0),
    scratch_dir: str | None = None,
) -> dict[str, Any]:
    """Certify Taylor--Green and Kolmogorov periodic flows and gate the verdicts.

    Parameters
    ----------
    n
        Grid resolution per axis.
    viscosity
        Dynamic viscosity (density is 1, so this is also the kinematic value).
    rollout_times
        Snapshot times at which the exact Taylor--Green decay is re-certified.
    scratch_dir
        Optional runtime directory.  When given, each generated sample's arrays
        and descriptor are cached there (nothing is written by default).
    """
    machine = build_default_machine()

    taylor_green = _adjudicate(
        machine,
        "Taylor-Green vortex (laminar baseline)",
        certified_taylor_green_residual(n, viscosity=viscosity),
    )
    kolmogorov = _adjudicate(
        machine,
        "Kolmogorov forced shear (chaotic-facing base state)",
        certified_kolmogorov_residual(n, viscosity=viscosity, wavenumber=4),
    )

    rollout: list[dict[str, Any]] = []
    for t in rollout_times:
        cert = certified_taylor_green_residual(n, viscosity=viscosity, time=float(t))
        verdict = machine.evaluate(
            Conjecture(
                f"Taylor-Green snapshot t={t}",
                "navier_stokes_periodic_residual",
                {"certificate": cert},
            )
        )
        rollout.append({
            "time": float(t),
            "residual_sup": cert["residual_sup"],
            "kinetic_energy": cert["kinetic_energy"],
            "verdict": verdict.status,
        })

    saved: dict[str, Any] = {}
    if scratch_dir is not None:
        sample = taylor_green_vortex(n, viscosity=viscosity)
        saved = save_periodic_flow_sample(sample, scratch_dir)

    cases = [taylor_green, kolmogorov]
    return {
        "cases": cases,
        "taylor_green_rollout": rollout,
        "all_proved": all(c["verdict"] == "PROVED" for c in cases),
        "all_replayed": all(c["replay_match"] for c in cases),
        "rollout_residual_drift_free": all(r["residual_sup"] < 1e-8 for r in rollout),
        "grid": {"n": n, "viscosity": viscosity},
        "saved_artifacts": saved,
        "claim_boundary": (
            "finite-grid, finite-time residual certificates for exact analytic "
            "model flows; not perfect weather, not continuum Navier-Stokes, not "
            "high-Reynolds turbulence or long-horizon chaos tracking"
        ),
    }


__all__ = ["evaluate_benchmark"]
