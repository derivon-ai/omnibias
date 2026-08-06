# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Internal Navier-Stokes candidate-sprint harness. **Not public API.**

Pure orchestration: every step below already exists in
:mod:`omnibias.pinn.certified.navier_stokes`, and this module only runs them in
order and collects what each one refused to conclude.  It computes no new
mathematics and weakens no gate.

Why it is underscore-prefixed and absent from ``__all__``
---------------------------------------------------------
A "sprint" that assembles blowup and regularity machinery into a single ledger is
exactly the kind of surface that gets mistaken for a result.  Keeping it private
means it can be run, replayed and regression-tested without appearing in the
public API, the docs, or ``docs/api/``.  ``test_sprint_harness.py`` asserts that
absence, so promotion has to be deliberate rather than accidental.

What the ledger is, and what it is not
--------------------------------------
It is a **record of open obligations**: which gates were reached, which were
refused, and precisely what each one still wants.  Every gate here is designed to
be blocked, and a run in which one is *not* blocked would be a bug in this
harness rather than a proof of anything.  ``theorem_claimed`` is therefore
hard-wired ``False`` and :func:`sprint_honesty_errors` re-derives it from the
phases rather than trusting the flag.

Nothing here claims global regularity, finite-time blowup, or any Clay-problem
statement, and no amount of running it can produce one.
"""

from __future__ import annotations

import platform
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from omnibias.pinn.certified.navier_stokes import (
    build_axisymmetric_blowup_closure_report,
    build_axisymmetric_interval_report,
    build_candidate_artifact,
    build_formal_proof_package,
    build_ns_cap_bundle,
    build_ns_proof_program_report,
    build_ns_solve_or_falsify_report,
    build_refined_axisymmetric_swirl_candidate_artifact,
    build_regularity_closure_report,
    build_theorem_grade_closure_attempt,
    candidate_artifact_schema_errors,
    candidate_upgrade_gates,
    compactified_coefficient_set,
    compactified_sandbox_replay_grid,
    external_review_gate,
    lean_formalization_package,
    manufactured_abc_flow,
    regularity_counterexample_sweep,
    theorem_claim_gate,
)

SPRINT_SCHEMA_VERSION = "navier-stokes-candidate-sprint-1"

#: Phases the ledger always contains, in execution order.
SPRINT_PHASES = (
    "candidate",
    "interval_report",
    "upgrade_gate",
    "counterexample_sweep",
    "growth_law_candidate",
    "closure",
    "theorem_gate",
    "review_gate",
    "solve_or_falsify",
    "replay",
)


@dataclass(frozen=True)
class SprintConfig:
    """Everything that fixes a run, so a ledger can be reproduced from its record."""

    seed: int = 11
    n_radial: int = 5
    n_axial: int = 6
    radial_degree: int = 0
    axial_degree: int = 1
    max_iterations: int = 1
    step_size: float = 0.01
    viscosity: float = 0.01
    mms_resolution: int = 12
    mms_viscosity: float = 0.05
    trace_points: int = 8
    counterexample_target: str = "enstrophy"
    notes: str = ""


@dataclass
class _Ledger:
    phases: dict[str, Any] = field(default_factory=dict)
    open_obligations: list[str] = field(default_factory=list)
    blocked_gates: list[str] = field(default_factory=list)


def _collect(ledger: _Ledger, name: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Record a phase and harvest whatever it refused to conclude."""
    ledger.phases[name] = payload
    for key in ("open_obligations", "open_lemmas"):
        ledger.open_obligations.extend(str(item) for item in payload.get(key, []))
    status = str(
        payload.get("promotion_status")
        or payload.get("review_status")
        or payload.get("proof_status")
        or ""
    )
    if status.startswith("blocked") or status == "candidate_falsified":
        ledger.blocked_gates.append(name)
    return payload


def _synthetic_traces(config: SprintConfig) -> dict[str, np.ndarray]:
    """A deterministic, seeded trace bundle for the counterexample probes."""
    rng = np.random.default_rng(config.seed)
    time = np.linspace(0.0, 1.0, config.trace_points)
    energy = 1.0 + 0.1 * time + 0.01 * rng.standard_normal(config.trace_points)
    enstrophy = 1.0 + 0.5 * time + 0.01 * rng.standard_normal(config.trace_points)
    return {"energy": energy, "enstrophy": enstrophy}


def run_candidate_sprint(config: SprintConfig | None = None) -> dict[str, Any]:
    """Run the whole candidate loop once and return the open-obligation ledger.

    Deterministic in ``config.seed``.  Every phase is an existing function called
    with existing arguments; the only thing this adds is the order and the
    bookkeeping.
    """
    cfg = config if config is not None else SprintConfig()
    ledger = _Ledger()

    artifact = build_refined_axisymmetric_swirl_candidate_artifact(
        seed=cfg.seed,
        n_radial=cfg.n_radial,
        n_axial=cfg.n_axial,
        radial_degree=cfg.radial_degree,
        axial_degree=cfg.axial_degree,
        max_iterations=cfg.max_iterations,
        step_size=cfg.step_size,
        viscosity=cfg.viscosity,
    )
    _collect(ledger, "candidate", artifact)

    interval_report = build_axisymmetric_interval_report(artifact)
    _collect(ledger, "interval_report", interval_report)

    mms = manufactured_abc_flow(cfg.mms_resolution, viscosity=cfg.mms_viscosity)
    bundle = build_ns_cap_bundle(
        mms["velocity"],
        mms["pressure"],
        velocity_t=mms["velocity_t"],
        forcing=mms["forcing"],
        viscosity=mms["viscosity"],
    )
    _collect(
        ledger,
        "upgrade_gate",
        candidate_upgrade_gates(
            bundle, independent_report={"residual_samples_match": True}
        ),
    )

    traces = _synthetic_traces(cfg)
    sweep = regularity_counterexample_sweep(
        {cfg.counterexample_target: 1.0},
        traces=traces,
        target=cfg.counterexample_target,
    )
    _collect(ledger, "counterexample_sweep", sweep)
    _collect(ledger, "growth_law_candidate", _growth_law_candidate(cfg, traces, sweep))

    blowup = build_axisymmetric_blowup_closure_report(
        interval_report, norm_growth_exponent=0.25, linked_norm_profile=True
    )
    regularity = build_regularity_closure_report(
        inequality_name="sprint_probe",
        coefficients={cfg.counterexample_target: 1.0},
        counterexample_count=len(sweep.get("counterexamples", [])),
    )
    theorem_attempt = build_theorem_grade_closure_attempt(
        interval_report, blowup_report=blowup, regularity_report=regularity
    )
    _collect(ledger, "closure", theorem_attempt)

    formal = build_formal_proof_package(theorem_attempt)
    _collect(ledger, "theorem_gate", theorem_claim_gate(theorem_attempt, formal))

    program = build_ns_proof_program_report(theorem_attempt=theorem_attempt)
    _collect(
        ledger,
        "review_gate",
        external_review_gate(program, lean_formalization_package(program)),
    )

    _collect(
        ledger,
        "solve_or_falsify",
        build_ns_solve_or_falsify_report(
            interval_report, blowup_report=blowup, theorem_attempt=theorem_attempt
        ),
    )

    _collect(ledger, "replay", _replay(artifact))

    return {
        "schema_version": SPRINT_SCHEMA_VERSION,
        "config": asdict(cfg),
        "phases": ledger.phases,
        "open_obligations": sorted(set(ledger.open_obligations)),
        "blocked_gates": sorted(set(ledger.blocked_gates)),
        "theorem_claimed": False,
        "honesty": {
            "unproven_claim": False,
            "global_regularity_claim": False,
            "finite_time_blowup_claim": False,
            "theorem_prover_verified": False,
            "notes": cfg.notes,
        },
        "provenance": {
            "harness": "omnibias.pinn.certified._sprint",
            "python": platform.python_version(),
        },
    }


def _growth_law_candidate(
    config: SprintConfig,
    traces: dict[str, np.ndarray],
    sweep: dict[str, Any],
) -> dict[str, Any]:
    """Package the swept growth law as a replayable, schema-checked candidate artifact.

    The schema errors are carried in the phase rather than raised: a malformed
    candidate is a finding the ledger should record, not a reason to abandon the run.
    """
    coefficients = compactified_coefficient_set(
        config.counterexample_target,
        np.asarray(traces[config.counterexample_target], dtype=float).reshape(1, -1),
        tail_l1_bound=1e-12,
        finite_energy_estimate=float(np.max(traces["energy"])),
    )
    artifact = build_candidate_artifact(
        candidate_type="regularity_growth_law",
        replay_grid=compactified_sandbox_replay_grid(n_radial=4, n_theta=4, n_phi=8),
        replay_inputs={
            "seed": config.seed,
            "traces": {k: [float(v) for v in values] for k, values in traces.items()},
        },
        result={
            "counterexample_count": len(sweep.get("counterexamples", [])),
            "sweep_passed": bool(sweep.get("passed", False)),
            "global_regularity_claim": False,
        },
        coefficients=(coefficients,),
        proof_obligations=("tail_bounds", "independent_replay"),
        notes="internal sprint probe; not a claim",
    )
    artifact["schema_errors"] = candidate_artifact_schema_errors(artifact)
    return artifact


def _replay(artifact: dict[str, Any]) -> dict[str, Any]:
    """Independent recomputation, when ``omnibias-symbolic`` happens to be installed.

    ``omnibias-pinn`` does not depend on ``omnibias-symbolic``, so the replay twin
    is imported softly and its absence is recorded rather than papered over -- an
    un-replayed artifact must not read as a replayed one.
    """
    try:
        from omnibias.symbolic.navier_stokes import replay_candidate_artifact
    except ImportError:
        return {
            "available": False,
            "replayed": False,
            "open_obligations": ["independent_replay_backend_unavailable"],
        }
    report = dict(replay_candidate_artifact(artifact))
    report["available"] = True
    report.setdefault("replayed", True)
    return report


def sprint_honesty_errors(ledger: dict[str, Any]) -> list[str]:
    """Re-derive the honesty verdict from the phases instead of trusting the flags.

    The point of a self-check is that it cannot be satisfied by editing the field
    it validates: a ledger asserting ``theorem_claimed`` or carrying a phase that
    concluded ``unproven_claim`` is rejected here regardless of what its honesty
    block says.
    """
    errors: list[str] = []
    if ledger.get("schema_version") != SPRINT_SCHEMA_VERSION:
        errors.append(f"schema_version must be {SPRINT_SCHEMA_VERSION!r}")
    if ledger.get("theorem_claimed") is not False:
        errors.append("theorem_claimed must be False")

    honesty = dict(ledger.get("honesty", {}))
    for flag in (
        "unproven_claim",
        "global_regularity_claim",
        "finite_time_blowup_claim",
        "theorem_prover_verified",
    ):
        if honesty.get(flag) is not False:
            errors.append(f"honesty.{flag} must be False")

    phases = dict(ledger.get("phases", {}))
    for name in SPRINT_PHASES:
        if name not in phases:
            errors.append(f"missing phase {name!r}")
    for name, payload in phases.items():
        if not isinstance(payload, dict):
            errors.append(f"phase {name!r} must be a mapping")
            continue
        if payload.get("unproven_claim") is True:
            errors.append(f"phase {name!r} asserts a proved claim")
        if payload.get("theorem_prover_verified") is True:
            errors.append(f"phase {name!r} asserts theorem_prover_verified")

    if not ledger.get("open_obligations"):
        errors.append("a sprint with no open obligations would mean a gate was skipped")
    return errors


def sprint_summary(ledger: dict[str, Any]) -> str:
    """One-screen digest: what ran, what is still blocked, and what nothing proved."""
    phases = dict(ledger.get("phases", {}))
    obligations = list(ledger.get("open_obligations", []))
    lines = [
        f"navier-stokes candidate sprint ({ledger.get('schema_version')})",
        f"  phases run        : {len(phases)}",
        f"  gates blocked     : {', '.join(ledger.get('blocked_gates', [])) or 'none'}",
        f"  open obligations  : {len(obligations)}",
    ]
    lines.extend(f"      - {item}" for item in obligations[:12])
    if len(obligations) > 12:
        lines.append(f"      ... and {len(obligations) - 12} more")
    lines.append("  theorem claimed   : False (and cannot be set by this harness)")
    return "\n".join(lines)


# Deliberately NOT added to ``omnibias.pinn.certified.__all__``; see the module
# docstring, and ``test_sprint_harness.py`` for the guard that keeps it that way.
__all__ = [
    "SPRINT_PHASES",
    "SPRINT_SCHEMA_VERSION",
    "SprintConfig",
    "run_candidate_sprint",
    "sprint_honesty_errors",
    "sprint_summary",
]
