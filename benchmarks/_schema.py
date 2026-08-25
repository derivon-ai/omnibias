# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Artifact-schema checks for ``docs/benchmarks/*.json`` (theory 06-01 G4/G5).

The protocol in ``theory/06-program/01-acceptance-gates-and-benchmarks.md``
asks every artifact for ``gates``, ``baseline``, and ``seeds``. Existing
files predate that envelope. Quietly excluding them would delete the gate;
this module **classifies** each file and records why ``seeds`` / ``baseline``
may be absent, then enforces the checks that do apply.

Classes
-------
``provenance``
    Has ``schema`` and ``generated_utc``. Must carry a well-shaped ``gates``
    block *if* ``gates`` is present. A present ``baseline`` must have ``name``.
``campaign_envelope``
    CCF / IPM / Boussinesq smokes that nest gates under ``absolute_gates``.
``legacy_cost``
    Early cost / PINN artifacts with ``schema`` but no ``gates`` block yet.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = REPO_ROOT / "docs" / "benchmarks"

# Deterministic protocol smokes: no stochastic seed axis, no named competitor.
# Presence of a baseline-without-name still fails (G5).
DETERMINISTIC_EXEMPT_REASON = (
    "deterministic protocol / algebra smoke: no stochastic seed axis; "
    "baseline is the named gate itself, not a competitor library"
)

CAMPAIGN_EXEMPT_REASON = (
    "campaign envelope: gates live under absolute_gates; not the "
    "provenance() header used by theory-program smokes"
)

LEGACY_COST_EXEMPT_REASON = (
    "legacy cost / PINN suite predates the gates block; schema + generated_utc "
    "are present. New artifacts must use provenance() and emit gates."
)


def classify_artifact(payload: dict[str, Any]) -> str:
    """Return ``provenance``, ``campaign_envelope``, or ``legacy_cost``."""
    if "absolute_gates" in payload and "schema" not in payload:
        return "campaign_envelope"
    if "schema" in payload and "generated_utc" in payload:
        if "gates" in payload:
            return "provenance"
        return "legacy_cost"
    if "absolute_gates" in payload or "benchmark" in payload:
        return "campaign_envelope"
    raise ValueError("unrecognized artifact envelope")


def exemption_reason(payload: dict[str, Any]) -> str | None:
    """Why ``seeds`` / ``baseline`` may be missing, or ``None`` if required."""
    kind = classify_artifact(payload)
    if kind == "campaign_envelope":
        return CAMPAIGN_EXEMPT_REASON
    if kind == "legacy_cost":
        return LEGACY_COST_EXEMPT_REASON
    if "seeds" in payload and "baseline" in payload:
        return None
    return DETERMINISTIC_EXEMPT_REASON


def validate_artifact(payload: Any, *, name: str) -> list[str]:
    """Return human-readable errors. Empty means the artifact conforms."""
    errors: list[str] = []
    if not isinstance(payload, dict):
        return [f"{name}: artifact must be a JSON object"]
    if "baseline" in payload:
        baseline = payload["baseline"]
        if not isinstance(baseline, dict) or not str(baseline.get("name") or "").strip():
            errors.append(f"{name}: baseline block is missing a name (06-01 G5)")
    try:
        kind = classify_artifact(payload)
    except ValueError as exc:
        errors.append(f"{name}: {exc}")
        return errors
    if kind == "provenance":
        gates = payload.get("gates")
        if not isinstance(gates, dict):
            errors.append(f"{name}: gates must be an object")
        elif "all_passed" not in gates or "entries" not in gates:
            errors.append(f"{name}: gates must contain all_passed and entries")
        elif not isinstance(gates["entries"], list):
            errors.append(f"{name}: gates.entries must be a list")
        if "seeds" in payload and "per_seed" in payload:
            seeds = payload["seeds"]
            per_seed = payload["per_seed"]
            if isinstance(seeds, list) and isinstance(per_seed, list) and per_seed:
                seen = {
                    row.get("seed")
                    for row in per_seed
                    if isinstance(row, dict)
                }
                missing = [s for s in seeds if s not in seen]
                if missing:
                    errors.append(
                        f"{name}: per_seed is missing seeds {missing!r}"
                    )
    elif kind == "campaign_envelope":
        if "absolute_gates" in payload:
            ag = payload["absolute_gates"]
            if not isinstance(ag, dict):
                errors.append(f"{name}: absolute_gates must be an object")
            else:
                gates = ag.get("gates")
                if not isinstance(gates, dict) or "all_passed" not in gates:
                    errors.append(
                        f"{name}: absolute_gates.gates must contain all_passed"
                    )
        elif "metrics" not in payload and "rows" not in payload:
            errors.append(
                f"{name}: campaign envelope needs absolute_gates, metrics, or rows"
            )
    elif kind == "legacy_cost":
        if "schema" not in payload or "generated_utc" not in payload:
            errors.append(f"{name}: legacy_cost artifact lost schema/generated_utc")
    _ = exemption_reason(payload)
    return errors


def iter_benchmark_artifacts() -> list[Path]:
    return sorted(BENCHMARK_DIR.glob("*.json"))


def validate_committed_artifacts() -> list[str]:
    errors: list[str] = []
    paths = iter_benchmark_artifacts()
    if not paths:
        return ["docs/benchmarks/ contains no JSON artifacts"]
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            errors.append(f"{path.name}: invalid JSON ({exc})")
            continue
        errors.extend(validate_artifact(payload, name=path.name))
    return errors


__all__ = [
    "BENCHMARK_DIR",
    "CAMPAIGN_EXEMPT_REASON",
    "DETERMINISTIC_EXEMPT_REASON",
    "LEGACY_COST_EXEMPT_REASON",
    "classify_artifact",
    "exemption_reason",
    "iter_benchmark_artifacts",
    "validate_artifact",
    "validate_committed_artifacts",
]
