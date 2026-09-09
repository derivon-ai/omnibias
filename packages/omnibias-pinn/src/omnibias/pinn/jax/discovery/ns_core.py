# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""NS-core axis-regular profile search (theory 07-14).

The family and the exact checker live in
:mod:`omnibias.pinn.certified.anisotropic` (Fraction / Interval only).
This harness wires :func:`~omnibias.core.proof.discovery.run_discovery`.
It does not grow :mod:`omnibias.pinn.certified.navier_stokes`.
"""

from __future__ import annotations

from typing import Any

from omnibias.core.proof.discovery import DiscoveryResult, run_discovery
from omnibias.pinn.certified.anisotropic import (
    NS_CORE_ORIGIN,
    NS_CORE_SECOND_WITNESS,
    NSCoreProfileFamily,
    honesty_payload,
    ns_core_statement,
)


def run_ns_core_discovery(
    *,
    budget: int = 64,
    collect: bool = False,
    proposer: str = "score_guided",
) -> DiscoveryResult:
    """Search the finite ``(c, a, b)`` box. ``budget == 0`` is incomplete."""
    return run_discovery(
        ns_core_statement(),
        NSCoreProfileFamily(),
        proposer,
        budget=budget,
        collect=collect,
    )


def run_ns_core_search(**kwargs: Any) -> dict[str, Any]:
    """JSON-able discovery payload, including a non-origin witness scan."""
    result = run_ns_core_discovery(**kwargs)
    collected = run_ns_core_discovery(budget=max(int(kwargs.get("budget", 64)), 27), collect=True)
    solutions = list(collected.solutions)
    non_origin = [item for item in solutions if item != NS_CORE_ORIGIN]
    payload = result.as_dict()
    payload["solutions"] = [
        [str(part) for part in item] for item in solutions
    ]
    payload["non_origin_witnesses"] = [
        [str(part) for part in item] for item in non_origin
    ]
    payload["second_witness"] = [str(part) for part in NS_CORE_SECOND_WITNESS]
    payload["origin_only"] = bool(solutions) and not non_origin
    payload["honesty"] = honesty_payload()
    return payload


__all__ = [
    "run_ns_core_discovery",
    "run_ns_core_search",
]
