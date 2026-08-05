# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Buckingham-Pi benchmark: recover known dimensionless groups, exactly.

For two textbook systems -- pipe/sphere drag (Reynolds number) and the simple
pendulum -- we build the unit-dimension matrix and read off the dimensionless
:math:`\\Pi`-groups from its exact integer null space.  We then show the
*dimensional-analysis prior* for discovery: filtering a candidate monomial
library down to its dimensionless members.

Everything is exact integer arithmetic (no floating point), so the recovered
groups are checked for byte-for-byte equality with the known answers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from omnibias.symbolic.dimensional import (
    DimensionalSystem,
    PiGroup,
    buckingham_pi_groups,
    filter_dimensionless_monomials,
    is_dimensionless,
    n_dimensionless_groups,
)


def reynolds_system() -> DimensionalSystem:
    """Fluid past a body: density, velocity, length scale, dynamic viscosity."""
    return DimensionalSystem.from_dimensions(
        {
            "rho": {"M": 1, "L": -3},
            "U": {"L": 1, "T": -1},
            "L": {"L": 1},
            "mu": {"M": 1, "L": -1, "T": -1},
        },
        base_dimensions=["M", "L", "T"],
    )


def pendulum_system() -> DimensionalSystem:
    """Simple pendulum: period, length, gravity, bob mass (mass drops out)."""
    return DimensionalSystem.from_dimensions(
        {
            "t": {"T": 1},
            "L": {"L": 1},
            "g": {"L": 1, "T": -2},
            "m": {"M": 1},
        },
        base_dimensions=["M", "L", "T"],
    )


def _canonical(group: PiGroup) -> dict[str, int]:
    powers = group.as_dict()
    for value in powers.values():
        if value != 0:
            if value < 0:
                return {k: -v for k, v in powers.items()}
            return powers
    return powers


def evaluate_benchmark() -> dict[str, Any]:
    """Recover the Reynolds and pendulum groups and filter a candidate library."""
    reynolds = reynolds_system()
    pendulum = pendulum_system()

    re_groups = buckingham_pi_groups(reynolds)
    pen_groups = buckingham_pi_groups(pendulum)

    re_recovered = _canonical(re_groups[0])
    pen_recovered = _canonical(pen_groups[0])

    # Dimensional-analysis prior: keep only dimensionless monomials from a library.
    pendulum_library = [
        {"t": 1},
        {"t": 2, "g": 1, "L": -1},
        {"t": 1, "g": 1},
        {"m": 1, "t": 2, "g": 1, "L": -1},
        {"L": 1, "g": -1, "t": 2},
    ]
    kept = filter_dimensionless_monomials(pendulum, pendulum_library)

    return {
        "reynolds": {
            "n_groups": n_dimensionless_groups(reynolds),
            "recovered_group": re_recovered,
            "formula": re_groups[0].formula(),
            "matches_reynolds_number": re_recovered == {"rho": 1, "U": 1, "L": 1, "mu": -1},
            "is_dimensionless": is_dimensionless(reynolds, re_groups[0].as_dict()),
        },
        "pendulum": {
            "n_groups": n_dimensionless_groups(pendulum),
            "recovered_group": pen_recovered,
            "formula": pen_groups[0].formula(),
            "mass_exponent_is_zero": pen_recovered.get("m", 0) == 0,
            "matches_period_law": pen_recovered == {"t": 2, "L": -1, "g": 1, "m": 0},
            "is_dimensionless": is_dimensionless(pendulum, pen_groups[0].as_dict()),
        },
        "library_filter": {
            "candidates": pendulum_library,
            "dimensionless_kept": kept,
        },
    }


def write_artifacts(results: dict[str, Any], out_dir: Path) -> None:
    """Write the benchmark report to ``out_dir/report.json``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(results, indent=2, sort_keys=True))
