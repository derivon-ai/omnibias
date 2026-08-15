# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Two-plaquette SU(2) Hamiltonian gap (theory 07-04, non-diagonal object).

Certifies λ1-λ0 of one finite Kogut–Susskind Hamiltonian. Not a
Yang-Mills / continuum / mass-gap claim. The G1 tightness factor is
measured, not invented.
"""

from __future__ import annotations

import argparse
import os
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(__file__))
from _common import provenance, write_json  # type: ignore[import-not-found]  # noqa: E402
from _gates import gates_block  # type: ignore[import-not-found]  # noqa: E402

SCRATCH = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))

SMOKE_COUPLINGS = (Fraction(1, 2), Fraction(1), Fraction(2), Fraction(4))
SMOKE_J_MAX = 1
FULL_J_MAX = 2


def _measure_g1(*, j_max: int) -> dict[str, Any]:
    import numpy as np
    from omnibias.geometry.gauge.transfer.hamiltonian import (
        LEHMANN_HOLONOMY_METHOD,
        LEHMANN_STANDARD_METHOD,
        candidate_gap,
        certified_hamiltonian_gap,
        plaquette_holonomy_trial_space,
        su2_two_plaquette_hamiltonian,
    )

    factors: list[float] = []
    fractions: list[float] = []
    ge_generic = True
    sound = True
    certified_lock = False
    for coupling in SMOKE_COUPLINGS:
        hamiltonian = su2_two_plaquette_hamiltonian(coupling, j_max=j_max)
        trial = plaquette_holonomy_trial_space(hamiltonian)
        generic_official = certified_hamiltonian_gap(hamiltonian)
        result = certified_hamiltonian_gap(hamiltonian, trial=trial)
        if result.spectral_gap_lower + 1e-12 < generic_official.spectral_gap_lower:
            ge_generic = False
        generic = candidate_gap(result, LEHMANN_STANDARD_METHOD)
        holonomy = candidate_gap(result, LEHMANN_HOLONOMY_METHOD)
        mid = np.array(
            [[0.5 * (c.lo + c.hi) for c in row] for row in hamiltonian.entries]
        )
        values = np.sort(np.linalg.eigvalsh(0.5 * (mid + mid.T)))
        numerical = float(values[1] - values[0])
        if holonomy + 1e-12 < generic:
            ge_generic = False
        if result.spectral_gap_lower > numerical + 1e-9:
            sound = False
        if generic > 0.0:
            factors.append(holonomy / generic)
        elif holonomy >= generic:
            factors.append(1.0)
        if numerical > 0.0:
            fractions.append(result.spectral_gap_lower / numerical)
        if coupling == Fraction(1, 2) and result.certified and result.spectral_gap_lower > 0.0:
            certified_lock = True
    factors.sort()
    fractions.sort()
    mid_factor = factors[len(factors) // 2] if factors else 0.0
    return {
        "n_hamiltonians": len(SMOKE_COUPLINGS),
        "j_max": j_max,
        "ge_generic": ge_generic,
        "sound": sound,
        "certified_at_lock": certified_lock,
        "measured_factor_min": min(factors) if factors else 0.0,
        "measured_factor_median": mid_factor,
        "measured_factor_max": max(factors) if factors else 0.0,
        "fraction_of_numerical_min": min(fractions) if fractions else 0.0,
        "fraction_of_numerical_median": (
            fractions[len(fractions) // 2] if fractions else 0.0
        ),
        "target_factor_5x": False,
        "note": (
            "G1 compares plaquette-character trials to the computational "
            "standard basis on a non-diagonal Hamiltonian; the factor is "
            "measured, not a claimed 5x"
        ),
    }


def _g3_gauss_and_trace() -> bool:
    from omnibias.geometry.gauge.band._core import su2_transverse_constant
    from omnibias.geometry.gauge.transfer.hamiltonian import (
        legal_triple,
        su2_two_plaquette_hamiltonian,
    )
    from omnibias.geometry.gauge.transfer.trial import su2_holonomy_trace

    hamiltonian = su2_two_plaquette_hamiltonian(Fraction(1, 2), j_max=1)
    two_j_max = 2
    if not all(
        legal_triple(t1, t2, ts, two_j_max=two_j_max)
        for t1, t2, ts in hamiltonian.basis
    ):
        return False
    components = (0.3, -0.2, 0.5)
    bare = su2_holonomy_trace(components, length=1.0, coupling=1.0)
    u00, u01, u10, u11 = su2_transverse_constant(
        components, length=1.0, coupling=1.0
    )
    g00, g01, g10, g11 = su2_transverse_constant(
        (0.1, 0.7, -0.4), length=1.2, coupling=0.8
    )
    gd00, gd01, gd10, gd11 = (
        g00.conjugate(),
        g10.conjugate(),
        g01.conjugate(),
        g11.conjugate(),
    )
    w00 = gd00 * u00 + gd01 * u10
    w01 = gd00 * u01 + gd01 * u11
    w10 = gd10 * u00 + gd11 * u10
    w11 = gd10 * u01 + gd11 * u11
    t00 = w00 * g00 + w01 * g10
    t11 = w10 * g01 + w11 * g11
    transformed = float((t00 + t11).real)
    return abs(transformed - bare) / max(abs(bare), 1e-15) < 1e-14


def _g5_honesty() -> bool:
    from omnibias.geometry.gauge.transfer.certificates import (
        hamiltonian_gap_schema_errors,
        seal_hamiltonian_gap_certificate,
    )
    from omnibias.geometry.gauge.transfer.hamiltonian import (
        certified_hamiltonian_gap,
        su2_two_plaquette_hamiltonian,
    )

    hamiltonian = su2_two_plaquette_hamiltonian(Fraction(1, 2), j_max=1)
    result = certified_hamiltonian_gap(hamiltonian)
    if not result.certified:
        return False
    cert = seal_hamiltonian_gap_certificate(result, hamiltonian)
    if hamiltonian_gap_schema_errors(cert):
        return False
    if cert["continuum_claim"] or cert["honesty"]["yang_mills_claim"]:
        return False
    return True


def _g6_kernel_obligation() -> bool:
    from omnibias.core.proof.lean_check import generate_obligation
    from omnibias.geometry.gauge.transfer.certificates import (
        seal_hamiltonian_gap_certificate,
    )
    from omnibias.geometry.gauge.transfer.hamiltonian import (
        certified_hamiltonian_gap,
        su2_two_plaquette_hamiltonian,
    )

    hamiltonian = su2_two_plaquette_hamiltonian(Fraction(1, 2), j_max=1)
    result = certified_hamiltonian_gap(hamiltonian)
    if not result.certified:
        return False
    source = generate_obligation(seal_hamiltonian_gap_certificate(result, hamiltonian))
    return source is not None and "spectral_gap_pos" in source


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    j_max = FULL_J_MAX if args.full else SMOKE_J_MAX

    g1 = _measure_g1(j_max=j_max)
    g2 = bool(g1["sound"])
    g3 = _g3_gauss_and_trace()
    g5 = _g5_honesty()
    g6 = _g6_kernel_obligation()

    entries: list[dict[str, Any]] = [
        {
            "name": "g1_ge_generic_measured",
            "passed": bool(g1["ge_generic"] and g1["n_hamiltonians"] >= 4),
            "in_ci_all_passed": True,
            "measured_factor_median": g1["measured_factor_median"],
            "n_hamiltonians": g1["n_hamiltonians"],
        },
        {
            "name": "g2_soundness",
            "passed": bool(g2),
            "in_ci_all_passed": True,
        },
        {
            "name": "g3_gauss_law_and_trace",
            "passed": bool(g3),
            "in_ci_all_passed": True,
        },
        {
            "name": "g5_honesty",
            "passed": bool(g5),
            "in_ci_all_passed": True,
        },
        {
            "name": "g6_kernel_obligation",
            "passed": bool(g6),
            "in_ci_all_passed": True,
        },
        {
            "name": "certified_gap_at_lock",
            "passed": bool(g1["certified_at_lock"]),
            "in_ci_all_passed": True,
        },
    ]
    payload: dict[str, Any] = provenance(
        schema="omnibias.benchmark.gauge_two_plaquette_gap.v1",
        config={"mode": "full" if args.full else "smoke", "j_max": j_max},
    )
    payload["g1"] = g1
    payload["gates"] = gates_block(entries)
    payload["honesty"] = {
        "yang_mills": False,
        "mass_gap": False,
        "continuum_claim": False,
        "fixed_hamiltonian": True,
        "note": (
            "certified spectral gap of one two-plaquette Hamiltonian at one "
            "coupling; the continuum limit is not taken"
        ),
    }
    if args.full:
        dest = SCRATCH / "gauge_gap" / "gauge_two_plaquette_gap.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(__import__("json").dumps(payload, indent=2) + "\n")
        print(f"wrote {dest}")
    else:
        print(f"wrote {write_json('gauge_two_plaquette_gap_smoke.json', payload)}")
    return 0 if payload["gates"]["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
