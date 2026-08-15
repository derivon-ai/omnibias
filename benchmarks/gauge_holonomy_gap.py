# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Holonomy trial spaces on finite transfer matrices (theory 07-04).

Certifies a spectral gap for one fixed matrix at one spacing. Not a
Yang-Mills / continuum / mass-gap claim. The G1 tightness factor is
measured, not invented.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(__file__))
from _common import provenance, write_json  # type: ignore[import-not-found]  # noqa: E402
from _gates import gates_block  # type: ignore[import-not-found]  # noqa: E402

SCRATCH = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))


def _true_gap(transfer: Any) -> float:
    ratio = transfer.exact_subdominant_ratio()
    assert ratio is not None
    return -math.log(0.5 * (ratio.lo + ratio.hi))


def _dense_suite() -> list[Any]:
    from omnibias.geometry.gauge.transfer.matrices import (
        su2_class_angle_transfer,
        u1_heat_kernel_transfer,
    )

    suite: list[Any] = []
    for n_max in (2, 3, 4):
        for coupling in (0.4, 0.6, 0.8, 1.0):
            suite.append(u1_heat_kernel_transfer(coupling, n_max=n_max, basis="angle"))
    for max_dynkin in (3, 4, 5):
        for coupling in (0.5, 0.8, 1.0):
            suite.append(su2_class_angle_transfer(coupling, max_dynkin=max_dynkin))
    return suite


def _measure_g1(suite: list[Any]) -> dict[str, Any]:
    from omnibias.geometry.gauge.transfer.gap import certified_transfer_matrix_gap
    from omnibias.geometry.gauge.transfer.trial import holonomy_trial_space

    factors: list[float] = []
    fractions: list[float] = []
    ge_generic = True
    sound = True
    for transfer in suite:
        generic = certified_transfer_matrix_gap(transfer)
        trial = holonomy_trial_space(transfer)
        holonomy = certified_transfer_matrix_gap(transfer, trial=trial)
        true = _true_gap(transfer)
        if holonomy.spectral_gap_lower + 1e-12 < generic.spectral_gap_lower:
            ge_generic = False
        if holonomy.spectral_gap_lower > true + 1e-9:
            sound = False
        if generic.spectral_gap_lower > 0.0:
            factors.append(holonomy.spectral_gap_lower / generic.spectral_gap_lower)
        if true > 0.0:
            fractions.append(holonomy.spectral_gap_lower / true)
    factors.sort()
    fractions.sort()
    mid = factors[len(factors) // 2] if factors else 0.0
    return {
        "n_matrices": len(suite),
        "ge_generic": ge_generic,
        "sound": sound,
        "measured_factor_min": min(factors) if factors else 0.0,
        "measured_factor_median": mid,
        "measured_factor_max": max(factors) if factors else 0.0,
        "fraction_of_numerical_min": min(fractions) if fractions else 0.0,
        "fraction_of_numerical_median": (
            fractions[len(fractions) // 2] if fractions else 0.0
        ),
        "target_factor_5x": False,
        "note": (
            "character-aligned holonomy trials on already-diagonal eigenbases "
            "cannot beat the constructor partner chain; the 5x target is a "
            "measured quantity, not a claimed improvement"
        ),
    }


def _synthetic_soundness(n: int) -> bool:
    import numpy as np
    from omnibias.core.verified.interval import Interval
    from omnibias.geometry.gauge.transfer.gap import certified_transfer_matrix_gap
    from omnibias.geometry.gauge.transfer.matrices import TransferMatrix
    from omnibias.geometry.gauge.transfer.trial import holonomy_trial_space

    evals = (1.0, 0.55, 0.3, 0.12)
    true_gap = -math.log(evals[1] / evals[0])
    for seed in range(n):
        rng = np.random.default_rng(seed)
        raw = rng.normal(size=(4, 4))
        q, _ = np.linalg.qr(raw)
        mid = q @ np.diag(evals) @ q.T
        mid = 0.5 * (mid + mid.T)
        entries = tuple(
            tuple(Interval.point(float(mid[i, j])) for j in range(4)) for i in range(4)
        )
        transfer = TransferMatrix(
            model="synthetic_spd",
            basis="angle",
            entries=entries,
            mode_labels=tuple(f"e{i}" for i in range(4)),
            exact_eigenvalues=None,
            parameters={"builder": "synthetic", "coupling": "1"},
            perron_vector=tuple(float(q[i, 0]) for i in range(4)),
            subdominant_vectors=tuple(
                tuple(float(q[i, j]) for i in range(4)) for j in range(1, 4)
            ),
            symmetric=True,
        )
        result = certified_transfer_matrix_gap(
            transfer, trial=holonomy_trial_space(transfer)
        )
        if result.spectral_gap_lower > true_gap + 1e-8:
            return False
    return True


def _g3_gauge_invariance() -> bool:
    from omnibias.geometry.gauge.band._core import su2_transverse_constant
    from omnibias.geometry.gauge.transfer.trial import su2_holonomy_trace

    components = (0.3, -0.2, 0.5)
    bare = su2_holonomy_trace(components, length=1.0, coupling=1.0)
    u00, u01, u10, u11 = su2_transverse_constant(
        components, length=1.0, coupling=1.0
    )
    g00, g01, g10, g11 = su2_transverse_constant((0.1, 0.7, -0.4), length=1.2, coupling=0.8)
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


def _g6_kernel_obligation() -> bool:
    from omnibias.core.proof.lean_check import generate_obligation
    from omnibias.geometry.gauge.transfer.certificates import (
        seal_transfer_gap_certificate,
    )
    from omnibias.geometry.gauge.transfer.gap import certified_transfer_matrix_gap
    from omnibias.geometry.gauge.transfer.matrices import su2_heat_kernel_transfer
    from omnibias.geometry.gauge.transfer.trial import holonomy_trial_space

    transfer = su2_heat_kernel_transfer(0.8, max_dynkin=3)
    result = certified_transfer_matrix_gap(
        transfer, trial=holonomy_trial_space(transfer)
    )
    source = generate_obligation(seal_transfer_gap_certificate(result, transfer))
    return source is not None and "spectral_gap_pos" in source


def _g5_honesty() -> bool:
    from omnibias.geometry.gauge.transfer.certificates import (
        seal_transfer_gap_certificate,
        transfer_gap_schema_errors,
    )
    from omnibias.geometry.gauge.transfer.gap import certified_transfer_matrix_gap
    from omnibias.geometry.gauge.transfer.matrices import su2_class_angle_transfer
    from omnibias.geometry.gauge.transfer.trial import holonomy_trial_space

    transfer = su2_class_angle_transfer(0.8, max_dynkin=3)
    result = certified_transfer_matrix_gap(
        transfer, trial=holonomy_trial_space(transfer)
    )
    cert = seal_transfer_gap_certificate(result, transfer)
    if transfer_gap_schema_errors(cert):
        return False
    if cert["continuum_claim"] or cert["honesty"]["yang_mills_claim"]:
        return False
    return cert["trial_gram_condition"] is not None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()

    suite = _dense_suite()
    g1 = _measure_g1(suite)
    n_synth = 1000 if args.full else 32
    g2 = g1["sound"] and _synthetic_soundness(n_synth)
    g3 = _g3_gauge_invariance()
    g5 = _g5_honesty()
    g6 = _g6_kernel_obligation()

    entries: list[dict[str, Any]] = [
        {
            "name": "g1_ge_generic_measured",
            "passed": bool(g1["ge_generic"] and g1["n_matrices"] >= 20),
            "in_ci_all_passed": True,
            "measured_factor_median": g1["measured_factor_median"],
            "n_matrices": g1["n_matrices"],
        },
        {
            "name": "g2_soundness",
            "passed": bool(g2),
            "in_ci_all_passed": True,
            "n_synthetic": n_synth,
        },
        {
            "name": "g3_gauge_invariance",
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
    ]
    payload: dict[str, Any] = provenance(
        schema="omnibias.benchmark.gauge_holonomy_gap.v1",
        config={"mode": "full" if args.full else "smoke", "n_synthetic": n_synth},
    )
    payload["g1"] = g1
    payload["gates"] = gates_block(entries)
    payload["honesty"] = {
        "yang_mills": False,
        "mass_gap": False,
        "continuum_claim": False,
        "fixed_matrix": True,
        "note": (
            "certified spectral gap of one fixed transfer matrix at one "
            "spacing; the continuum limit is not taken"
        ),
    }
    if args.full:
        dest = SCRATCH / "gauge_gap" / "gauge_holonomy_gap.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(__import__("json").dumps(payload, indent=2) + "\n")
        print(f"wrote {dest}")
    else:
        print(f"wrote {write_json('gauge_holonomy_gap_smoke.json', payload)}")
    return 0 if payload["gates"]["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
