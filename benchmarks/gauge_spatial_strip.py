# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Finite 2+1-D spatial-strip transfer gap, RP, and cluster tail.

Certifies one finite SU(2) class-angle strip. Not a Yang-Mills /
continuum / Osterwalder-Seiler claim.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(__file__))
from _common import provenance, write_json  # type: ignore[import-not-found]  # noqa: E402
from _gates import gates_block  # type: ignore[import-not-found]  # noqa: E402

SCRATCH = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))

SMOKE_SITES = 2
SMOKE_ANGLES = 4
FULL_ANGLES = 6


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    n_angles = FULL_ANGLES if args.full else SMOKE_ANGLES

    from omnibias.geometry.gauge.transfer.certificates import (
        seal_strip_rp_certificate,
        seal_transfer_gap_certificate,
        strip_rp_schema_errors,
        transfer_gap_schema_errors,
    )
    from omnibias.geometry.gauge.transfer.gap import certified_transfer_matrix_gap
    from omnibias.geometry.gauge.transfer.strip import (
        STRIP_COUPLING_LOCK,
        certified_strip_cluster_tail,
        certified_strip_reflection_positivity,
        su2_spatial_strip_transfer,
    )

    transfer = su2_spatial_strip_transfer(
        STRIP_COUPLING_LOCK, n_sites=SMOKE_SITES, n_angles=n_angles
    )
    gap = certified_transfer_matrix_gap(transfer)
    rp = certified_strip_reflection_positivity(transfer)
    cluster = certified_strip_cluster_tail(transfer, n_keep=2)
    honesty = True
    if gap.certified:
        sealed_gap = seal_transfer_gap_certificate(gap, transfer)
        honesty = honesty and not transfer_gap_schema_errors(sealed_gap)
        honesty = honesty and sealed_gap["honesty"]["yang_mills_claim"] is False
    if rp.certified:
        sealed_rp = seal_strip_rp_certificate(rp, transfer)
        honesty = honesty and not strip_rp_schema_errors(sealed_rp)
        honesty = honesty and sealed_rp["honesty"]["yang_mills_claim"] is False

    entries: list[dict[str, Any]] = [
        {
            "name": "strip_gap_certified",
            "passed": bool(gap.certified and gap.spectral_gap_lower > 0.0),
            "in_ci_all_passed": True,
        },
        {
            "name": "strip_reflection_positivity",
            "passed": bool(rp.certified and all(form.lo >= 0.0 for form in rp.forms)),
            "in_ci_all_passed": True,
        },
        {
            "name": "strip_cluster_tail_contains_sample",
            "passed": bool(cluster.certified and cluster.tail.contains(cluster.sample)),
            "in_ci_all_passed": True,
        },
        {
            "name": "honesty",
            "passed": bool(honesty),
            "in_ci_all_passed": True,
        },
    ]
    payload: dict[str, Any] = provenance(
        schema="omnibias.benchmark.gauge_spatial_strip.v1",
        config={
            "mode": "full" if args.full else "smoke",
            "n_sites": SMOKE_SITES,
            "n_angles": n_angles,
        },
    )
    payload["gap"] = {
        "certified": gap.certified,
        "spectral_gap_lower": gap.spectral_gap_lower,
        "dimension": gap.dimension,
        "method": gap.method,
    }
    payload["reflection_positivity"] = {
        "certified": rp.certified,
        "n_forms": len(rp.forms),
        "min_lo": min(form.lo for form in rp.forms),
    }
    payload["cluster"] = {
        "certified": cluster.certified,
        "n_keep": cluster.n_keep,
        "sample": cluster.sample,
        "tail_lo": cluster.tail.lo,
        "tail_hi": cluster.tail.hi,
    }
    payload["gates"] = gates_block(entries)
    payload["honesty"] = {
        "yang_mills": False,
        "mass_gap": False,
        "continuum_claim": False,
        "osterwalder_seiler": False,
        "fixed_matrix": True,
        "note": (
            "certified gap / RP / cluster tail of one finite spatial-strip "
            "transfer; the continuum limit is not taken"
        ),
    }
    if args.full:
        dest = SCRATCH / "gauge_gap" / "gauge_spatial_strip.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(__import__("json").dumps(payload, indent=2) + "\n")
        print(f"wrote {dest}")
    else:
        print(f"wrote {write_json('gauge_spatial_strip_smoke.json', payload)}")
    return 0 if payload["gates"]["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
