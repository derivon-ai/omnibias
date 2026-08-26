# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Jet-Padé CCF profile diagnostic (library leftover, not the 1e-13 campaign).

Runs Padé on a fixed geometric profile snapshot. Optional ``--multipack``
and ``--irregular`` arms are named flags, not default CI. Stretch stays
``1e-13``. ``navier_stokes_proof_claim`` stays False.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(__file__))
from _common import provenance, write_json  # type: ignore[import-not-found]  # noqa: E402
from _gates import (  # type: ignore[import-not-found]  # noqa: E402
    CCF_STRETCH_RESIDUAL_GATE,
    gates_block,
)

SCRATCH = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--multipack", action="store_true")
    parser.add_argument("--irregular", action="store_true")
    args = parser.parse_args()

    from omnibias.difference._core.ccf_profile import (
        diagnose_ccf_profile,
        geometric_profile_jet,
        honesty_payload,
        optional_irregular_arm,
        optional_multipack_arm,
    )

    if float(CCF_STRETCH_RESIDUAL_GATE) != 1e-13:
        raise RuntimeError("CCF_STRETCH_RESIDUAL_GATE must stay 1e-13")

    coeffs = geometric_profile_jet(radius=2.0, order=12)
    report = diagnose_ccf_profile(coeffs, snapshot="geometric_radius_2")
    payload_diag = report.to_payload()
    loc = report.estimate.location
    pole_ok = (
        loc is not None
        and abs(loc.real - 2.0) < 0.25
        and abs(loc.imag) < 0.25
        and not report.estimate.failed
    )
    honesty = honesty_payload()
    entries: list[dict[str, Any]] = [
        {
            "name": "pade_locates_profile_pole",
            "passed": bool(pole_ok),
        },
        {
            "name": "not_a_residual_substitute",
            "passed": honesty["residual_substitute"] is False,
        },
        {
            "name": "stretch_untouched",
            "passed": float(CCF_STRETCH_RESIDUAL_GATE) == 1e-13,
        },
        {
            "name": "honesty",
            "passed": honesty["navier_stokes_proof_claim"] is False
            and honesty["whole_line_certified"] is False
            and honesty["dirichlet_consumed"] is False,
        },
    ]
    extra: dict[str, Any] = {}
    if args.multipack or args.full:
        extra["multipack"] = optional_multipack_arm()
    if args.irregular or args.full:
        extra["irregular"] = optional_irregular_arm()
    payload: dict[str, Any] = provenance(
        schema="omnibias.benchmark.ccf_pade_profile.v1",
        config={
            "mode": "full" if args.full else "smoke",
            "multipack": bool(args.multipack or args.full),
            "irregular": bool(args.irregular or args.full),
        },
    )
    payload["diagnostic"] = payload_diag
    payload["stretch_gate"] = float(CCF_STRETCH_RESIDUAL_GATE)
    payload.update(extra)
    payload["gates"] = gates_block(entries)
    payload["honesty"] = honesty
    if args.full:
        dest = SCRATCH / "ccf" / "ccf_pade_profile.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(__import__("json").dumps(payload, indent=2) + "\n")
        print(f"wrote {dest}")
    else:
        print(f"wrote {write_json('ccf_pade_profile_smoke.json', payload)}")
    return 0 if payload["gates"]["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
