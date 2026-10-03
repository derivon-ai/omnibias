#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
"""DF_1a anchor replay benchmark."""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any

sys.path.insert(0, os.path.dirname(__file__))
from _common import provenance, write_json  # type: ignore[import-not-found]  # noqa: E402
from _gates import gates_block  # type: ignore[import-not-found]  # noqa: E402
from omnibias.dynamics.df1a import (  # noqa: E402
    DF1A_CYCLICITY_BOUND,
    reproduce_df1a_cyclicity,
    verify_df1a_certificate,
)


def _df1a_replay() -> dict[str, Any]:
    certificate = reproduce_df1a_cyclicity()
    ok = verify_df1a_certificate(certificate) and certificate.cyclicity_bound <= DF1A_CYCLICITY_BOUND
    return {
        "name": "g1_df1a_replay",
        "passed": bool(ok),
        "cyclicity_bound": certificate.cyclicity_bound,
        "displacement_bound": certificate.displacement_bound,
        "compactification_certified": verify_df1a_certificate(certificate),
        "entry_exit_certified": certificate.entry_exit.certified,
        "detail": "declared interior model reproduces published <=3 bound",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    started = time.time()
    entries = [_df1a_replay()]
    report = {
        **provenance(
            schema="omnibias.benchmarks.hilbert16_df1a.v1",
            config={"family": "hilbert16_df1a", "full": args.full},
        ),
        "benchmark": "hilbert16_df1a",
        "gates": gates_block(entries),
        "full_hilbert16_solved": False,
        "wall_seconds": time.time() - started,
    }
    artifact = "hilbert16_df1a.json" if args.full else "hilbert16_df1a_smoke.json"
    path = write_json(artifact, report)
    print(f"wrote {path}")
    return 0 if report["gates"]["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
