# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Frontier 07-05 G5: arrangement-adapted SOS bases.

Smoke earns G5: on a named positivity set the adapted basis
certifies at a degree at least 2 lower than total-degree on
at least half the problems. Failures are reported, not
dropped. Not a continuum spectral-gap or Yang-Mills claim.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(__file__))
from _common import provenance, write_json  # type: ignore[import-not-found]  # noqa: E402
from _gates import gates_block  # type: ignore[import-not-found]  # noqa: E402
from omnibias.sos.certify import degree_reduction_report

SCRATCH = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))


def _run_g5() -> dict[str, Any]:
    rows = degree_reduction_report()
    wins = sum(1 for r in rows if r["win"])
    ok = wins >= (len(rows) + 1) // 2
    return {
        "name": "g5_degree_reduction",
        "passed": ok,
        "wins": wins,
        "n": len(rows),
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args(argv)
    full = bool(args.full)
    artifact = "sos_adapted_basis.json" if full else "sos_adapted_basis_smoke.json"
    t0 = time.perf_counter()
    entries = [_run_g5()]
    for e in entries:
        print(e["name"], "ok" if e["passed"] else "FAIL")
        if not e["passed"]:
            raise AssertionError(f"{e['name']} failed: {e}")
    payload = {
        **provenance(
            schema="omnibias.benchmarks.sos_adapted_basis.v1",
            config={"family": "sos_adapted_basis", "full": full},
        ),
        "gates": dict(gates_block(entries)),
        "wall_seconds": time.perf_counter() - t0,
    }
    if full:
        dest = SCRATCH / "spectral"
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / artifact
        path.write_text(__import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8")
    else:
        path = write_json(artifact, payload)
    print(f"wrote {path}")
    return payload


if __name__ == "__main__":
    main()
