#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Finite replay of parametric-sep chart-O matching-chart Lohner cover. Not G1."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from omnibias.dynamics.stage_c_origin_span import report

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))
        / "hilbert16"
        / "stage_c_origin_span.json",
    )
    args = parser.parse_args()
    payload = report().to_payload()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    honesty = payload["honesty"]
    print(
        json.dumps(
            {
                "stage_c_origin_span": payload["stage_c_origin_span"],
                "g1_passed": honesty["g1_passed"],
                "full_hilbert16_solved": honesty["full_hilbert16_solved"],
            }
        )
    )


if __name__ == "__main__":
    main()
