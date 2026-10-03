#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Replay the H5 finite Bautin-stabilization audit. Not G2 or Hilbert XVI."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from omnibias.dynamics.bautin_stabilization_barrier import report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))
        / "hilbert16"
        / "bautin_stabilization.json",
    )
    args = parser.parse_args()
    payload = report().to_payload()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "finite_order_stabilization_verified": payload[
                    "finite_order_stabilization_verified"
                ],
                "formal_jet_counterexample_verified": payload[
                    "formal_jet_counterexample_verified"
                ],
                "bautin_ideal_stabilization_proved": payload[
                    "bautin_ideal_stabilization_proved"
                ],
                "g2_passed": payload["g2_passed"],
                "full_hilbert16_solved": payload["full_hilbert16_solved"],
            }
        )
    )


if __name__ == "__main__":
    main()
