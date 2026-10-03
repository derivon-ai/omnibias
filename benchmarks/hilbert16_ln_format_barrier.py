#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Replay the H3 direct LN-format barrier. Not G3 or Hilbert XVI."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from omnibias.dynamics.ln_format_barrier import report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))
        / "hilbert16"
        / "ln_format_barrier.json",
    )
    args = parser.parse_args()
    payload = report().to_payload()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "finite_truncations_certified": payload[
                    "finite_truncations_certified"
                ],
                "direct_bounded_format": payload["direct_bounded_format"],
                "normalized_zero_equivalent_route_open": payload[
                    "normalized_zero_equivalent_route_open"
                ],
                "g3_passed": payload["g3_passed"],
                "full_hilbert16_solved": payload["full_hilbert16_solved"],
            }
        )
    )


if __name__ == "__main__":
    main()
