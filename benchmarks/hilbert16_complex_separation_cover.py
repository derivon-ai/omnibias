#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Replay the complex regular-event separation cover. Not G3 or Hilbert XVI."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from omnibias.dynamics.complex_separation_cover import report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))
        / "hilbert16"
        / "complex_separation_cover.json",
    )
    args = parser.parse_args()
    payload = report().to_payload()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "d_c_endpoint_included": payload["d_c_endpoint_included"],
                "chart_o_endpoint_included": payload[
                    "chart_o_endpoint_included"
                ],
                "complex_separation_event_cover_certified": payload[
                    "complex_separation_event_cover_certified"
                ],
                "physical_overlap_matching_proved": payload[
                    "physical_overlap_matching_proved"
                ],
                "g3_passed": payload["g3_passed"],
                "full_hilbert16_solved": payload["full_hilbert16_solved"],
            }
        )
    )


if __name__ == "__main__":
    main()
