#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Replay H7 polygonal barriers and the reduced SOS obstruction audit."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from omnibias.geometry.part_a_obstruction import (
    audit_part_a_obstruction_route,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))
        / "hilbert16"
        / "part_a_obstruction.json",
    )
    args = parser.parse_args()
    payload = audit_part_a_obstruction_route(
        run_route2_lp=args.full,
    ).to_payload()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "polygonal_layout_valid": payload["polygonal_layout_valid"],
                "route2_lp_candidate_found": payload[
                    "route2_lp_candidate_found"
                ],
                "toy_positivstellensatz_empty_set_certified": payload[
                    "toy_positivstellensatz_empty_set_certified"
                ],
                "octic_scheme_obstructed": payload[
                    "octic_scheme_obstructed"
                ],
                "part_a_22_oval_realized": payload[
                    "part_a_22_oval_realized"
                ],
                "full_hilbert16_solved": payload["full_hilbert16_solved"],
            }
        )
    )


if __name__ == "__main__":
    main()
