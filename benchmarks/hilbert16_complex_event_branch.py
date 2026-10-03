#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Replay the local complex normal-form event branch. Not G3 or Hilbert XVI."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from omnibias.dynamics.complex_event_branch import report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))
        / "hilbert16"
        / "complex_event_branch.json",
    )
    args = parser.parse_args()
    payload = report().to_payload()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "complex_normal_event_branch_certified": payload[
                    "complex_normal_event_branch_certified"
                ],
                "real_first_hit_replayed": payload["real_first_hit_replayed"],
                "complex_physical_return_family_certified": payload[
                    "complex_physical_return_family_certified"
                ],
                "g3_passed": payload["g3_passed"],
                "full_hilbert16_solved": payload["full_hilbert16_solved"],
            }
        )
    )


if __name__ == "__main__":
    main()
