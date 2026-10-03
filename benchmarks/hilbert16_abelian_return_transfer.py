#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Replay the H4 Picard--Fuchs-to-return transfer audit. Not DRR closure."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from omnibias.dynamics.abelian_return_transfer import report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))
        / "hilbert16"
        / "abelian_return_transfer.json",
    )
    args = parser.parse_args()
    payload = report().to_payload()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "picard_fuchs_verified": payload["picard_fuchs_verified"],
                "conditional_transfer_verified": payload[
                    "conditional_transfer_verified"
                ],
                "drr_transfer_certified": payload["drr_transfer_certified"],
                "full_hilbert16_solved": payload["full_hilbert16_solved"],
            }
        )
    )


if __name__ == "__main__":
    main()
