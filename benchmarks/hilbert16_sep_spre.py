#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Finite replay of sep*S_pre on every sep in (0, 1]. Not G1."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from omnibias.dynamics.sep_spre import report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))
        / "hilbert16"
        / "sep_spre.json",
    )
    args = parser.parse_args()
    payload = report().to_payload()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    honesty = payload["honesty"]
    print(
        json.dumps(
            {
                "sep_spre": payload["sep_spre"],
                "g1_passed": honesty["g1_passed"],
                "full_hilbert16_solved": honesty["full_hilbert16_solved"],
            }
        )
    )


if __name__ == "__main__":
    main()
