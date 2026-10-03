#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Replay the H6 Songling reproduction-readiness audit."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from omnibias.dynamics.songling_lower_bound import report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))
        / "hilbert16"
        / "songling_lower_bound.json",
    )
    args = parser.parse_args()
    payload = report().to_payload()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "published_h2_lower_bound": payload[
                    "published_h2_lower_bound"
                ],
                "published_certificate_precedes_omnibias": payload[
                    "published_certificate_precedes_omnibias"
                ],
                "governing_epsilon_sign_resolved": payload[
                    "governing_epsilon_sign_resolved"
                ],
                "four_hyperbolic_returns_certified": payload[
                    "four_hyperbolic_returns_certified"
                ],
                "full_hilbert16_solved": payload["full_hilbert16_solved"],
            }
        )
    )


if __name__ == "__main__":
    main()
