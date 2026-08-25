# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Dump named gate constants from ``benchmarks/_gates.py`` (theory 06-01).

Loosening a constant needs a reason in the same diff. This script does not
rewrite thresholds; it prints them so a reviewer can diff the introducing
commit against HEAD.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))

import _gates  # noqa: E402


def named_thresholds() -> dict[str, float]:
    out: dict[str, float] = {}
    for name in sorted(dir(_gates)):
        if name.startswith("_"):
            continue
        if "GATE" not in name and "LAMBDA" not in name:
            continue
        value = getattr(_gates, name)
        if isinstance(value, (int, float)):
            out[name] = float(value)
    return out


def main() -> None:
    rows = named_thresholds()
    if "CCF_STRETCH_RESIDUAL_GATE" not in rows:
        raise SystemExit("CCF_STRETCH_RESIDUAL_GATE missing from _gates.py")
    for name, value in rows.items():
        print(f"{name}={value!r}")


if __name__ == "__main__":
    main()
