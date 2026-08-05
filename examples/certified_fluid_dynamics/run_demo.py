# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Run the certified fluid-dynamics demo (CPU, generated data only).

Examples
--------
Run from the repository root::

    python -m examples.certified_fluid_dynamics.run_demo
    python -m examples.certified_fluid_dynamics.run_demo --n 128 --viscosity 0.05
    python -m examples.certified_fluid_dynamics.run_demo --scratch-dir "$OMNIBIAS_SCRATCH/fluid"

The ``--scratch-dir`` is purely a runtime cache location for generated arrays; it
is read from the flag or the ``OMNIBIAS_SCRATCH`` environment variable and is
never written by default.  No external data is downloaded.
"""

from __future__ import annotations

import argparse
import json
import os

from examples.certified_fluid_dynamics.benchmark import evaluate_benchmark


def _default_scratch_dir() -> str | None:
    base = os.environ.get("OMNIBIAS_SCRATCH")
    return os.path.join(base, "certified_fluid_dynamics") if base else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Certified fluid-dynamics demo")
    parser.add_argument("--n", type=int, default=64, help="grid resolution per axis")
    parser.add_argument("--viscosity", type=float, default=0.1, help="dynamic viscosity")
    parser.add_argument(
        "--scratch-dir",
        type=str,
        default=_default_scratch_dir(),
        help="optional runtime cache directory (defaults to $OMNIBIAS_SCRATCH/... if set)",
    )
    args = parser.parse_args()
    result = evaluate_benchmark(
        n=args.n, viscosity=args.viscosity, scratch_dir=args.scratch_dir
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
