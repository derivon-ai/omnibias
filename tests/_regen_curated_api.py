# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Regenerate the frozen curated-core public-API baseline.

Run with every curated-core package installed (and a Keras backend selected)::

    KERAS_BACKEND=torch JAX_PLATFORMS=cpu python tests/_regen_curated_api.py

This rewrites ``tests/data/curated_core_public_api.json``. Review the diff and
record any removed/renamed symbol as a breaking change in the CHANGELOG before
committing (see ``tests/test_curated_core_api_surface.py``).
"""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path

CURATED_CORE = (
    "core",
    "torch",
    "jax",
    "ferminet",
    "keras",
    "fields",
    "pinn",
    "geometry",
)

_OUTPUT = Path(__file__).parent / "data" / "curated_core_public_api.json"


def main() -> None:
    os.environ.setdefault("KERAS_BACKEND", "torch")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

    surface: dict[str, list[str]] = {}
    for name in CURATED_CORE:
        module = importlib.import_module(f"omnibias.{name}")
        surface[f"omnibias.{name}"] = sorted(getattr(module, "__all__", []) or [])

    _OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with _OUTPUT.open("w", encoding="utf-8") as handle:
        json.dump(surface, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"wrote {_OUTPUT}")
    for module_name, symbols in surface.items():
        print(f"  {module_name}: {len(symbols)} symbols")


if __name__ == "__main__":
    main()
