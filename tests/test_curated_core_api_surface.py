# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Frozen public-API contract for the curated-core release set.

The eight curated-core packages (``core``, ``torch``, ``jax``, ``ferminet``,
``keras``, ``fields``, ``pinn``, ``geometry``) are published first under an
API-stability contract. This test pins each package's top-level ``__all__`` to a
committed baseline so any *unintended* change to the public surface fails CI.

Changing the surface is allowed -- it is a deliberate, reviewed act:

1. make the API change,
2. regenerate the baseline::

       KERAS_BACKEND=torch JAX_PLATFORMS=cpu python tests/_regen_curated_api.py

3. commit the updated ``tests/data/curated_core_public_api.json`` alongside a
   CHANGELOG entry (a removed/renamed symbol is a breaking change).

Packages that are not importable in the current environment (missing backend)
are skipped, so the test contributes partial coverage in backend-limited jobs
and full coverage where all eight are installed (the ``curated_api`` CI job).
"""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path

import pytest

_BASELINE_PATH = Path(__file__).parent / "data" / "curated_core_public_api.json"

# Keras needs a backend selected before import; default to torch for the check.
os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("JAX_PLATFORMS", "cpu")


def _load_baseline() -> dict[str, list[str]]:
    with _BASELINE_PATH.open(encoding="utf-8") as handle:
        data: dict[str, list[str]] = json.load(handle)
    return data


_BASELINE = _load_baseline()


@pytest.mark.parametrize("module_name", sorted(_BASELINE))
def test_public_surface_is_frozen(module_name: str) -> None:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - backend not installed in this job
        pytest.skip(f"{module_name} not importable here ({type(exc).__name__}); skipping")

    actual = sorted(getattr(module, "__all__", []) or [])
    expected = _BASELINE[module_name]

    added = sorted(set(actual) - set(expected))
    removed = sorted(set(expected) - set(actual))

    assert not (added or removed), (
        f"Public API surface of {module_name} changed.\n"
        f"  added:   {added}\n"
        f"  removed: {removed}\n"
        "If this change is intentional, regenerate the baseline with\n"
        "  KERAS_BACKEND=torch JAX_PLATFORMS=cpu python tests/_regen_curated_api.py\n"
        "and record it in the CHANGELOG (removed/renamed symbols are breaking)."
    )
