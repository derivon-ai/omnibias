# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Shared helpers for the public ``benchmarks/`` suite.

Every script writes a JSON artifact under ``docs/benchmarks/`` with a common
provenance header so README numbers stay traceable to a committed file anyone
can regenerate.
"""

from __future__ import annotations

import json
import os
import platform
import resource
import statistics
import sys
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "docs" / "benchmarks"


def enable_x64() -> None:
    os.environ.setdefault("JAX_ENABLE_X64", "true")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")


def median_time_ms(
    fn: Callable[[], Any],
    *,
    warmup: int = 3,
    repeats: int = 7,
) -> float:
    """Return the median wall-clock of ``fn`` in milliseconds."""
    for _ in range(warmup):
        fn()
    samples: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1e3)
    return float(statistics.median(samples))


def rss_mb() -> float:
    """Current max-RSS in MiB (Linux / macOS; best-effort elsewhere)."""
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB; macOS reports bytes.
    if sys.platform == "darwin":
        return usage / (1024.0 * 1024.0)
    return usage / 1024.0


def provenance(*, schema: str, config: dict[str, Any]) -> dict[str, Any]:
    """Build the common header every artifact carries."""
    versions: dict[str, str] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for mod_name, attr in (
        ("numpy", "__version__"),
        ("jax", "__version__"),
        ("torch", "__version__"),
        ("folx", "__version__"),
    ):
        try:
            mod = __import__(mod_name)
            versions[mod_name] = str(getattr(mod, attr, "unknown"))
        except Exception:  # noqa: BLE001 -- optional deps
            versions[mod_name] = "absent"

    nproc = os.cpu_count() or 0
    return {
        "schema": schema,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "hardware_class": f"commodity x86-64 CPU ({nproc} logical cores; float64; JAX_PLATFORMS=cpu)",
        "versions": versions,
        "config": config,
        "rss_mb_at_start": round(rss_mb(), 2),
    }


def write_json(name: str, payload: dict[str, Any]) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return path
