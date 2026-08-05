#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Import every shipped ``omnibias`` top-level package; fail on any error.

Used by the ``wheel_import_smoke`` CI job after installing the built wheels into
a clean, backend-enabled virtualenv. The package list is derived from the
``packages/omnibias-*`` directories, so it stays correct as packages are added
or removed (every package's import module is ``omnibias.<name>``).
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

_ROOT = Path(__file__).resolve().parents[1]
_PREFIX = "omnibias-"


def _package_names() -> list[str]:
    return sorted(
        path.name[len(_PREFIX) :]
        for path in (_ROOT / "packages").glob(f"{_PREFIX}*")
        if path.is_dir()
    )


def main() -> int:
    names = _package_names()
    failed: list[tuple[str, str]] = []
    for name in names:
        module = f"omnibias.{name}"
        try:
            importlib.import_module(module)
        except Exception as exc:  # noqa: BLE001 - smoke test reports every failure
            failed.append((module, f"{type(exc).__name__}: {exc}"))
            print(f"FAIL {module} - {type(exc).__name__}: {exc}")
        else:
            print(f"ok   {module}")

    if failed:
        print(f"\n{len(failed)} import(s) failed:")
        for module, err in failed:
            print(f"  {module}: {err}")
        return 1

    print(f"\nall {len(names)} shipped top-level packages imported cleanly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
