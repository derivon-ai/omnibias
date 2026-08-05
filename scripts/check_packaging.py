#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Verify built distributions carry no packaging junk.

Makes the packaging-hygiene invariant enforceable: ``python -m build`` output
must be clean. Scans every ``dist/*.whl`` and ``dist/*.tar.gz`` and fails on

* build byproducts / caches anywhere (``__pycache__``, ``*.pyc`` / ``*.pyo``,
  ``.pytest_cache``, ``.mypy_cache``, ``.ruff_cache``, ``.DS_Store``);
* a stray top-level ``build/`` directory inside an sdist (a setuptools
  byproduct that must never ship);
* any wheel member outside the ``omnibias/`` package tree or the
  ``*.dist-info/`` metadata directory (e.g. a leaked ``.egg-info``, ``tests/``,
  or stray top-level module).

Run after the ``build_wheels`` CI job builds every distribution into ``dist/``.
An sdist legitimately ships auto-generated ``*.egg-info`` metadata and
``tests/`` -- those are standard setuptools output and are *not* flagged.
"""

from __future__ import annotations

import sys
import tarfile
import zipfile
from pathlib import Path

_CACHE_JUNK = (
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".DS_Store",
)


def _is_cache_junk(name: str) -> bool:
    if any(part in _CACHE_JUNK for part in name.split("/")):
        return True
    return name.endswith((".pyc", ".pyo"))


def _wheel_offenders(whl: Path) -> list[str]:
    """A wheel must contain only the ``omnibias/`` tree and ``*.dist-info/``."""
    bad: list[str] = []
    with zipfile.ZipFile(whl) as zf:
        for name in zf.namelist():
            if name.endswith("/"):
                continue
            top = name.split("/", 1)[0]
            if _is_cache_junk(name):
                bad.append(f"{whl.name}: cache junk -> {name}")
            elif top != "omnibias" and not top.endswith(".dist-info"):
                bad.append(f"{whl.name}: unexpected wheel member -> {name}")
    return bad


def _sdist_offenders(sd: Path) -> list[str]:
    """An sdist may ship src/tests/egg-info; caches and a stray build/ may not."""
    bad: list[str] = []
    with tarfile.open(sd) as tf:
        for name in tf.getnames():
            parts = name.split("/")
            if _is_cache_junk(name):
                bad.append(f"{sd.name}: cache junk -> {name}")
            # A setuptools byproduct sits at ``<root>/build/...`` (build is the
            # component right under the sdist root directory).
            elif len(parts) > 2 and parts[1] == "build":
                bad.append(f"{sd.name}: stray build dir -> {name}")
    return bad


def main() -> int:
    dist = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("dist")
    wheels = sorted(dist.glob("*.whl"))
    sdists = sorted(dist.glob("*.tar.gz"))
    if not wheels and not sdists:
        print(f"no distributions found in {dist}/ (nothing to check)")
        return 1

    offenders: list[str] = []
    for whl in wheels:
        offenders += _wheel_offenders(whl)
    for sd in sdists:
        offenders += _sdist_offenders(sd)

    if offenders:
        print("Packaging junk detected in built distributions:")
        for line in offenders:
            print(f"  {line}")
        return 1

    print(
        f"packaging clean: {len(wheels)} wheel(s) + {len(sdists)} sdist(s) "
        "carry no caches, stray build dirs, or unexpected wheel members"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
