#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Local pre-commit gating script for omnibias-pinn.

Runs ``scripts/run_tests.sh fast`` on every commit; promotes to
``scripts/run_tests.sh cross`` if the staged diff touches solver
surface (fields, ops, cage, equations, losses, diagnostics, _core
schemas).

Wire into ``.git/hooks/pre-commit`` like this::

    ln -sf ../../packages/omnibias-pinn/scripts/pre_commit.py \\
        .git/hooks/pre-commit

Local-only; never pushed to a remote runner.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PKG_ROOT.parent.parent

SOLVER_PREFIXES = (
    "packages/omnibias-pinn/src/omnibias/pinn/_core/",
    "packages/omnibias-pinn/src/omnibias/pinn/torch/fields/",
    "packages/omnibias-pinn/src/omnibias/pinn/torch/ops/",
    "packages/omnibias-pinn/src/omnibias/pinn/torch/cage/",
    "packages/omnibias-pinn/src/omnibias/pinn/torch/equations/",
    "packages/omnibias-pinn/src/omnibias/pinn/torch/losses/",
    "packages/omnibias-pinn/src/omnibias/pinn/torch/diagnostics/",
    "packages/omnibias-pinn/src/omnibias/pinn/jax/fields/",
    "packages/omnibias-pinn/src/omnibias/pinn/jax/ops/",
    "packages/omnibias-pinn/src/omnibias/pinn/jax/cage/",
    "packages/omnibias-pinn/src/omnibias/pinn/jax/equations/",
    "packages/omnibias-pinn/src/omnibias/pinn/jax/losses/",
    "packages/omnibias-pinn/src/omnibias/pinn/jax/diagnostics/",
)


def _staged_files() -> list[str]:
    res = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    )
    return [ln.strip() for ln in res.stdout.splitlines() if ln.strip()]


def _touches_solver(staged: list[str]) -> bool:
    return any(any(p.startswith(pref) for pref in SOLVER_PREFIXES) for p in staged)


def main() -> int:
    runner = PKG_ROOT / "scripts" / "run_tests.sh"
    if not runner.exists():
        print(f"[pre-commit] runner missing at {runner}", file=sys.stderr)
        return 0

    if os.environ.get("OMNIBIAS_PINN_PRECOMMIT_SKIP", "0") == "1":
        print("[pre-commit] OMNIBIAS_PINN_PRECOMMIT_SKIP=1 -> skipping")
        return 0

    try:
        staged = _staged_files()
    except subprocess.CalledProcessError as exc:
        print(f"[pre-commit] git diff failed: {exc}", file=sys.stderr)
        return 0

    pinn_files = [s for s in staged if s.startswith("packages/omnibias-pinn/")]
    if not pinn_files:
        print("[pre-commit] no omnibias-pinn changes; skipping local CI")
        return 0

    tier = "cross" if _touches_solver(pinn_files) else "fast"
    print(f"[pre-commit] tier={tier}  files={len(pinn_files)}")
    res = subprocess.run(
        [str(runner), tier],
        cwd=REPO_ROOT,
    )
    if res.returncode != 0:
        print(
            f"[pre-commit] local CI ({tier}) failed (exit {res.returncode}). "
            f"Set OMNIBIAS_PINN_PRECOMMIT_SKIP=1 to bypass.",
            file=sys.stderr,
        )
    return res.returncode


if __name__ == "__main__":
    sys.exit(main())
