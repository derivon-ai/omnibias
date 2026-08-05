# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Package smoke: version, public surface, and the core-only import guard."""

from __future__ import annotations

import os
import subprocess
import sys

import omnibias.difference as D


def test_version() -> None:
    assert D.__version__ == "0.1.0a1"


def test_public_surface_present() -> None:
    for name in (
        "certified_derivative_enclosure",
        "finite_difference_estimate",
        "certified_fd_error",
        "stirling_second",
        "bernoulli_number",
        "euler_number",
        "eulerian_number",
        "forward_difference",
        "monomial_to_falling",
        "bell_number_asymptotic",
    ):
        assert name in D.__all__ and hasattr(D, name), name


def test_all_is_sorted() -> None:
    assert list(D.__all__) == sorted(D.__all__)


def test_core_import_needs_no_backend() -> None:
    """Importing ``omnibias.difference`` must not drag in torch or jax."""
    code = (
        "import sys, omnibias.difference as D; "
        "D.certified_derivative_enclosure('tanh', 0.0, 2); "
        "assert 'torch' not in sys.modules, 'torch imported'; "
        "assert 'jax' not in sys.modules, 'jax imported'; "
        "print('ok')"
    )
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env
    )
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout
