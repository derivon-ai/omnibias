# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Shared fixtures for the omnibias-tab test suite.

The soft-tree forward parity and the verified enclosures run in float64, so we set the
JAX x64 environment flag *before* any jax import (mirrors the omnibias-qubo /
omnibias-discrete setup). We deliberately do **not** import jax here -- the env flag is
enough to enable x64 when a test imports jax, and keeping conftest jax-free avoids paying
the (slow) jax import in the torch-only test files."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "1")

try:  # tiny float64 models train faster single-threaded (avoid op-dispatch oversubscription)
    import torch

    torch.set_num_threads(1)
except ModuleNotFoundError:  # pragma: no cover
    pass
