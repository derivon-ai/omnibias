# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Shared fixtures for the omnibias-partition test suite.

The partition weights parity and the verified enclosures run in float64, so we set the JAX
x64 environment flag *before* any jax import (mirrors omnibias-tab / omnibias-qubo). We do
**not** import jax here -- the env flag is enough, and a jax-free conftest avoids paying the
slow jax import in the numpy / torch-only test files."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "1")

try:  # tiny float64 tensors run faster single-threaded (avoid op-dispatch oversubscription)
    import torch

    torch.set_num_threads(1)
except ModuleNotFoundError:  # pragma: no cover
    pass
