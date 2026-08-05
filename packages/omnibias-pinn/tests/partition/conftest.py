# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Torch-only fixtures for the omnibias.pinn.partition bridge tests (single-threaded)."""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

try:  # tiny float64 models train faster single-threaded
    import torch

    torch.set_num_threads(1)
except ModuleNotFoundError:  # pragma: no cover
    pass
