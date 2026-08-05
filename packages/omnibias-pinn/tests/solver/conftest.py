# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Shared test fixtures: run backend tests in double precision for bit parity."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _default_dtype_f64():
    try:
        import torch
    except ImportError:
        yield
        return
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)
