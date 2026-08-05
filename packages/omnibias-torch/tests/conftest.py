# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Shared fixtures for the omnibias-torch test suite.

``torch.set_default_dtype`` mutates process-wide state. Several modules in this
suite flip the default to ``float64`` to exercise the closed-form derivative
oracles in double precision. If that state leaks across tests (e.g. via a
module-level set or a future reordering plugin), dtype-sensitive regressions
break: ``test_audit_regressions`` asserts the float32 passthrough, the
``test_fastpath_stability`` cases rely on float32 catastrophic cancellation,
and the float32 :class:`JointOperatorRegressor` path desyncs from its inputs.

The autouse fixture below resets to the process-original default (captured at
conftest import time, before any test module's import-time set could leak in)
and restores it afterwards, so every test is hermetic regardless of collection
order or what any other test sets.
"""

from __future__ import annotations

import pytest
import torch

# Captured before any test module is imported, so this is the genuine process
# default (float32) even if a module sets float64 at import time.
_ORIGINAL_DEFAULT_DTYPE = torch.get_default_dtype()


@pytest.fixture(autouse=True)
def _restore_default_dtype() -> object:
    if torch.get_default_dtype() is not _ORIGINAL_DEFAULT_DTYPE:
        torch.set_default_dtype(_ORIGINAL_DEFAULT_DTYPE)
    prev = torch.get_default_dtype()
    try:
        yield
    finally:
        if torch.get_default_dtype() is not prev:
            torch.set_default_dtype(prev)
