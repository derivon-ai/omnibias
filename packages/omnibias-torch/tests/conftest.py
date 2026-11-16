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

The autouse fixture sets float32 explicitly for each test and restores its
incoming state afterwards. Tests requiring float64 use their own fixture.
Collection order cannot determine the precision contract.
"""

from __future__ import annotations

import os

import pytest
import torch

# Triton decides at decoration time whether a kernel runs on a GPU or in its interpreter, so the choice has to be made
# before any module defining a kernel is imported: without a GPU the fused-kernel tests use the interpreter.
if not torch.cuda.is_available():
    os.environ.setdefault("TRITON_INTERPRET", "1")


@pytest.fixture(autouse=True)
def _reset_dynamo() -> object:
    """Every test compiles its own models; Dynamo's per-code-object recompile limit is a per-process counter."""
    yield
    torch._dynamo.reset()


@pytest.fixture(autouse=True)
def _restore_default_dtype() -> object:
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        yield
    finally:
        if torch.get_default_dtype() is not prev:
            torch.set_default_dtype(prev)
