# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Weak-form VPINN test functions (theory 02-04, gated).

Import assembly kernels from :mod:`omnibias.fields.weak.torch` or
:mod:`omnibias.fields.weak.jax`. Exact integrals only for polynomial
coefficients on boxes; the boundary bound is on by default.
"""

from __future__ import annotations

from omnibias.fields.weak._core import (
    TestFunctionSpace,
    WeakForm,
    boundary_bound,
    eval_test,
    exact_moment,
    poly_eval,
)

__all__ = [
    "TestFunctionSpace",
    "WeakForm",
    "boundary_bound",
    "eval_test",
    "exact_moment",
    "poly_eval",
]
