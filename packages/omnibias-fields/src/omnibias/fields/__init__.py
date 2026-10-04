# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Backend-neutral field schemas, derivative caching, and reusable calculus.

FieldState exposes gradient, divergence, curl, Laplacian, Hessian, integration,
and norms through PyTorch/JAX operators. Backends are imported lazily. Use
matching dtypes (JAX_ENABLE_X64=1 for float64) for cross-backend comparisons.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError as _PkgNotFound
from importlib.metadata import version as _pkg_version

from omnibias.fields._core import (
    DISPATCH_ATTR,
    DOMAINS,
    READOUT_INDEPENDENT_ATTR,
    ComponentSpec,
    ComponentView,
    CoordinateSpec,
    FieldBase,
    FieldState,
    OperatorInfo,
    SigmaCache,
    VectorView,
    did_you_mean,
    get_operator,
    list_operators,
    operator_names,
    ops_registry,
)
from omnibias.fields.locus import (
    AffineSet,
    EqualitySystem,
    NewtonResult,
    UnitTerm,
    affine_locus,
    certify_locus_point,
)
from omnibias.fields.weak import (
    TestFunctionSpace,
    WeakForm,
    boundary_bound,
    exact_moment,
)

try:
    __version__ = _pkg_version("omnibias-fields")
except _PkgNotFound:  # pragma: no cover - bare source checkout
    __version__ = "0.0.0+unknown"

# Limit family exposed as package metadata.
__lineage__ = "bias collapse"

__all__ = [
    "AffineSet",
    "ComponentSpec",
    "ComponentView",
    "CoordinateSpec",
    "DISPATCH_ATTR",
    "DOMAINS",
    "EqualitySystem",
    "FieldBase",
    "FieldState",
    "NewtonResult",
    "OperatorInfo",
    "READOUT_INDEPENDENT_ATTR",
    "SigmaCache",
    "TestFunctionSpace",
    "UnitTerm",
    "VectorView",
    "WeakForm",
    "__lineage__",
    "__version__",
    "affine_locus",
    "boundary_bound",
    "certify_locus_point",
    "did_you_mean",
    "exact_moment",
    "get_operator",
    "list_operators",
    "operator_names",
    "ops_registry",
]
