# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""omnibias-partition: a light, certified soft partition-of-unity primitive.

A hard partition of ``R^d`` into regions is a set of indicator functions ``1[region l]``.
omnibias makes it a **soft partition** built from oblique split gates
``g(x) = sigmoid(beta * (w.x - t))``: ``depth`` gates route an input into ``2**depth``
regions with weights ``w_l(x)`` that are non-negative, sum to one (a genuine partition of
unity), and **harden** to a crisp ``{0, 1}`` partition as ``beta -> inf``.

The keystone shared by four downstream bridges (a discontinuity-capturing PINN, a
region-wise Riemannian atlas, per-region symbolic discovery, and a certified decision
layer). It ships:

1. :func:`partition_weights` -- the numpy reference plus bit-identical
   :mod:`omnibias.partition.torch` / :mod:`omnibias.partition.jax` twins (parity ``~1e-9``);
2. :func:`hard_assignment` / :func:`hardened_rules` -- the crisp region index and the
   exported human-readable ``if w.x > t`` boundaries;
3. :func:`certify_partition_gap` -- a **sound** soft->hard membership-gap certificate
   (outward-rounded :class:`~omnibias.core.verified.Interval` + closed-form
   ``log(n_regions)/beta``), a well-posed **yes-if** object;
4. :class:`RegionModels` -- a per-region model registry whose single
   ``combine(X, beta) = sum_l w_l * out_l`` engine every bridge calls.

Terminology: the gate's ``beta -> inf`` hardening is **temperature collapse** -- the
feasibility sense (a soft indicator becoming a 0/1 step), distinct from
the **founding bias collapse** (the multi-bias ``delta -> 0`` limit of an ``OMBU`` to the
closed-form derivative ``sigma^(K-1)``; see ``docs/theory.md``). The bridges differentiate
products of sigmoids by autodiff -- the closed-form derivative tower does not auto-extend to
products.

The torch / jax weight twins need the ``torch`` / ``jax`` extra (each degrades gracefully;
the numpy core + certificate need neither).

.. important::

    **Bit-parity with the PyTorch twin requires 64-bit JAX** --
    ``jax.config.update("jax_enable_x64", True)`` before the first JAX array is
    created (or ``JAX_ENABLE_X64=1``). JAX otherwise truncates to ``float32``
    while PyTorch uses ``float64``, so the twins stay internally consistent but
    agree only to ``float32`` tolerance. Where a value feeds a threshold, a
    rounding step or an ``argmax``, that is enough to change the decision rather
    than just the last digits. See :mod:`omnibias.jax.precision`.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError as _PkgNotFound
from importlib.metadata import version as _pkg_version

from omnibias.partition._core.config import PartitionConfig
from omnibias.partition._core.params import (
    PartitionParams,
    init_params,
    region_code_matrix,
)
from omnibias.partition._core.weights import (
    gate_activations,
    hard_assignment,
    hard_weights,
    hardened_rules,
    partition_weights,
    region_rule,
)
from omnibias.partition.arrangement import (
    Arrangement,
    CellGapCertificate,
    certify_cell_gap,
    max_cells,
    soft_membership,
)
from omnibias.partition.certify import PartitionGapCertificate, certify_partition_gap
from omnibias.partition.registry import RegionModels, combine_outputs

try:
    __version__ = _pkg_version("omnibias-partition")
except _PkgNotFound:  # pragma: no cover - bare source checkout
    __version__ = "0.0.0+unknown"

# Founding-idea lineage (see docs/theory.md "Two senses of collapse").
__lineage__ = "temperature collapse"

__all__ = [
    "Arrangement",
    "CellGapCertificate",
    "PartitionConfig",
    "PartitionGapCertificate",
    "PartitionParams",
    "RegionModels",
    "__lineage__",
    "__version__",
    "certify_cell_gap",
    "certify_partition_gap",
    "combine_outputs",
    "gate_activations",
    "hard_assignment",
    "hard_weights",
    "hardened_rules",
    "init_params",
    "max_cells",
    "partition_weights",
    "region_code_matrix",
    "region_rule",
    "soft_membership",
]
