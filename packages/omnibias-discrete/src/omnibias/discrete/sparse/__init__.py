# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""omnibias.discrete.sparse: certified sparse recovery via the ``l_p -> l_0`` collapse.

The substrate's second in-tree consumer (after :mod:`omnibias.discrete.maxsat`). Best-subset
selection is NP-hard, so the deliverable is a *certified optimality gap*, never exactness.
Three layers, explicit about which object each certificate seals:

* **Fork A** -- :class:`SupportSelectionProblem`, the pseudo-Boolean QUBO surrogate
  ``E(z) = 1/2 ||A z - b||^2 + lambda 1^T z``, certified directly by
  :func:`omnibias.discrete.certify_gap` (Lasserre / SOS).
* **Fork B** -- :class:`BestSubsetProblem`, the continuous-coefficient objective
  ``min_w ||A_S w - b||^2 + lambda |z|``, certified by :func:`certify_best_subset_gap`
  (a sound convex box-QP bound sealed by ``omnibias-convex``, back-stopped by the
  full-OLS-residual floor).
* **Fork C** -- :func:`certified_sparse_fit`, a hybrid that seals the pseudo-Boolean
  **surrogate** (Fork A) and ships an OLS refit on the decoded support for the continuous
  coefficients, returning a :class:`SparseFitResult`.

The new axis is the ``l_p -> l_0`` **penalty-exponent collapse**: a concave
``sum_i x_i^p`` whose ``p -> 0`` reweighting drives small entries to zero, riding on the
substrate's ``beta -> inf`` sigmoid **temperature collapse** in the relaxation twins
(:mod:`omnibias.discrete.sparse.jax` / :mod:`omnibias.discrete.sparse.torch`). Both are the
feasibility sense of "collapse", distinct from the **founding bias collapse** (the
multi-bias ``delta -> 0`` limit of an ``OMBU`` to the closed-form derivative
``sigma^(K-1)``; see ``docs/theory.md``).
"""

from __future__ import annotations

from omnibias.discrete.sparse.certify import (
    SparseFitResult,
    certified_sparse_fit,
    certify_best_subset_gap,
)
from omnibias.discrete.sparse.frontends import cardinality_constrained, sparse_least_squares
from omnibias.discrete.sparse.problem import BestSubsetProblem, SupportSelectionProblem

__all__ = [
    "BestSubsetProblem",
    "SparseFitResult",
    "SupportSelectionProblem",
    "cardinality_constrained",
    "certified_sparse_fit",
    "certify_best_subset_gap",
    "sparse_least_squares",
]
