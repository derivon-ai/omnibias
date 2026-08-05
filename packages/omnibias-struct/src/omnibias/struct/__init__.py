# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""omnibias-struct: certified differentiable dynamic programming (Viterbi / shortest-path / CTC).

Exact hard DP is not differentiable -- its ``argmax`` gradient is a.e. zero -- so the
sound differentiable object is a **relaxation plus a certified gap**, never an exactness
claim. This package keeps two limits rigorously apart (conflating them is the contagious
mistake):

* **``beta -> inf`` (the temperature / relaxation axis).** The hard ``max`` combine is
  replaced by ``lse_beta(a) = beta^-1 log sum_i exp(beta a_i)``; since ``lse_beta >= max``
  and ``lse_beta -> max`` as ``beta -> inf``, the soft DP anneals to exact hard DP. This
  is the *feasibility / temperature* sense of "collapse", the same axis as
  ``omnibias-discrete`` / ``omnibias-qubo`` / ``omnibias-routing`` -- **not** bias
  collapse.
* **``delta -> 0`` (the founding bias collapse tower): the exact differentiation
  engine.** Pairwise ``lse_beta(a, b) = a + beta^-1 softplus(beta (b - a))``, and
  ``softplus`` is Riccati with the closed-form tower ``softplus^(n) = sigma^(n-1)`` in
  :mod:`omnibias.core`. Its beta-tempered tower propagated through ``compose_jet``
  (:mod:`omnibias.torch.jet` / :mod:`omnibias.jax.jet`) yields the closed-form
  log-sum-exp / softmax jets; the first-order sensitivity is the softmax marginal, and
  the soft-DP gradient is the forward-backward path marginal assembled from it.

This module exposes the backend-agnostic surface (problem containers, exact hard DP,
brute-force oracles, and the closed-form gap certificate). The differentiable soft-DP
layers live under :mod:`omnibias.struct.torch` and :mod:`omnibias.struct.jax`
(bit-identical twins).

Relation to ``omnibias-graph`` (decided: intentionally divergent). The ``beta``-scaled
``logsumexp_beta`` soft-DP here is a *different* operator from the ``tau``-scaled
SoftSort / Sinkhorn / soft-top-``k`` relaxations in :mod:`omnibias.graph`, not a shared
substrate: those relax *uncoupled* sorting / assignment / selection on a flat score
vector, whereas these relax the *coupled* marginals of an exponentially large
structured state space (a trellis / grammar / lattice) and carry a ``log(N)/beta`` gap
certificate. The two surfaces meet only at the trivial single-choice ``softmax`` case
and are deliberately kept apart.

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

from omnibias.struct._core import (
    DAG,
    AcyclicMDP,
    AlignmentLattice,
    BinaryGrammar,
    ChainTrellis,
    CountingSemiring,
    CTCLattice,
    DPGapCertificate,
    DTWLattice,
    HyperEdge,
    Hypergraph,
    LogSemiring,
    LSEBeta,
    MaxPlusSemiring,
    SelectionCertificate,
    Semiring,
    argmax_stability_margin,
    best_derivation,
    best_parse_tree,
    best_projective_tree,
    beta_for_confidence,
    brute_force_align,
    brute_force_arborescence,
    brute_force_cky,
    brute_force_ctc,
    brute_force_dtw,
    brute_force_eisner,
    brute_force_entropy,
    brute_force_gotoh,
    brute_force_kbest,
    brute_force_local_align,
    brute_force_mas,
    brute_force_optimal_return,
    brute_force_partition,
    brute_force_projective,
    brute_force_shortest_path,
    brute_force_soft_align,
    brute_force_soft_dtw,
    brute_force_soft_gotoh,
    brute_force_soft_local_align,
    brute_force_soft_mas,
    brute_force_value,
    brute_force_viterbi,
    build_chart,
    build_gotoh_dag,
    build_local_dag,
    certify_argmax,
    certify_soft_dp,
    count_alignments,
    count_arborescences,
    count_derivations,
    count_parse_trees,
    count_paths,
    count_projective_trees,
    ctc_best,
    ctc_best_alignment,
    derivation_weight,
    eisner_hypergraph,
    enumerate_derivations,
    from_dag,
    hard_align,
    hard_cky,
    hard_dtw,
    hard_eisner,
    hard_gotoh,
    hard_local_align,
    hard_mas,
    hard_matrix_tree,
    hard_value,
    hard_value_iteration,
    iter_arborescences,
    iter_projective_trees,
    kbest_derivations,
    log_num_paths,
    logsumexp_gap_bound,
    mass_concentration_bound,
    matrix_tree_marginals,
    matrix_tree_partition,
    max_arborescence,
    sample_derivations,
    seal_selection_certificate,
    semiring_value,
    shortest_path,
    soft_cky,
    soft_eisner,
    soft_value,
    stepwise_gap_bound,
    viterbi,
)

try:
    __version__ = _pkg_version("omnibias-struct")
except _PkgNotFound:  # pragma: no cover - bare source checkout
    __version__ = "0.0.0+unknown"

# Founding-idea lineage (see docs/theory.md "Two senses of collapse").
__lineage__ = "both"

__all__ = [
    "AcyclicMDP",
    "AlignmentLattice",
    "BinaryGrammar",
    "CTCLattice",
    "ChainTrellis",
    "CountingSemiring",
    "DAG",
    "DPGapCertificate",
    "DTWLattice",
    "HyperEdge",
    "Hypergraph",
    "LSEBeta",
    "LogSemiring",
    "MaxPlusSemiring",
    "SelectionCertificate",
    "Semiring",
    "__lineage__",
    "__version__",
    "argmax_stability_margin",
    "best_derivation",
    "best_parse_tree",
    "best_projective_tree",
    "beta_for_confidence",
    "brute_force_align",
    "brute_force_arborescence",
    "brute_force_cky",
    "brute_force_ctc",
    "brute_force_dtw",
    "brute_force_eisner",
    "brute_force_entropy",
    "brute_force_gotoh",
    "brute_force_kbest",
    "brute_force_local_align",
    "brute_force_mas",
    "brute_force_optimal_return",
    "brute_force_partition",
    "brute_force_projective",
    "brute_force_shortest_path",
    "brute_force_soft_align",
    "brute_force_soft_dtw",
    "brute_force_soft_gotoh",
    "brute_force_soft_local_align",
    "brute_force_soft_mas",
    "brute_force_value",
    "brute_force_viterbi",
    "build_chart",
    "build_gotoh_dag",
    "build_local_dag",
    "certify_argmax",
    "certify_soft_dp",
    "count_alignments",
    "count_arborescences",
    "count_derivations",
    "count_parse_trees",
    "count_paths",
    "count_projective_trees",
    "ctc_best",
    "ctc_best_alignment",
    "derivation_weight",
    "eisner_hypergraph",
    "enumerate_derivations",
    "from_dag",
    "hard_align",
    "hard_cky",
    "hard_dtw",
    "hard_eisner",
    "hard_gotoh",
    "hard_local_align",
    "hard_mas",
    "hard_matrix_tree",
    "hard_value",
    "hard_value_iteration",
    "iter_arborescences",
    "iter_projective_trees",
    "kbest_derivations",
    "log_num_paths",
    "logsumexp_gap_bound",
    "mass_concentration_bound",
    "matrix_tree_marginals",
    "matrix_tree_partition",
    "max_arborescence",
    "sample_derivations",
    "seal_selection_certificate",
    "semiring_value",
    "shortest_path",
    "soft_cky",
    "soft_eisner",
    "soft_value",
    "stepwise_gap_bound",
    "viterbi",
]
