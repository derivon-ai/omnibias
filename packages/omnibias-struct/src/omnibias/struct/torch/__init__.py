# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Torch backend for omnibias-struct: soft-DP layers + closed-form lse/softmax jets."""

from __future__ import annotations

from omnibias.struct.torch._logsumexp import (
    logsumexp_beta,
    logsumexp_beta_hessian,
    logsumexp_beta_jacobian,
    pairwise_lse,
    pairwise_lse_jet,
    softmax_beta,
)
from omnibias.struct.torch.align import (
    soft_align,
    soft_align_batched,
    soft_align_marginals,
    soft_gotoh,
    soft_gotoh_marginals,
    soft_local_align,
    soft_local_align_marginals,
)
from omnibias.struct.torch.attention import structured_attention, structured_attention_batched
from omnibias.struct.torch.curvature import (
    chain_directional_curvature,
    chain_hessian,
    chain_sharpness,
    chain_value_and_hvp,
)
from omnibias.struct.torch.distributions import (
    gumbel_relaxed_sample,
    path_entropy,
    sample_paths,
    topk_free_energy,
    topk_paths,
)
from omnibias.struct.torch.dtw import (
    soft_dtw,
    soft_dtw_batched,
    soft_dtw_marginals,
)
from omnibias.struct.torch.eisner import eisner_marginals, soft_eisner, soft_eisner_batched
from omnibias.struct.torch.jets import (
    chain_lse_jet,
    cky_lse_jet,
    dag_lse_jet,
    dp_value_jet,
    eisner_lse_jet,
    hypergraph_lse_jet,
    lse2_jet,
)
from omnibias.struct.torch.monotonic import (
    soft_mas,
    soft_mas_batched,
    soft_mas_marginals,
)
from omnibias.struct.torch.mtt import matrix_tree_marginals, soft_matrix_tree
from omnibias.struct.torch.parse import (
    inside_outside,
    soft_inside,
    soft_inside_batched,
    span_marginals,
)
from omnibias.struct.torch.plan import soft_value_iteration, soft_value_iteration_batched
from omnibias.struct.torch.select import (
    certified_argmax,
    gibbs_covariance,
    gibbs_cumulants_directional,
    gibbs_mean,
    soft_argmax,
    soft_max_value,
)
from omnibias.struct.torch.semiring import semiring_marginals, semiring_value
from omnibias.struct.torch.soft_dp import (
    soft_ctc,
    soft_ctc_batched,
    soft_ctc_marginals,
    soft_shortest_path,
    soft_shortest_path_batched,
    soft_shortest_path_marginals,
    soft_viterbi,
    soft_viterbi_batched,
    soft_viterbi_marginals,
    soft_viterbi_transition_marginals,
)

__all__ = [
    "certified_argmax",
    "chain_directional_curvature",
    "chain_hessian",
    "chain_lse_jet",
    "chain_sharpness",
    "chain_value_and_hvp",
    "cky_lse_jet",
    "dag_lse_jet",
    "dp_value_jet",
    "eisner_lse_jet",
    "eisner_marginals",
    "gibbs_covariance",
    "gibbs_cumulants_directional",
    "gibbs_mean",
    "gumbel_relaxed_sample",
    "hypergraph_lse_jet",
    "inside_outside",
    "logsumexp_beta",
    "logsumexp_beta_hessian",
    "logsumexp_beta_jacobian",
    "lse2_jet",
    "matrix_tree_marginals",
    "pairwise_lse",
    "pairwise_lse_jet",
    "path_entropy",
    "sample_paths",
    "semiring_marginals",
    "semiring_value",
    "soft_align",
    "soft_align_batched",
    "soft_align_marginals",
    "soft_argmax",
    "soft_ctc",
    "soft_ctc_batched",
    "soft_ctc_marginals",
    "soft_dtw",
    "soft_dtw_batched",
    "soft_dtw_marginals",
    "soft_eisner",
    "soft_eisner_batched",
    "soft_gotoh",
    "soft_gotoh_marginals",
    "soft_inside",
    "soft_inside_batched",
    "soft_local_align",
    "soft_local_align_marginals",
    "soft_mas",
    "soft_mas_batched",
    "soft_mas_marginals",
    "soft_matrix_tree",
    "soft_max_value",
    "soft_shortest_path",
    "soft_shortest_path_batched",
    "soft_shortest_path_marginals",
    "soft_value_iteration",
    "soft_value_iteration_batched",
    "soft_viterbi",
    "soft_viterbi_batched",
    "soft_viterbi_marginals",
    "soft_viterbi_transition_marginals",
    "softmax_beta",
    "span_marginals",
    "structured_attention",
    "structured_attention_batched",
    "topk_free_energy",
    "topk_paths",
]
