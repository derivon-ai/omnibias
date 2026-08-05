# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Public verified (interval) soft-DP surface -- sound, outward-rounded enclosures.

Re-exports the rigorous register of the soft DP from :mod:`omnibias.struct._core.verified`:
``lse_beta_iv`` / ``pairwise_lse_iv`` reductions and the ``chain_value_iv`` /
``dag_value_iv`` / ``dtw_value_iv`` / ``align_value_iv`` / ``ctc_value_iv`` recurrences that
enclose ``V_beta`` for every point in an input box, the certified ``chain_marginals_iv``
forward-backward enclosure, the ``matrix_tree_partition_iv`` interval determinant, plus the
:func:`box` constructor. These are the ``beta -> inf`` relaxation computed with directed
rounding; scope is always the given ``local_box``.
"""

from __future__ import annotations

from omnibias.struct._core.verified import (
    align_value_iv,
    box,
    chain_marginals_iv,
    chain_value_iv,
    ctc_value_iv,
    dag_value_iv,
    dtw_value_iv,
    lse_beta_iv,
    matrix_tree_partition_iv,
    pairwise_lse_iv,
)

__all__ = [
    "align_value_iv",
    "box",
    "chain_marginals_iv",
    "chain_value_iv",
    "ctc_value_iv",
    "dag_value_iv",
    "dtw_value_iv",
    "lse_beta_iv",
    "matrix_tree_partition_iv",
    "pairwise_lse_iv",
]
