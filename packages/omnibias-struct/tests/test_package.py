# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Smoke tests for the omnibias-struct package scaffold and its pure-numpy core."""

from __future__ import annotations

import numpy as np
import omnibias.struct as st


def test_version() -> None:
    assert st.__version__ == "0.1.0a1"


def test_public_surface() -> None:
    for name in (
        "ChainTrellis",
        "DAG",
        "CTCLattice",
        "count_paths",
        "viterbi",
        "shortest_path",
        "ctc_best",
        "certify_soft_dp",
        "DPGapCertificate",
        "logsumexp_gap_bound",
    ):
        assert hasattr(st, name), name


def test_core_runs_without_a_backend() -> None:
    # The backend-agnostic core is pure numpy: build + solve + certify with no torch/jax.
    rng = np.random.default_rng(0)
    trellis = st.ChainTrellis(rng.standard_normal((4, 3)), rng.standard_normal((3, 3)))
    hard, path = st.viterbi(trellis)
    brute, brute_path = st.brute_force_viterbi(trellis)
    assert abs(hard - brute) < 1e-9
    assert len(path) == trellis.n_steps
    cert = st.certify_soft_dp(hard, hard + 0.1, st.count_paths(trellis), beta=4.0)
    assert cert.method == "logsumexp_gap"
