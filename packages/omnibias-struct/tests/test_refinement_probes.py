# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""CPU-tiny deterministic subset of the data-driven probe harness (:mod:`_harness`).

These lock the invariants the cluster sweep measures at scale: hard DP == brute force, the
``lse_beta`` gap sandwich is sound, torch <-> jax parity is bit-identical, the log-space
path count is overflow-free, and the soft value stays finite as ``beta`` grows in float64.
"""

from __future__ import annotations

import math

import pytest
from _harness import (
    PROBLEMS,
    probe_beta_stability,
    probe_gap_tightness,
    probe_marginals_vs_autodiff,
    probe_oracle_agreement,
    probe_parity,
    probe_path_count,
    reference_log_count,
)
from _struct_helpers import random_chain

BETAS = (1.0, 16.0, 256.0)


@pytest.mark.parametrize("problem", PROBLEMS)
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_oracle_agreement(problem: str, seed: int) -> None:
    rec = probe_oracle_agreement(problem, seed)
    assert rec.ok, rec


@pytest.mark.parametrize("problem", PROBLEMS)
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_logspace_path_count_matches_exact(problem: str, seed: int) -> None:
    rec = probe_path_count(problem, seed)
    assert rec.ok, rec


@pytest.mark.parametrize("problem", PROBLEMS)
@pytest.mark.parametrize("beta", BETAS)
def test_gap_sandwich_is_sound_and_never_optimistic(problem: str, beta: float) -> None:
    pytest.importorskip("torch")
    rec = probe_gap_tightness(problem, 0, beta)
    assert rec.ok, rec
    assert rec.metrics["realized_gap"] <= rec.metrics["bound"] + 1e-9


@pytest.mark.parametrize("problem", PROBLEMS)
@pytest.mark.parametrize("beta", BETAS)
def test_torch_jax_parity_is_bit_identical(problem: str, beta: float) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("jax")
    rec = probe_parity(problem, 0, beta)
    assert rec.metrics["value_abs_diff"] < 1e-9, rec


@pytest.mark.parametrize("problem", PROBLEMS)
@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_closed_form_marginals_match_autodiff(problem: str, backend: str) -> None:
    pytest.importorskip(backend)
    rec = probe_marginals_vs_autodiff(problem, 0, 6.0, backend)
    assert rec.ok, rec


@pytest.mark.parametrize("problem", PROBLEMS)
@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_soft_value_finite_at_large_beta_float64(problem: str, backend: str) -> None:
    pytest.importorskip(backend)
    rec = probe_beta_stability(problem, 0, 1.0e6, backend)
    assert math.isfinite(rec.metrics["value"]), rec


def test_logspace_count_is_overflow_free_for_astronomical_instances() -> None:
    # 10 ** 400 paths: the exact integer is a 401-digit bignum; the log-space count
    # stays a finite float without ever materialising it.
    big = random_chain(0, n_steps=400, n_states=10)
    log_n = reference_log_count(big)
    assert math.isfinite(log_n)
    assert abs(log_n - 400 * math.log(10)) < 1e-9
