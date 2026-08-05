# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""CPU-tiny deterministic subset of the new-family probes (:mod:`_harness`).

Locks the semiring-driver families' honesty axes at small scale: hard DP == brute force
(CKY / Eisner / matrix-tree / local + affine alignment), closed-form marginals == backend
autodiff, torch <-> jax parity, the ``lse_beta`` gap sandwich is sound, the entropy identity
matches enumeration (and the exact sampler), the sampler's empirical marginals converge, and
exact k-best equals the enumerate-and-sort oracle. The multi-seed / high-beta sweep runs on
the cluster via ``struct_refinement/sweep.py`` in the separate ``omnibias_experiments`` project.
"""

from __future__ import annotations

import pytest
from _harness import (
    probe_align_gap,
    probe_align_oracle,
    probe_align_parity,
    probe_eisner_gap,
    probe_eisner_marginals,
    probe_eisner_oracle,
    probe_eisner_parity,
    probe_entropy,
    probe_kbest,
    probe_mtt_gap,
    probe_mtt_marginals,
    probe_mtt_oracle,
    probe_mtt_parity,
    probe_parse_gap,
    probe_parse_marginals,
    probe_parse_oracle,
    probe_parse_parity,
    probe_sampling,
)

BETA = 6.0


# --- oracle agreement (beta -> inf limit is exact) -----------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_parse_hard_equals_brute_and_count(seed: int) -> None:
    assert probe_parse_oracle(seed).ok


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_eisner_hard_equals_brute_and_count(seed: int) -> None:
    assert probe_eisner_oracle(seed).ok


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_mtt_edmonds_equals_hard_and_brute(seed: int) -> None:
    assert probe_mtt_oracle(seed).ok


@pytest.mark.parametrize("kind", ["local", "affine"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_alignment_hard_equals_brute(kind: str, seed: int) -> None:
    assert probe_align_oracle(kind, seed).ok


# --- closed-form marginals == autodiff (delta -> 0 tower is exact) -------


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_parse_marginals_match_autodiff(backend: str) -> None:
    pytest.importorskip(backend)
    assert probe_parse_marginals(0, BETA, backend).ok


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_eisner_marginals_match_autodiff(backend: str) -> None:
    pytest.importorskip(backend)
    assert probe_eisner_marginals(0, BETA, backend).ok


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_mtt_marginals_match_autodiff(backend: str) -> None:
    pytest.importorskip(backend)
    assert probe_mtt_marginals(0, BETA, backend).ok


# --- certified gap sandwich is sound (temperature axis) ------------------


@pytest.mark.parametrize("probe", [probe_parse_gap, probe_eisner_gap, probe_mtt_gap])
def test_family_gap_sandwich_is_sound(probe) -> None:  # noqa: ANN001
    pytest.importorskip("torch")
    rec = probe(0, BETA, "torch")
    assert rec.ok
    assert rec.metrics["realized_gap"] <= rec.metrics["bound"] + 1e-9


@pytest.mark.parametrize("kind", ["local", "affine"])
def test_alignment_soft_ge_hard(kind: str) -> None:
    pytest.importorskip("torch")
    assert probe_align_gap(kind, 0, BETA, "torch").ok


# --- torch <-> jax parity ------------------------------------------------


@pytest.mark.parametrize(
    "probe", [probe_parse_parity, probe_eisner_parity, probe_mtt_parity]
)
def test_family_parity_bit_identical(probe) -> None:  # noqa: ANN001
    pytest.importorskip("torch")
    pytest.importorskip("jax")
    assert probe(0, BETA).metrics["value_abs_diff"] < 1e-9


@pytest.mark.parametrize("kind", ["local", "affine"])
def test_alignment_parity_bit_identical(kind: str) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("jax")
    assert probe_align_parity(kind, 0, BETA).metrics["value_abs_diff"] < 1e-9


# --- matrix-tree numerical domain (honest det / inverse limit) -----------


@pytest.mark.parametrize("beta", [16.0, 1.0e4])
def test_mtt_wellseparated_twin_and_marginals_hold_at_high_beta(beta: float) -> None:
    """Representative (tree-optimum) arcs keep the twin bit-identical and marginals == autodiff
    across the whole ``beta`` ladder: the exp-Laplacian tends to a *tree's* (non-singular) minor."""
    pytest.importorskip("torch")
    pytest.importorskip("jax")
    assert probe_mtt_parity(0, beta).metrics["value_abs_diff"] < 1e-9
    assert probe_mtt_marginals(0, beta, "torch").ok
    assert probe_mtt_marginals(0, beta, "jax").ok


def test_matrix_tree_singular_at_high_beta_for_cyclic_argmax() -> None:
    """Honest limit the cluster sweep surfaced: when the greedy argmax is a *cycle* rather than a
    tree the exp-Laplacian minor goes singular, so the determinant / inverse lose all
    conditioning at large ``beta`` (this is why the probes use ``_make_arc_mtt``). A tree-optimum
    minor stays perfectly conditioned -- the contrast pins the root cause, not an exception type."""
    torch = pytest.importorskip("torch")
    from omnibias.struct.torch.mtt import _laplacian_tilde

    cyclic = torch.full((3, 3), -5.0, dtype=torch.float64)  # 2 words, optimum 1 <-> 2 (a 2-cycle)
    cyclic[2, 1] = 5.0
    cyclic[1, 2] = 5.0
    assert float(torch.linalg.cond(_laplacian_tilde(cyclic, 1.0)[0])) < 1.0e6  # invertible low beta
    assert float(torch.linalg.cond(_laplacian_tilde(cyclic, 1.0e4)[0])) > 1.0e12  # singular high beta

    tree = torch.full((3, 3), -5.0, dtype=torch.float64)  # both words attach to ROOT (a tree)
    tree[0, 1] = 5.0
    tree[0, 2] = 5.0
    assert float(torch.linalg.cond(_laplacian_tilde(tree, 1.0e4)[0])) < 1.0e3  # tree minor stays sound


# --- distribution operators ----------------------------------------------


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_entropy_identity_and_mc(backend: str) -> None:
    pytest.importorskip(backend)
    rec = probe_entropy(0, BETA, backend)
    assert rec.ok, rec
    assert rec.metrics["identity_err"] < 1e-8


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_sampler_empirical_marginals_converge(backend: str) -> None:
    pytest.importorskip(backend)
    assert probe_sampling(0, BETA, backend).ok


def test_exact_kbest_equals_bruteforce() -> None:
    assert probe_kbest(0).ok
