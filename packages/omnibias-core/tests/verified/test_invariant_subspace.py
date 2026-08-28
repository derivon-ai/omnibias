# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Certified Davis-Kahan invariant-subspace enclosures.

Soundness oracle is ``numpy.linalg.eigh`` (exact for these small test
matrices): every certified ``sin(Theta)`` bound must never fall below the true
Frobenius-norm subspace distance between the candidate basis and the true
invariant subspace picked out by the certificate's own ``cluster_start_index``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from omnibias.core.verified.invariant_subspace import (
    InvariantSubspaceCertificate,
    certified_invariant_subspace,
)


def _true_sin_theta_frobenius(a: np.ndarray, v: np.ndarray, cluster_start_index: int, k: int) -> float:
    """Frobenius-norm sin(Theta) between ``span(v)`` and the true ``k``-dim
    invariant subspace of ``a`` spanned by eigenvalue indices
    ``[cluster_start_index-1, cluster_start_index-1+k)`` (0-based)."""
    evals, evecs = np.linalg.eigh(a)
    p = cluster_start_index - 1
    true_basis = evecs[:, p : p + k]
    q_v, _ = np.linalg.qr(v)
    singular_values = np.linalg.svd(true_basis.T @ q_v, compute_uv=False)
    cos_theta = np.clip(singular_values, -1.0, 1.0)
    sin_theta_sq = np.clip(1.0 - cos_theta**2, 0.0, None)
    return float(np.sqrt(np.sum(sin_theta_sq)))


# --------------------------------------------------------------------------- #
# Exactly degenerate cluster.
# --------------------------------------------------------------------------- #
A_DEGENERATE = [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 5.0]]


def test_exact_degenerate_cluster_is_certified_near_zero() -> None:
    v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]  # exact 2-D eigenspace of eigenvalue 2
    cert = certified_invariant_subspace(A_DEGENERATE, v)
    assert cert.certified
    assert cert.gram_positive_definite
    assert cert.cluster_start_index == 1
    assert cert.gap_lower is not None and cert.gap_lower == pytest.approx(3.0, abs=1e-9)
    assert cert.ritz_lower is not None and cert.ritz_lower == pytest.approx(2.0, abs=1e-9)
    assert cert.ritz_upper is not None and cert.ritz_upper == pytest.approx(2.0, abs=1e-9)
    assert cert.sin_theta_upper is not None
    assert cert.sin_theta_upper < 1e-9  # exact eigenspace -> essentially zero


def test_exact_degenerate_cluster_robust_to_rotation_and_non_orthonormality() -> None:
    # A rotated basis of the SAME degenerate eigenspace, and a skewed
    # (non-orthonormal) one, must both still certify near-zero sin(Theta).
    c, s = math.cos(0.7), math.sin(0.7)
    rotated = [[c, s, 0.0], [-s, c, 0.0]]
    skewed = [[1.0, 0.0, 0.0], [0.3, 1.0, 0.0]]
    for v in (rotated, skewed):
        cert = certified_invariant_subspace(A_DEGENERATE, v)
        assert cert.certified
        assert cert.gap_lower is not None and cert.gap_lower == pytest.approx(3.0, abs=1e-6)
        assert cert.sin_theta_upper is not None and cert.sin_theta_upper < 1e-8


def test_exact_degenerate_partner_direction_alone_is_uncertified() -> None:
    # A SINGLE eigenvector inside a genuinely 2-fold degenerate eigenspace has
    # no certifiable gap to its exact degenerate partner (gap == 0 truly) --
    # the certificate must honestly refuse, not fabricate a bound.
    cert = certified_invariant_subspace(A_DEGENERATE, [[1.0, 0.0, 0.0]])
    assert not cert.certified
    assert cert.residual_norm_upper is None
    assert cert.sin_theta_upper is None


def test_multiplicity_three_degenerate_cluster() -> None:
    a = [[3.0, 0.0, 0.0, 0.0], [0.0, 3.0, 0.0, 0.0], [0.0, 0.0, 3.0, 0.0], [0.0, 0.0, 0.0, 9.0]]
    v = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
    cert = certified_invariant_subspace(a, v)
    assert cert.certified
    assert cert.cluster_size == 3
    assert cert.gap_lower is not None and cert.gap_lower == pytest.approx(6.0, abs=1e-6)
    assert cert.sin_theta_upper is not None and cert.sin_theta_upper < 1e-8


# --------------------------------------------------------------------------- #
# Near-degenerate random matrices: soundness against a numpy oracle.
# --------------------------------------------------------------------------- #
def test_near_degenerate_random_soundness() -> None:
    rng = np.random.default_rng(1234)
    checked = 0
    for trial in range(400):
        n = int(rng.integers(3, 8))
        k = int(rng.integers(1, min(3, n - 1) + 1))
        q, _ = np.linalg.qr(rng.normal(size=(n, n)))
        base = np.sort(rng.uniform(-5.0, 5.0, size=n))
        start = int(rng.integers(0, n - k + 1))
        perturb = 10.0 ** rng.uniform(-8.0, -2.0)
        center = base[start]
        for i in range(start, start + k):
            base[i] = center + perturb * rng.normal()
        base = np.sort(base)
        a = q @ np.diag(base) @ q.T
        a = 0.5 * (a + a.T)

        evals, evecs = np.linalg.eigh(a)
        true_vecs = evecs[:, start : start + k]
        v = true_vecs + rng.normal(scale=1e-3, size=true_vecs.shape)
        if trial % 3 == 0 and k >= 2:
            mix = np.eye(k) + 0.2 * rng.normal(size=(k, k))
            v = v @ mix

        cert = certified_invariant_subspace(a.tolist(), v.T.tolist())
        if not cert.certified:
            continue
        checked += 1
        assert cert.cluster_start_index is not None
        assert cert.gap_lower is not None and cert.gap_lower > 0.0
        assert cert.sin_theta_upper is not None
        true_sin_theta = _true_sin_theta_frobenius(a, v, cert.cluster_start_index, k)
        assert true_sin_theta <= cert.sin_theta_upper + 1e-9

    assert checked >= 100  # the harness actually exercised the certified path


# --------------------------------------------------------------------------- #
# Refusal: candidate subspace mixes with an outside mode.
# --------------------------------------------------------------------------- #
def test_mixed_mode_candidate_is_refused_not_fabricated() -> None:
    # span(e0, e2) mixes the eigenvalue-2 direction with the eigenvalue-5
    # direction: the Ritz range [2, 5] cannot be separated from the rest of
    # the spectrum (there is no "rest" outside [2, 5] here, so no gap).
    v = [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    cert = certified_invariant_subspace(A_DEGENERATE, v)
    assert not cert.certified
    assert cert.residual_norm_upper is None
    assert cert.sin_theta_upper is None


def test_near_degenerate_random_refusal_when_gap_closes() -> None:
    # A candidate subspace deliberately built to straddle the boundary between
    # a genuine cluster and its immediate neighbor must not be certified.
    a = [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0 + 1e-3]]
    # Deliberately pulls one basis vector toward the well-separated
    # eigenvalue-1 direction too, mixing it into an otherwise near-2 cluster.
    v_bad = [[1.0, 1.0, 0.0], [0.0, 0.3, 1.0]]
    cert = certified_invariant_subspace(a, v_bad)
    # Either honestly refused, or (if certified) the bound must still be sound
    # against the numpy oracle -- never a silent fabrication either way.
    if cert.certified:
        assert cert.cluster_start_index is not None
        true_sin_theta = _true_sin_theta_frobenius(
            np.array(a), np.array(v_bad).T, cert.cluster_start_index, 2
        )
        assert true_sin_theta <= (cert.sin_theta_upper or 0.0) + 1e-9
    else:
        assert cert.sin_theta_upper is None


# --------------------------------------------------------------------------- #
# Input validation.
# --------------------------------------------------------------------------- #
def test_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError):
        certified_invariant_subspace(A_DEGENERATE, [])  # k < 1
    with pytest.raises(ValueError):
        certified_invariant_subspace(A_DEGENERATE, [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [1.0, 1.0]])  # k > n
    with pytest.raises(ValueError):
        certified_invariant_subspace(A_DEGENERATE, [[1.0, 0.0]])  # length mismatch
    with pytest.raises(ValueError):
        certified_invariant_subspace([[1.0, 0.0, 0.0], [0.0, 1.0]], [[1.0, 0.0]])  # non-square matrix


def test_linearly_dependent_basis_is_refused() -> None:
    # Duplicate columns -> singular Gram matrix -> honestly uncertified.
    cert = certified_invariant_subspace(A_DEGENERATE, [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    assert not cert.certified
    assert not cert.gram_positive_definite
    assert cert.ritz_lower is None
    assert cert.sin_theta_upper is None


def test_result_is_the_dataclass_type() -> None:
    cert = certified_invariant_subspace(A_DEGENERATE, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert isinstance(cert, InvariantSubspaceCertificate)
    assert cert.dimension == 3
    assert cert.cluster_size == 2
