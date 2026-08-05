# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Differentiable combinatorial relaxations (torch backend).

Oracles:

* ``sinkhorn_normalize`` -> doubly-stochastic (row / column sums == 1).
* ``gumbel_sinkhorn`` -> a hard permutation matrix as ``tau -> 0``.
* ``soft_sort`` -> ``torch.sort`` as ``tau -> 0``; ``soft_sort_permutation`` is
  row-stochastic.
* ``soft_top_k`` -> weights in ``[0, 1]`` summing to exactly ``k``, and the hard
  top-k indicator as ``tau -> 0``.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
torch.set_default_dtype(torch.float64)

import omnibias.graph.torch.ops as G


class TestSinkhorn:
    def test_doubly_stochastic(self) -> None:
        rng = np.random.default_rng(0)
        log_alpha = torch.tensor(rng.normal(size=(6, 6)))
        p = G.sinkhorn_normalize(log_alpha, n_iters=300)
        assert torch.allclose(p.sum(dim=-1), torch.ones(6), atol=1e-9)
        assert torch.allclose(p.sum(dim=-2), torch.ones(6), atol=1e-9)
        assert (p >= 0).all()

    def test_non_square_raises(self) -> None:
        with pytest.raises(ValueError, match="square"):
            G.sinkhorn_normalize(torch.zeros(3, 4))


class TestGumbelSinkhorn:
    def test_low_temperature_is_permutation(self) -> None:
        # A near-identity affinity: at low temperature Gumbel-Sinkhorn (no noise)
        # recovers the identity permutation matrix.
        log_alpha = 3.0 * torch.eye(5)
        p = G.gumbel_sinkhorn(log_alpha, temperature=0.01, n_iters=200)
        assert torch.allclose(p, torch.eye(5), atol=1e-3)

    def test_permutation_recovered_from_scores(self) -> None:
        # Assignment affinity that prefers the reverse permutation.
        perm = torch.tensor([4, 3, 2, 1, 0])
        log_alpha = torch.full((5, 5), -5.0)
        for i, j in enumerate(perm):
            log_alpha[i, j] = 5.0
        p = G.gumbel_sinkhorn(log_alpha, temperature=0.05, n_iters=200)
        recovered = p.argmax(dim=-1)
        assert torch.equal(recovered, perm)
        # doubly-stochastic at any temperature
        assert torch.allclose(p.sum(dim=-1), torch.ones(5), atol=1e-6)

    def test_temperature_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="temperature"):
            G.gumbel_sinkhorn(torch.eye(3), temperature=0.0)


class TestSoftSort:
    def test_permutation_row_stochastic(self) -> None:
        s = torch.tensor([3.0, 1.0, 4.0, 1.5, 2.0])
        p = G.soft_sort_permutation(s, temperature=0.5)
        assert torch.allclose(p.sum(dim=-1), torch.ones(5), atol=1e-10)
        assert (p >= 0).all()

    @pytest.mark.parametrize("descending", [True, False])
    def test_hard_limit_matches_torch_sort(self, descending: bool) -> None:
        s = torch.tensor([3.0, 1.0, 4.0, 1.5, 9.0, 2.0])
        out = G.soft_sort(s, temperature=1e-4, descending=descending)
        ref = torch.sort(s, descending=descending).values
        assert torch.allclose(out, ref, atol=1e-6)

    def test_differentiable(self) -> None:
        s = torch.tensor([3.0, 1.0, 4.0, 1.5], requires_grad=True)
        G.soft_sort(s, temperature=0.7).sum().backward()
        assert s.grad is not None and torch.isfinite(s.grad).all()


class TestSoftTopK:
    def test_sums_to_k_at_any_temperature(self) -> None:
        s = torch.tensor([3.0, 1.0, 4.0, 1.5, 9.0, 2.0])
        for tau in (0.01, 0.5, 3.0):
            m = G.soft_top_k(s, 3, temperature=tau)
            assert abs(float(m.sum()) - 3.0) < 1e-9
            assert (m >= -1e-12).all() and (m <= 1.0 + 1e-9).all()

    def test_hard_limit_selects_top_k(self) -> None:
        s = torch.tensor([3.0, 1.0, 4.0, 1.5, 9.0, 2.0])
        m = G.soft_top_k(s, 3, temperature=1e-4)
        selected = set((m > 0.5).nonzero().flatten().tolist())
        # top-3 values 9, 4, 3 at indices 4, 2, 0
        assert selected == {4, 2, 0}

    def test_bad_k_raises(self) -> None:
        with pytest.raises(ValueError, match=r"k must be"):
            G.soft_top_k(torch.zeros(4), 5)
