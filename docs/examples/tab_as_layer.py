# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Tab layers as neural heads: encoder + soft-tree / boosted arrangement.

Run:

    python docs/examples/tab_as_layer.py

``forward`` is the plugin API (tensor-in / tensor-out). ``as_head(z, kind)``
constructs the head on ``z.device`` / ``z.dtype`` so the dtype move cannot be
forgotten. Joint Adam updates the encoder **and** the head. Tabular ``fit_*``
trainers are optional pretrain, not required.
"""

from __future__ import annotations

import torch
from omnibias.tab.torch.arrangement import ArrangementBoosted, ArrangementClassifier
from omnibias.tab.torch.plugin import TabHead, as_head
from torch import nn


class EncoderHead(nn.Module):
    def __init__(self, head: nn.Module, *, in_features: int = 8) -> None:
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(in_features, head_in_features(head)), nn.Tanh())
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.enc(x))


def _inner(head: nn.Module) -> nn.Module:
    return head.module if isinstance(head, TabHead) else head


def head_in_features(head: nn.Module) -> int:
    inner = _inner(head)
    if isinstance(inner, ArrangementBoosted):
        return int(inner.members[0].n_features)
    if isinstance(inner, ArrangementClassifier):
        return int(inner.n_features)
    return int(inner.config.n_features)


def _joint_adam(model: EncoderHead, X: torch.Tensor, y: torch.Tensor) -> None:
    enc_w = model.enc[0].weight
    inner = _inner(model.head)
    if isinstance(inner, ArrangementBoosted):
        head_w = inner.members[0].W
    else:
        head_w = inner.W
    before_enc = enc_w.detach().clone()
    before_head = head_w.detach().clone()
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    for _ in range(8):
        opt.zero_grad(set_to_none=True)
        logits = model(X)
        if logits.ndim > 1 and logits.shape[-1] == 1:
            logits = logits.reshape(-1)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, y)
        assert torch.isfinite(loss)
        loss.backward()
        opt.step()
    assert not torch.allclose(enc_w, before_enc), "encoder did not move"
    assert not torch.allclose(head_w, before_head), "head did not move"


def softtree_head() -> None:
    print("=== encoder + SoftTreeEnsemble (joint Adam) ===")
    torch.manual_seed(0)
    X = torch.randn(32, 8, dtype=torch.float64)
    y = (X[:, 0] > 0).to(dtype=torch.float64)
    probe = torch.zeros(1, 16, dtype=X.dtype, device=X.device)
    head = as_head(
        probe, "softtree", n_trees=4, depth=2, task="binary", beta_final=8.0, seed=0
    )
    model = EncoderHead(head, in_features=8).to(dtype=X.dtype, device=X.device)
    _joint_adam(model, X, y)
    print("  encoder and SoftTree head both updated")


def boosted_arrangement_head() -> None:
    print("=== encoder + ArrangementBoosted (joint Adam) ===")
    torch.manual_seed(1)
    X = torch.randn(32, 8, dtype=torch.float64)
    y = (X[:, 0] > 0).to(dtype=torch.float64)
    probe = torch.zeros(1, 16, dtype=X.dtype, device=X.device)
    head = as_head(probe, "boosted", n_members=2, n_hyperplanes=2, beta=4.0)
    model = EncoderHead(head, in_features=8).to(dtype=X.dtype, device=X.device)
    _joint_adam(model, X, y)
    print("  encoder and boosted-arrangement head both updated")


def main() -> None:
    softtree_head()
    boosted_arrangement_head()
    print("OK: tab layers plug into a host net; autograd reaches encoder and head.")


if __name__ == "__main__":
    main()
