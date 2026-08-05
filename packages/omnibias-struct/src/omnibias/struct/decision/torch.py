# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Torch :class:`DecisionLayer`: an embeddable certified argmax-over-N decision layer.

The trainable face of :mod:`omnibias.struct.decision`. ``forward`` is the relaxed decision
``softmax(beta * scores)`` (differentiable, batched); ``hard`` is the ``beta -> inf`` decision
``argmax(scores)``; ``certificate`` returns the closed-form
:class:`~omnibias.struct.SelectionCertificate` (value gap ``log(N)/beta``, mode-mass
concentration, ``L^inf`` argmax-stability radius). Reuses
:func:`omnibias.struct.torch.certified_argmax` and ``soft_argmax`` -- no new math.

Terminology: the ``beta -> inf`` annealing is the feasibility / temperature sense of
"collapse", distinct from the founding ``delta -> 0`` bias collapse (``sigma^(K-1)``).
"""

from __future__ import annotations

import torch
from omnibias.struct._core.select import SelectionCertificate
from omnibias.struct.torch.select import certified_argmax, soft_argmax
from torch import Tensor, nn


def soft_decision(scores: Tensor, beta: float, *, axis: int = -1) -> Tensor:
    r"""The relaxed decision ``softmax(beta * scores)`` (``-> one-hot argmax`` as ``beta -> inf``)."""
    out: Tensor = soft_argmax(scores, beta, axis=axis)
    return out


def expected_reward(p: Tensor, rewards: Tensor, *, axis: int = -1) -> Tensor:
    r"""The differentiable decision objective ``E_p[reward] = <p, rewards>`` along ``axis``."""
    return (p * rewards).sum(dim=axis)


def decision_regret(scores_hat: Tensor, rewards: Tensor) -> Tensor:
    r"""Per-sample hard regret of ``argmax(scores_hat)`` under true ``rewards`` (``(n, N)`` -> ``(n,)``)."""
    picked = torch.argmax(scores_hat, dim=-1)
    realized = rewards.gather(-1, picked.unsqueeze(-1)).squeeze(-1)
    return rewards.max(dim=-1).values - realized


class DecisionLayer(nn.Module):
    r"""An embeddable certified decision (argmax-over-``N``) layer.

    Parameters
    ----------
    beta:
        Inverse temperature of the relaxed decision (larger -> sharper / closer to argmax).
    eps:
        Optional default ``L^inf`` perturbation radius for the argmax-stability sub-claim.
    axis:
        The axis the decision is taken over (default the last).
    """

    def __init__(self, beta: float = 10.0, *, eps: float | None = None, axis: int = -1) -> None:
        super().__init__()
        if beta <= 0.0:
            raise ValueError(f"beta must be > 0, got {beta}")
        self.beta = float(beta)
        self.eps = None if eps is None else float(eps)
        self.axis = int(axis)

    def forward(self, scores: Tensor) -> Tensor:
        r"""The relaxed decision ``softmax(beta * scores)`` (differentiable)."""
        out: Tensor = soft_argmax(scores, self.beta, axis=self.axis)
        return out

    def hard(self, scores: Tensor) -> Tensor:
        r"""The ``beta -> inf`` hard decision ``argmax(scores)`` (indices)."""
        return torch.argmax(scores, dim=self.axis)

    def certified(
        self, scores: Tensor, *, eps: float | None = None
    ) -> tuple[Tensor, SelectionCertificate]:
        r"""Relaxed decision + :class:`SelectionCertificate` for a single 1-D score vector."""
        soft, cert = certified_argmax(scores, self.beta, eps=self.eps if eps is None else eps)
        return soft, cert

    def certificate(self, scores: Tensor, *, eps: float | None = None) -> SelectionCertificate:
        r"""The :class:`SelectionCertificate` for a single 1-D score vector (no soft output)."""
        _, cert = certified_argmax(scores, self.beta, eps=self.eps if eps is None else eps)
        return cert

    def certificates(
        self, scores: Tensor, *, eps: float | None = None
    ) -> list[SelectionCertificate]:
        r"""One :class:`SelectionCertificate` per row of a ``(n, N)`` batch of score vectors."""
        s = torch.as_tensor(scores)
        if s.ndim != 2:
            raise ValueError(f"certificates expects a 2-D (n, N) batch, got shape {tuple(s.shape)}")
        out: list[SelectionCertificate] = []
        for row in s:
            _, cert = certified_argmax(row, self.beta, eps=self.eps if eps is None else eps)
            out.append(cert)
        return out

    def extra_repr(self) -> str:
        return f"beta={self.beta}, eps={self.eps}, axis={self.axis}"


__all__ = [
    "DecisionLayer",
    "decision_regret",
    "expected_reward",
    "soft_decision",
]
