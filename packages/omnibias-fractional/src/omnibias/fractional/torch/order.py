# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Learnable fractional order (torch): a bounded, differentiable order parameter.

The fractional operators in :mod:`omnibias.fractional.torch.ops` accept a
tensor-valued ``alpha`` and are differentiable w.r.t. it.  :class:`LearnableOrder`
wraps an unconstrained parameter and maps it through a sigmoid into the open band
``(lo, hi)``, so the order can be trained end-to-end without wandering into an
unstable region (e.g. keep it in ``(0, 2)`` for a first-/second-order regime).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class LearnableOrder(nn.Module):
    r"""A differentiable fractional order constrained to the open interval ``(lo, hi)``.

    Stores an unconstrained ``raw`` parameter and returns
    ``alpha = lo + (hi - lo) * sigmoid(raw)``.  ``raw`` is initialised (via the
    logit) so the initial order equals ``init``.  Pass ``module()`` (or
    ``module.alpha``) straight into a fractional op's ``alpha`` argument.
    """

    def __init__(self, init: float = 0.5, *, lo: float = 0.0, hi: float = 2.0) -> None:
        super().__init__()
        if not lo < hi:
            raise ValueError(f"require lo < hi, got lo={lo}, hi={hi}")
        if not (lo < init < hi):
            raise ValueError(f"init {init} must lie in the open interval ({lo}, {hi})")
        p = (init - lo) / (hi - lo)
        self.raw = nn.Parameter(torch.tensor(math.log(p / (1.0 - p))))
        self.lo = float(lo)
        self.hi = float(hi)

    def forward(self) -> Tensor:
        return self.lo + (self.hi - self.lo) * torch.sigmoid(self.raw)

    @property
    def alpha(self) -> Tensor:
        """The current constrained order (alias for :meth:`forward`)."""
        return self.forward()

    def extra_repr(self) -> str:
        return f"lo={self.lo}, hi={self.hi}"


__all__ = ["LearnableOrder"]
