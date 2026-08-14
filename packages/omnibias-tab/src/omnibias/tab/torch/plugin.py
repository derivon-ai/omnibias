# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Plugin helpers: construct a tab head on the host tensor's device and dtype."""

from __future__ import annotations

from typing import Any

from omnibias.tab._core.config import SoftTreeConfig
from omnibias.tab.torch.arrangement import ArrangementBoosted, ArrangementClassifier
from omnibias.tab.torch.model import SoftTreeEnsemble
from torch import Tensor, nn

_KINDS = ("softtree", "arrangement", "boosted", "arrangement_boosted")


class TabHead(nn.Module):
    """Plugin wrapper: inner SoftTree / Arrangement / Boosted, logits ``(..., k)``.

    ``forward`` unsqueezes a squeezed 1-D logit so the trailing class axis is
    always present. Attribute access forwards to ``.module`` after the usual
    ``nn.Module`` lookup, so ``head.W`` / ``head.members`` keep working.
    """

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, *args: Any, **kwargs: Any) -> Tensor:
        out = self.module(*args, **kwargs)
        if out.ndim == 1:
            out = out.unsqueeze(-1)
        return out

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            modules = self.__dict__.get("_modules")
            inner = None if modules is None else modules.get("module")
            if inner is None:
                raise
            return getattr(inner, name)


def as_head(z: Tensor, kind: str = "softtree", **kwargs: Any) -> TabHead:
    r"""Build a :class:`TabHead` that already lives on ``z.device`` / ``z.dtype``.

    Infers ``n_features`` from ``z.shape[-1]``. Constructors stay float64 CPU
    internally, then ``.to(device, dtype)`` so plugin callers cannot forget the
    move. ``kind`` is ``"softtree"``, ``"arrangement"``, or ``"boosted"``.
    """
    if z.ndim < 1:
        raise ValueError("z must have a trailing feature dim")
    n_features = int(z.shape[-1])
    name = str(kind).lower()
    if name == "softtree":
        kwargs.setdefault("n_features", n_features)
        kwargs["n_features"] = n_features
        inner: nn.Module = SoftTreeEnsemble(SoftTreeConfig(**kwargs))
    elif name == "arrangement":
        n_hyperplanes = int(kwargs.pop("n_hyperplanes", 2))
        inner = ArrangementClassifier(n_features, n_hyperplanes, **kwargs)
    elif name in ("boosted", "arrangement_boosted"):
        n_members = int(kwargs.pop("n_members", 2))
        n_hyperplanes = int(kwargs.pop("n_hyperplanes", 2))
        learning_rate = float(kwargs.pop("learning_rate", 0.3))
        base = float(kwargs.pop("base", 0.0))
        members = [
            ArrangementClassifier(n_features, n_hyperplanes, **kwargs)
            for _ in range(max(1, n_members))
        ]
        inner = ArrangementBoosted(
            members, learning_rate=learning_rate, base=base
        )
    else:
        raise ValueError(f"kind must be one of {_KINDS}, got {kind!r}")
    return TabHead(inner.to(device=z.device, dtype=z.dtype))


__all__ = ["TabHead", "as_head"]
