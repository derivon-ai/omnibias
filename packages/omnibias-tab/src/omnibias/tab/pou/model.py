# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Trainable TabPOU module: preprocess tokens + axis forest + optional residual."""

from __future__ import annotations

import numpy as np
import torch
from omnibias.tab._core.params import TabParams
from omnibias.tab.pou.config import TabPOUConfig
from omnibias.tab.pou.ensemble import TabMResidual
from omnibias.tab.pou.preprocess import TabPreprocessor
from omnibias.tab.torch.embed import BandFeatureEmbedder
from omnibias.tab.torch.model import SoftTreeEnsemble
from torch import Tensor, nn

_DTYPE = torch.float64


class TabPOU(nn.Module):
    r"""Axis-aligned POU booster with optional band tokens and TabM residual.

    ``forward`` expects already-preprocessed numeric tokens (the matrix
    :meth:`transform` produces). :meth:`predict` runs the numpy preprocessor
    then the torch graph.
    """

    def __init__(
        self,
        config: TabPOUConfig,
        tree: SoftTreeEnsemble,
        *,
        preprocessor: TabPreprocessor | None = None,
        embedder: BandFeatureEmbedder | None = None,
        residual: TabMResidual | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.tree = tree
        self.preprocessor = preprocessor
        self.embedder = embedder
        self.residual = residual
        self.binarize_eval = False

    def transform(self, X: np.ndarray) -> np.ndarray:
        r"""Raw rows -> the numeric matrix the forest was trained on."""
        prep = self.preprocessor
        Xs = prep.transform(X) if prep is not None else np.asarray(X, dtype=np.float64)
        if self.embedder is None:
            return Xs
        with torch.no_grad():
            Xt = torch.as_tensor(Xs, dtype=_DTYPE)
            E = self.embedder(Xt).cpu().numpy()
        if self.config.concat_raw:
            return np.concatenate([E, Xs], axis=1)
        return np.asarray(E, dtype=np.float64)

    def forward(self, X: Tensor, beta: float | None = None) -> Tensor:
        F = self.tree(X, beta=beta)
        if self.residual is not None:
            pou = None
            if self.residual.n_regions is not None:
                pou = self.tree.memberships(X, beta=beta).mean(dim=-2)
            F = F + self.residual(X, pou_weights=pou)
        return F

    def score(self, X: np.ndarray, beta: float | None = None) -> np.ndarray:
        Xtok = self.transform(X)
        if self.binarize_eval:
            from omnibias.tab._core.forward import hard_forward_np

            arr = np.asarray(hard_forward_np(self.tree.to_params(), Xtok), dtype=np.float64)
            if self.residual is not None:
                with torch.no_grad():
                    Xt = torch.as_tensor(Xtok, dtype=_DTYPE)
                    pou = None
                    if self.residual.n_regions is not None:
                        pou = self.tree.memberships(Xt, beta=beta).mean(dim=-2)
                    arr = arr + self.residual(Xt, pou_weights=pou).cpu().numpy()
            return arr
        with torch.no_grad():
            F = self.forward(torch.as_tensor(Xtok, dtype=_DTYPE), beta=beta)
        out: np.ndarray = F.detach().cpu().numpy()
        return out

    def predict(self, X: np.ndarray, beta: float | None = None) -> np.ndarray:
        F = self.score(X, beta=beta)
        task = self.config.task
        if task == "binary":
            return (F[:, 0] > 0.0).astype(np.float64)
        if task == "multiclass":
            return np.argmax(F, axis=-1).astype(np.float64)
        return F if F.shape[1] > 1 else F[:, 0]

    def predict_proba(self, X: np.ndarray, beta: float | None = None) -> np.ndarray:
        F = self.score(X, beta=beta)
        if self.config.task == "binary":
            from omnibias.tab._core.forward import sigmoid_np

            return sigmoid_np(F[:, 0])
        if self.config.task == "multiclass":
            from omnibias.tab._core.forward import softmax_np

            return softmax_np(F)
        raise ValueError("predict_proba is only defined for classification")

    def to_params(self) -> TabParams:
        return self.tree.to_params()

    def extra_repr(self) -> str:
        return (
            f"task={self.config.task!r}, depth={self.config.depth}, "
            f"use_embed={self.embedder is not None}, residual={self.residual is not None}"
        )


def tokens_from_scaled(
    Xs: np.ndarray,
    embedder: BandFeatureEmbedder | None,
    *,
    concat_raw: bool,
) -> np.ndarray:
    r"""Scale matrix ``Xs`` -> forest input tokens (numpy)."""
    if embedder is None:
        return np.asarray(Xs, dtype=np.float64)
    with torch.no_grad():
        E = embedder(torch.as_tensor(np.asarray(Xs, dtype=np.float64), dtype=_DTYPE))
        En = E.cpu().numpy()
    if concat_raw:
        return np.concatenate([En, np.asarray(Xs, dtype=np.float64)], axis=1)
    return np.asarray(En, dtype=np.float64)


def make_embedder(
    Xs: np.ndarray, config: TabPOUConfig, *, freeze: bool = True
) -> BandFeatureEmbedder:
    return BandFeatureEmbedder(
        n_features=int(Xs.shape[1]),
        n_bins=int(config.n_bins),
        role=str(config.embed_role),
        init="quantile",
        learnable_beta=not freeze,
        learnable_thresholds=not freeze,
        X_ref=Xs,
    )


__all__ = ["TabPOU", "make_embedder", "tokens_from_scaled"]
