# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Train-only robust scale + smooth clip + low-card one-hot (theory 05-04).

RealMLP (Holzmüller et al., NeurIPS 2024) uses robust scaling and smooth clipping
on numerical columns. Low-cardinality integer columns are one-hot encoded (card
``<= onehot_max_card``) instead of being treated as ordered scalars. Fit on train
only; transform is a pure numpy map.

No ``beta`` / temperature schedule lives here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _as_2d(X: np.ndarray) -> FloatArray:
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {arr.shape}")
    return arr


@dataclass
class TabPreprocessor:
    r"""Median/IQR scale, ``clip * tanh(z / clip)``, optional low-card one-hot."""

    clip: float = 5.0
    onehot_max_card: int = 16
    robust_scale: bool = True
    n_features_in: int = 0
    median: FloatArray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    iqr: FloatArray = field(default_factory=lambda: np.ones(0, dtype=np.float64))
    cat_cols: tuple[int, ...] = ()
    cat_levels: tuple[FloatArray, ...] = ()
    num_cols: tuple[int, ...] = ()
    n_features_out: int = 0

    def fit(self, X: np.ndarray) -> TabPreprocessor:
        Xv = _as_2d(X)
        n, d = Xv.shape
        self.n_features_in = d
        cat: list[int] = []
        levels: list[FloatArray] = []
        num: list[int] = []
        max_card = int(self.onehot_max_card)
        for j in range(d):
            col = Xv[:, j]
            uniq = np.unique(col)
            looks_int = np.max(np.abs(col - np.round(col))) < 1e-8
            if looks_int and 2 <= uniq.size <= max_card:
                cat.append(j)
                levels.append(np.sort(uniq))
            else:
                num.append(j)
        self.cat_cols = tuple(cat)
        self.cat_levels = tuple(levels)
        self.num_cols = tuple(num)
        if num:
            med = np.median(Xv[:, num], axis=0)
            q75 = np.quantile(Xv[:, num], 0.75, axis=0)
            q25 = np.quantile(Xv[:, num], 0.25, axis=0)
            iqr = np.clip(q75 - q25, 1e-8, None)
            self.median = np.asarray(med, dtype=np.float64)
            self.iqr = np.asarray(iqr, dtype=np.float64)
        else:
            self.median = np.zeros(0, dtype=np.float64)
            self.iqr = np.ones(0, dtype=np.float64)
        n_cat = int(sum(lv.size for lv in self.cat_levels))
        self.n_features_out = len(num) + n_cat
        return self

    def transform(self, X: np.ndarray) -> FloatArray:
        Xv = _as_2d(X)
        if Xv.shape[1] != self.n_features_in:
            raise ValueError(
                f"expected {self.n_features_in} columns, got {Xv.shape[1]}"
            )
        parts: list[FloatArray] = []
        if self.num_cols:
            z = Xv[:, list(self.num_cols)]
            if self.robust_scale:
                z = (z - self.median[None, :]) / self.iqr[None, :]
            c = float(self.clip)
            if c > 0.0:
                z = c * np.tanh(z / c)
            parts.append(z)
        for j, levels in zip(self.cat_cols, self.cat_levels, strict=True):
            col = Xv[:, j]
            oh = (col[:, None] == levels[None, :]).astype(np.float64)
            parts.append(oh)
        if not parts:
            return np.zeros((Xv.shape[0], 0), dtype=np.float64)
        return np.concatenate(parts, axis=1)

    def fit_transform(self, X: np.ndarray) -> FloatArray:
        return self.fit(X).transform(X)


__all__ = ["TabPreprocessor"]
