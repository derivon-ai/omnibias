# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Learnable soft-binning feature embedding + local target consistency (theory 05-03).

:class:`BandFeatureEmbedder` reimplements :func:`omnibias.core.band_embed.band_embed` /
:func:`omnibias.core.band_embed.integral_embed`'s closed form natively in torch
(``torch.sigmoid`` / ``torch.nn.functional.softplus``, not a numpy call) so it
differentiates through *learnable* per-feature thresholds and a learnable ``beta`` --
the same window as :mod:`omnibias.torch.blocks.operator`'s ``OperatorBlock(op="band"|
"integral")``, matching that module's own precedent of reimplementing the shared closed
form per backend rather than calling into the framework-free numpy reference at runtime.
Parity with the numpy reference and the jax twin (:mod:`omnibias.tab.jax.embed`) is a test
(``test_embed_parity.py``), not shared code.

:func:`local_target_consistency_loss` is the closed-form angle on the source paper's
(Kartashev et al., arXiv:2509.04430, section 5.1) "local target consistency": neighbors in
embedding space should have similar targets. Given a usable ``df/dx`` (only available where
the clean target is itself a known differentiable function -- section 10 of the spec is
explicit that this is *not* every tabular dataset), the embedder's pullback metric
``J^T J`` (``J = de/dx``) should not suppress the direction ``df/dx``; this loss maximizes a
stable surrogate of the Rayleigh quotient ``R = ||J df/dx||^2 / ||df/dx||^2`` via a single
Jacobian-vector product (:func:`torch.autograd.functional.jvp`), with no sampled triplets.

Terminology: the embedder's ``beta -> inf`` hardens each soft bin into a crisp histogram
indicator -- **temperature collapse** (feasibility sense), not the founding ``delta -> 0``
bias collapse. No founding collapse limit appears in this module.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from omnibias.core.band_embed import quantile_thresholds
from torch import Tensor, nn

_DTYPE = torch.float64
_ROLES = ("band", "integral")
_INITS = ("quantile", "uniform")


def _inverse_softplus(value: Tensor) -> Tensor:
    """``log(expm1(value))``, the inverse of :func:`torch.nn.functional.softplus`."""
    return torch.log(torch.expm1(value))


def _stable_softplus(z: Tensor) -> Tensor:
    r"""``log(1 + exp(z))``, stable for any ``z`` (the torch twin of
    :func:`omnibias.core.band_embed._softplus`'s ``max(z,0) + log1p(exp(-|z|))`` form).

    :func:`torch.nn.functional.softplus` is deliberately **not** used here: its default
    ``threshold=20`` reverts to the plain linear function ``z`` once ``beta*z > 20``,
    silently dropping the ``log1p(exp(-|z|))`` correction -- fine for typical NN use, but
    that correction is still ``>= 1e-9`` up to ``z ~ 20.7`` (``log1p(exp(-20.7)) ~ 1e-9``),
    which breaks this module's bit-identical (``~1e-9``) parity with the numpy reference at
    the ``beta*(x-t)`` magnitudes ``BandFeatureEmbedder``'s large-``beta`` regime reaches
    (a large learnable ``beta`` -- section 4(e)'s temperature collapse -- is exactly the
    intended end state, not an edge case to shrug off).
    """
    return torch.clamp(z, min=0.0) + torch.log1p(torch.exp(-torch.abs(z)))


class BandFeatureEmbedder(nn.Module):
    r"""Per-feature open-tail soft-histogram (or soft-cumulative) embedding.

    Maps ``X`` of shape ``(..., n_features)`` to ``(..., n_features * (n_bins + 1))``: for
    each input feature, ``n_bins`` learnable interior thresholds give ``n_bins + 1`` soft
    bins, using exactly the :func:`omnibias.core.band_embed.band_embed` window formula when
    ``role="band"`` (every row's ``(n_bins + 1)``-block then sums to ``1`` exactly, for any
    ``beta``) or its closed-form antiderivative
    (:func:`omnibias.core.band_embed.integral_embed`) when ``role="integral"``.

    The thresholds are parameterized as ``t_first`` plus ``n_bins - 1`` positive gaps in
    ``softplus`` space, so they stay strictly increasing under unconstrained gradient steps
    (no explicit sort, no constraint violation possible).

    Parameters
    ----------
    n_features : int
        Input feature count.
    n_bins : int, default 16
        Number of *interior thresholds* per feature (the embedding has ``n_bins + 1`` soft
        bins per feature -- matching :func:`omnibias.core.band_embed.band_embed`'s ``J``).
    beta_init : float, default 1.0
    role : ``"band"`` | ``"integral"``, default ``"band"``
        Which half of the closed-form pair (section 3, ``ftc_block``) to expose as
        features: the bounded soft-histogram density, or its unbounded closed-form
        antiderivative.
    init : ``"quantile"`` | ``"uniform"``, default ``"quantile"``
        Threshold initialization. ``"quantile"`` requires ``X_ref`` (the empirical
        per-feature quantiles of a reference batch, typically the training data, via
        :func:`omnibias.core.band_embed.quantile_thresholds`); ``"uniform"`` spaces
        thresholds evenly over ``uniform_range`` and needs no reference data.
    learnable_beta, learnable_thresholds : bool, default True
    X_ref : array_like, optional
        Reference batch ``(n, n_features)`` for ``init="quantile"`` (required then).
    thresholds_init : array_like, optional
        Escape hatch: exact ``(n_features, n_bins)`` strictly-increasing thresholds,
        overriding ``init`` entirely (used by the parity tests to pin known values).
    uniform_range : tuple[float, float], default (-1.0, 1.0)
        Range for ``init="uniform"``.
    """

    def __init__(
        self,
        n_features: int,
        n_bins: int = 16,
        *,
        beta_init: float = 1.0,
        role: str = "band",
        init: str = "quantile",
        learnable_beta: bool = True,
        learnable_thresholds: bool = True,
        X_ref: Any = None,
        thresholds_init: Any = None,
        uniform_range: tuple[float, float] = (-1.0, 1.0),
    ) -> None:
        super().__init__()
        if n_features < 1:
            raise ValueError(f"n_features must be >= 1, got {n_features}")
        if n_bins < 1:
            raise ValueError(f"n_bins must be >= 1, got {n_bins}")
        if role not in _ROLES:
            raise ValueError(f"role must be one of {_ROLES}, got {role!r}")
        if init not in _INITS:
            raise ValueError(f"init must be one of {_INITS}, got {init!r}")
        self.n_features = int(n_features)
        self.n_bins = int(n_bins)
        self.role = role

        if thresholds_init is not None:
            t0 = torch.as_tensor(np.asarray(thresholds_init, dtype=np.float64), dtype=_DTYPE)
        elif init == "quantile":
            if X_ref is None:
                raise ValueError('init="quantile" requires X_ref (a reference batch to take quantiles of)')
            X_np = np.asarray(X_ref, dtype=np.float64)
            if X_np.ndim != 2 or X_np.shape[1] != n_features:
                raise ValueError(f"X_ref must have shape (n, {n_features}), got {X_np.shape}")
            thresholds = np.stack([quantile_thresholds(X_np[:, k], n_bins + 1) for k in range(n_features)])
            t0 = torch.as_tensor(thresholds, dtype=_DTYPE)
        else:  # init == "uniform"
            lo, hi = uniform_range
            t0 = torch.linspace(float(lo), float(hi), n_bins, dtype=_DTYPE).unsqueeze(0).repeat(n_features, 1)

        if tuple(t0.shape) != (n_features, n_bins):
            raise ValueError(f"thresholds must have shape ({n_features}, {n_bins}), got {tuple(t0.shape)}")
        if n_bins > 1 and torch.any(t0[:, 1:] - t0[:, :-1] <= 0.0):
            raise ValueError("thresholds must be strictly increasing along the last axis")

        t_first = t0[:, 0].clone()
        gaps0 = t0[:, 1:] - t0[:, :-1] if n_bins > 1 else None
        if learnable_thresholds:
            self.t_first = nn.Parameter(t_first)
            self._raw_gaps = nn.Parameter(_inverse_softplus(gaps0)) if gaps0 is not None else None
        else:
            self.register_buffer("t_first", t_first, persistent=True)
            if gaps0 is not None:
                self.register_buffer("_raw_gaps", _inverse_softplus(gaps0), persistent=True)
            else:
                self._raw_gaps = None

        beta0 = torch.tensor(float(beta_init), dtype=_DTYPE)
        if learnable_beta:
            self._beta = nn.Parameter(beta0)
        else:
            self.register_buffer("_beta", beta0, persistent=True)

    @property
    def beta(self) -> float:
        return float(self._beta.detach().cpu().item())

    def set_beta(self, beta: float) -> None:
        with torch.no_grad():
            self._beta.fill_(float(beta))

    def thresholds(self) -> Tensor:
        r"""Current ``(n_features, n_bins)`` strictly-increasing thresholds."""
        if self._raw_gaps is None:
            return self.t_first.unsqueeze(-1)
        gaps = F.softplus(self._raw_gaps)
        return torch.cat(
            [self.t_first.unsqueeze(-1), self.t_first.unsqueeze(-1) + torch.cumsum(gaps, dim=-1)],
            dim=-1,
        )

    def forward(self, X: Tensor, beta: float | None = None) -> Tensor:
        r"""Embed ``X`` of shape ``(..., n_features)`` into ``(..., n_features * (n_bins + 1))``."""
        if X.shape[-1] != self.n_features:
            raise ValueError(f"X's last dim must be n_features={self.n_features}, got {X.shape[-1]}")
        t = self.thresholds()  # (n_features, n_bins)
        b = self._beta if beta is None else X.new_tensor(float(beta), dtype=t.dtype)
        z = b * (X.unsqueeze(-1) - t)  # (..., n_features, n_bins)
        if self.role == "band":
            bins = self._band_bins(z)
        else:
            bins = self._integral_bins(X, t, b, z)
        return bins.reshape(*X.shape[:-1], self.n_features * (self.n_bins + 1))

    @staticmethod
    def _band_bins(z: Tensor) -> Tensor:
        sig = torch.sigmoid(z)
        ones = torch.ones_like(sig[..., :1])
        zeros = torch.zeros_like(sig[..., :1])
        padded = torch.cat([ones, sig, zeros], dim=-1)  # (..., n_features, n_bins + 2)
        bins: Tensor = padded[..., :-1] - padded[..., 1:]
        return bins

    @staticmethod
    def _integral_bins(X: Tensor, t: Tensor, beta: Tensor, z: Tensor) -> Tensor:
        r"""``integral_embed``'s bins, closed form (see :mod:`omnibias.core.band_embed`)."""
        sp = _stable_softplus(z)  # (..., n_features, n_bins)
        bin0 = (beta * (X - t[..., 0]) - sp[..., 0]).unsqueeze(-1)
        zeros = torch.zeros_like(sp[..., :1])
        padded = torch.cat([sp, zeros], dim=-1)  # (..., n_features, n_bins + 1)
        tail = padded[..., :-1] - padded[..., 1:]  # (..., n_features, n_bins)
        bins: Tensor = torch.cat([bin0, tail], dim=-1)
        return bins

    def extra_repr(self) -> str:
        return f"n_features={self.n_features}, n_bins={self.n_bins}, role={self.role!r}, beta={self.beta:.4g}"


def local_target_consistency_loss(
    embedder: BandFeatureEmbedder,
    X: Tensor,
    *,
    df_dx: Tensor,
    eps: float = 1e-8,
) -> Tensor:
    r"""Closed-form, sample-free local-target-consistency surrogate (spec section 4(f)).

    Given the embedder ``e = embedder(X)`` and the clean target's gradient ``df_dx`` at the
    same points, this maximizes a stable surrogate of the Rayleigh quotient

    .. math:: R(x) = \frac{\lVert J \, df/dx \rVert^2}{\lVert df/dx \rVert^2}, \quad J = de/dx

    per-row, where ``J df/dx`` is computed as a single Jacobian-vector product (never the
    full ``(D, d)`` Jacobian): the embedder treats each input feature independently, so its
    batched JVP evaluated with a per-row tangent ``df_dx`` already gives the desired per-row
    ``J_i @ df_dx_i`` with no manual block-diagonal bookkeeping. Returns
    ``mean(log(||df_dx||^2 + eps) - log(||J df_dx||^2 + eps))``, whose gradient is bounded
    even as the ratio approaches ``0`` or grows large -- minimizing it maximizes
    ``log R`` and hence ``R``, without ever forming ``R`` as an explicit (unstable) ratio.

    Requires a usable ``df_dx``, e.g. from a known-closed-form synthetic target (see spec
    section 10 for exactly where this is and is not available on real tabular data).
    """
    if X.shape != df_dx.shape:
        raise ValueError(f"X and df_dx must share a shape, got {tuple(X.shape)} and {tuple(df_dx.shape)}")

    def _embed(xx: Tensor) -> Tensor:
        return embedder(xx)

    _e, jvp_val = torch.autograd.functional.jvp(_embed, (X,), (df_dx,), create_graph=True)
    num = (jvp_val * jvp_val).sum(dim=-1)  # ||J df_dx||^2 per row
    den = (df_dx * df_dx).sum(dim=-1)  # ||df_dx||^2 per row
    return torch.mean(torch.log(den + eps) - torch.log(num + eps))


__all__ = ["BandFeatureEmbedder", "local_target_consistency_loss"]
