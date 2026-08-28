# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Functional JAX soft-binning embedding forward (bit-identical twin, theory 05-03).

``band_feature_embed(X, t, beta)`` reproduces
:meth:`omnibias.tab.torch.embed.BandFeatureEmbedder.forward` bit-for-bit (float64,
parity ``~1e-9``) given already-resolved thresholds ``t`` -- there is no jax *trainer*
for this or any other ``omnibias.tab`` model (matching the existing convention stated in
:mod:`omnibias.tab.jax.model`), so this module skips the torch side's monotone
gap-parametrization trick and simply takes the resolved ``(n_features, n_bins)``
threshold array directly, exactly as :func:`omnibias.tab.jax.model.forward_arrays` takes
raw parameter arrays rather than a training-time reparametrization.

Terminology: ``beta -> inf`` hardens each soft bin into a crisp histogram indicator --
temperature collapse (feasibility sense), not the founding ``delta -> 0`` bias collapse.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

_ROLES = ("band", "integral")


def band_feature_embed(X: Any, t: Any, beta: float, *, role: str = "band") -> Any:
    r"""Embed ``X`` of shape ``(..., n_features)`` into ``(..., n_features * (n_bins + 1))``.

    ``t``: ``(n_features, n_bins)`` strictly-increasing interior thresholds per feature.
    ``role="band"`` reuses the :func:`omnibias.core.band_embed.band_embed` window formula;
    ``role="integral"`` reuses its closed-form antiderivative twin
    (:func:`omnibias.core.band_embed.integral_embed`) -- both reimplemented with
    ``jax.nn.sigmoid`` / ``jax.nn.softplus`` (traceable / ``jit``-able), matching the torch
    twin's own native reimplementation rather than calling the numpy reference at runtime.
    """
    if role not in _ROLES:
        raise ValueError(f"role must be one of {_ROLES}, got {role!r}")
    Xv = jnp.asarray(X, dtype=jnp.float64)
    tv = jnp.asarray(t, dtype=jnp.float64)
    n_features, n_bins = tv.shape
    if Xv.shape[-1] != n_features:
        raise ValueError(f"X's last dim must equal t.shape[0]={n_features}, got {Xv.shape[-1]}")
    b = float(beta)
    z = b * (Xv[..., None] - tv)  # (..., n_features, n_bins)
    lead = z.shape[:-1]
    if role == "band":
        sig = jax.nn.sigmoid(z)
        ones = jnp.ones(lead + (1,), dtype=jnp.float64)
        zeros = jnp.zeros(lead + (1,), dtype=jnp.float64)
        padded = jnp.concatenate([ones, sig, zeros], axis=-1)  # (..., n_features, n_bins + 2)
        bins = padded[..., :-1] - padded[..., 1:]  # (..., n_features, n_bins + 1)
    else:
        sp = jax.nn.softplus(z)
        bin0 = (b * (Xv - tv[..., 0]) - sp[..., 0])[..., None]
        zeros = jnp.zeros(lead + (1,), dtype=jnp.float64)
        padded = jnp.concatenate([sp, zeros], axis=-1)  # (..., n_features, n_bins + 1)
        tail = padded[..., :-1] - padded[..., 1:]  # (..., n_features, n_bins)
        bins = jnp.concatenate([bin0, tail], axis=-1)  # (..., n_features, n_bins + 1)
    return bins.reshape(Xv.shape[:-1] + (n_features * (n_bins + 1),))


__all__ = ["band_feature_embed"]
