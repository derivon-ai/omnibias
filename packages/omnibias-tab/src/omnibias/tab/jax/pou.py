# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Functional JAX TabPOU forward (band tokens + axis forest, theory 05-04).

No trainer -- matches :mod:`omnibias.tab.jax.model`. Bit-identical to the numpy
/ torch tokens+tree path at float64 (parity ``~1e-9``). ``beta -> inf`` on either
the embedder or the tree is temperature collapse, not founding bias collapse.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from omnibias.tab.jax.embed import band_feature_embed
from omnibias.tab.jax.model import forward_arrays


def pou_tokens(
    X: Any,
    t_embed: Any,
    beta_embed: float,
    *,
    role: str = "band",
    concat_raw: bool = True,
) -> Any:
    r"""Band (or integral) tokens, optionally concatenated with the raw scaled row."""
    Xv = jnp.asarray(X, dtype=jnp.float64)
    E = band_feature_embed(Xv, t_embed, float(beta_embed), role=role)
    if concat_raw:
        return jnp.concatenate([E, Xv], axis=-1)
    return E


def pou_forward_arrays(
    X: Any,
    t_embed: Any,
    beta_embed: float,
    W: Any,
    t: Any,
    leaves: Any,
    b0: Any,
    beta_tree: float,
    depth: int,
    *,
    role: str = "band",
    concat_raw: bool = True,
    use_embed: bool = True,
) -> Any:
    r"""Raw scores from scaled rows ``X`` plus optional band tokens and a forest."""
    Xv = jnp.asarray(X, dtype=jnp.float64)
    if use_embed:
        tok = pou_tokens(Xv, t_embed, beta_embed, role=role, concat_raw=concat_raw)
    else:
        tok = Xv
    return forward_arrays(W, t, leaves, b0, tok, float(beta_tree), int(depth))


__all__ = ["pou_forward_arrays", "pou_tokens"]
