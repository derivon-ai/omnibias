# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Jet-Hopfield retrieve (jax; theory 09-13).

Contact scores feed the existing ``softmax`` / ``logsumexp_value``
path. ``beta -> inf`` is temperature collapse (feasibility). Stored
jets may use founding bias collapse (``delta -> 0``). Do not
conflate the two.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from omnibias.core.jet_hopfield import JetHopfieldConfig, contact_sq, honesty_payload
from omnibias.hopfield.jax.ops.hopfield import logsumexp_value, softmax

DISCLAIMER = (
    "Jet-Hopfield stores germs and retrieves by contact; not vector "
    "Hopfield, not ImageNet, and not CCF stretch"
)


def jet_hopfield_retrieve(
    query_jet: Array,
    memory_jets: Array,
    *,
    config: JetHopfieldConfig | None = None,
) -> Array:
    cfg = JetHopfieldConfig() if config is None else config
    q_arr = jnp.asarray(query_jet)
    mem = jnp.asarray(memory_jets)
    q = [float(v) for v in q_arr.reshape(-1).tolist()]
    bank = [[float(v) for v in jnp.asarray(row).reshape(-1).tolist()] for row in mem]
    d2 = jnp.asarray([contact_sq(q, row, config=cfg) for row in bank], dtype=q_arr.dtype)
    scores = -d2
    _lse = logsumexp_value(scores, beta=cfg.beta)
    del _lse
    weights = softmax(scores, beta=cfg.beta, axis=-1)
    return (weights[..., None] * mem).sum(axis=0)


def worked_example() -> dict[str, float]:
    from omnibias.core.jet_hopfield import worked_example as core_example

    return core_example()


__all__ = [
    "DISCLAIMER",
    "JetHopfieldConfig",
    "honesty_payload",
    "jet_hopfield_retrieve",
    "worked_example",
]
