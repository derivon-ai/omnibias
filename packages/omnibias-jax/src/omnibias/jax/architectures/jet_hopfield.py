# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Jet-Hopfield (jax; theory 09-13).

Memories are germs. Retrieval is contact mismatch. ``beta -> inf``
is temperature collapse (feasibility). Stored jets may use founding
bias collapse (``delta -> 0``). Do not conflate the two.
"""

from __future__ import annotations

from collections.abc import Sequence

from omnibias.core import jet_hopfield as core

import jax.numpy as jnp
from jax import Array

DISCLAIMER = core.DISCLAIMER
JetHopfieldConfig = core.JetHopfieldConfig
honesty_payload = core.honesty_payload


def jet_hopfield_retrieve(
    query_jet: Array,
    memory_jets: Array | Sequence[Sequence[float]],
    *,
    config: JetHopfieldConfig | None = None,
) -> Array:
    arr = jnp.asarray(query_jet)
    q = [float(v) for v in arr.reshape(-1).tolist()]
    if isinstance(memory_jets, Array):
        bank = [[float(v) for v in jnp.asarray(row).reshape(-1).tolist()] for row in memory_jets]
    else:
        bank = [[float(v) for v in mem] for mem in memory_jets]
    germ = core.jet_hopfield_retrieve(q, bank, config=config)
    return jnp.asarray(germ, dtype=arr.dtype)


def worked_example() -> dict[str, float]:
    return core.worked_example()


__all__ = [
    "DISCLAIMER",
    "JetHopfieldConfig",
    "honesty_payload",
    "jet_hopfield_retrieve",
    "worked_example",
]
