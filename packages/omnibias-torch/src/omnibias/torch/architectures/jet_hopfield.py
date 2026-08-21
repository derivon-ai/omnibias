# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Jet-Hopfield (torch; theory 09-13).

Memories are germs. Retrieval is contact mismatch. ``beta -> inf``
is temperature collapse (feasibility). Stored jets may use founding
bias collapse (``delta -> 0``). Do not conflate the two.
"""

from __future__ import annotations

from collections.abc import Sequence

from omnibias.core import jet_hopfield as core

from torch import Tensor

DISCLAIMER = core.DISCLAIMER
JetHopfieldConfig = core.JetHopfieldConfig
honesty_payload = core.honesty_payload


def jet_hopfield_retrieve(
    query_jet: Tensor,
    memory_jets: Tensor | Sequence[Sequence[float]],
    *,
    config: JetHopfieldConfig | None = None,
) -> Tensor:
    q = [float(v) for v in query_jet.reshape(-1).tolist()]
    if isinstance(memory_jets, Tensor):
        bank = [[float(v) for v in row.reshape(-1).tolist()] for row in memory_jets]
    else:
        bank = [[float(v) for v in mem] for mem in memory_jets]
    germ = core.jet_hopfield_retrieve(q, bank, config=config)
    return query_jet.new_tensor(germ)


def worked_example() -> dict[str, float]:
    return core.worked_example()


__all__ = [
    "DISCLAIMER",
    "JetHopfieldConfig",
    "honesty_payload",
    "jet_hopfield_retrieve",
    "worked_example",
]
