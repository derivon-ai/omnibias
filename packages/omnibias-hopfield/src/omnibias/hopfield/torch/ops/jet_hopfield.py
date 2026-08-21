# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Jet-Hopfield retrieve (torch; theory 09-13).

Contact scores feed the existing ``softmax`` / ``logsumexp_value``
path. ``beta -> inf`` is temperature collapse (feasibility). Stored
jets may use founding bias collapse (``delta -> 0``). Do not
conflate the two.
"""

from __future__ import annotations

from omnibias.core.jet_hopfield import JetHopfieldConfig, contact_sq, honesty_payload
from omnibias.hopfield.torch.ops.hopfield import logsumexp_value, softmax
from torch import Tensor

DISCLAIMER = (
    "Jet-Hopfield stores germs and retrieves by contact; not vector "
    "Hopfield, not ImageNet, and not CCF stretch"
)


def jet_hopfield_retrieve(
    query_jet: Tensor,
    memory_jets: Tensor,
    *,
    config: JetHopfieldConfig | None = None,
) -> Tensor:
    cfg = JetHopfieldConfig() if config is None else config
    q = [float(v) for v in query_jet.reshape(-1).tolist()]
    bank = [[float(v) for v in row.reshape(-1).tolist()] for row in memory_jets]
    d2 = query_jet.new_tensor([contact_sq(q, mem, config=cfg) for mem in bank])
    scores = -d2
    _lse = logsumexp_value(scores, beta=cfg.beta)
    del _lse
    weights = softmax(scores, beta=cfg.beta, axis=-1)
    mem = memory_jets.to(dtype=query_jet.dtype)
    return (weights.unsqueeze(-1) * mem).sum(dim=0)


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
