# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Jet-Hopfield memory (theory 09-13).

Memories are germs (value plus derivatives). Retrieval scores a
contact mismatch, not a vector inner product. Softmax of
``-beta d^2`` is the existing modern-Hopfield kernel.

``beta -> inf`` is temperature collapse (hard nearest germ,
feasibility of a 0/1 assignment) and is labelled, not the default.
Stored profile jets may come from founding bias collapse
(``delta -> 0``) of an OMBU dictionary. Do not conflate the two.

Not a rewrite of vector Hopfield. Not ImageNet retrieval. Not CCF
stretch.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

DISCLAIMER = (
    "Jet-Hopfield stores germs and retrieves by contact; not vector "
    "Hopfield, not ImageNet, and not CCF stretch"
)

Jet = Sequence[float]
MemoryBank = Sequence[Jet]


def honesty_payload(*, beta: float = 1.0) -> dict[str, bool]:
    return {
        "temperature_collapse_used": math.isinf(float(beta)),
        "imagenet_claim": False,
        "stretch_claim": False,
        "theorem_prover_verified": False,
    }


@dataclass(frozen=True)
class JetHopfieldConfig:
    jet_order: int = 1
    lam: float = 1.0
    beta: float = 1.0


DEFAULT_CONFIG = JetHopfieldConfig()


def _check_config(config: JetHopfieldConfig) -> JetHopfieldConfig:
    if config.jet_order < 0:
        raise ValueError(f"jet_order must be >= 0, got {config.jet_order}")
    if config.lam < 0.0:
        raise ValueError(f"lam must be >= 0, got {config.lam}")
    if config.beta <= 0.0 and not math.isinf(config.beta):
        raise ValueError(f"beta must be > 0 or +inf, got {config.beta}")
    return config


def _as_jet(jet: Jet, order: int, *, name: str) -> tuple[float, ...]:
    if len(jet) < order + 1:
        raise ValueError(f"{name} jet must have length >= {order + 1}, got {len(jet)}")
    return tuple(float(v) for v in jet[: order + 1])


def contact_sq(query: Jet, memory: Jet, *, config: JetHopfieldConfig | None = None) -> float:
    """``|u-u_mu|^2 + lam sum_k |u^{(k)}-u_mu^{(k)}|^2``."""
    cfg = DEFAULT_CONFIG if config is None else _check_config(config)
    q = _as_jet(query, cfg.jet_order, name="query")
    m = _as_jet(memory, cfg.jet_order, name="memory")
    acc = (q[0] - m[0]) ** 2
    for k in range(1, cfg.jet_order + 1):
        acc += cfg.lam * (q[k] - m[k]) ** 2
    return acc


def _stable_softmax(logits: Sequence[float]) -> tuple[float, ...]:
    top = max(logits)
    exps = [math.exp(v - top) for v in logits]
    total = sum(exps)
    return tuple(v / total for v in exps)


def jet_hopfield_weights(
    query: Jet,
    memories: MemoryBank,
    *,
    config: JetHopfieldConfig | None = None,
) -> tuple[float, ...]:
    """Softmax masses ``softmax(-beta d^2)``. Infinite ``beta`` is one-hot."""
    cfg = DEFAULT_CONFIG if config is None else _check_config(config)
    if not memories:
        raise ValueError("memory bank must be non-empty")
    dists = [contact_sq(query, mem, config=cfg) for mem in memories]
    if math.isinf(cfg.beta):
        best = min(dists)
        return tuple(1.0 if d == best else 0.0 for d in dists)
    logits = [-cfg.beta * d for d in dists]
    return _stable_softmax(logits)


def jet_hopfield_retrieve(
    query: Jet,
    memories: MemoryBank,
    *,
    config: JetHopfieldConfig | None = None,
) -> tuple[float, ...]:
    """Retrieve a germ as the softmax mixture of stored jets."""
    cfg = DEFAULT_CONFIG if config is None else _check_config(config)
    weights = jet_hopfield_weights(query, memories, config=cfg)
    order = cfg.jet_order
    bank = [_as_jet(mem, order, name="memory") for mem in memories]
    retrieved = [0.0] * (order + 1)
    for weight, mem in zip(weights, bank, strict=True):
        for k in range(order + 1):
            retrieved[k] += weight * mem[k]
    return tuple(retrieved)


def nearest_index(
    query: Jet,
    memories: MemoryBank,
    *,
    config: JetHopfieldConfig | None = None,
) -> int:
    cfg = DEFAULT_CONFIG if config is None else _check_config(config)
    dists = [contact_sq(query, mem, config=cfg) for mem in memories]
    best = min(dists)
    return dists.index(best)


def worked_example() -> dict[str, float]:
    """Spec 09-13: query near ``J1=(1,0)`` with ``lam=1``, ``beta=10``."""
    cfg = JetHopfieldConfig(jet_order=1, lam=1.0, beta=10.0)
    memories: tuple[tuple[float, float], ...] = ((1.0, 0.0), (0.0, 1.0))
    query = (1.0, 0.01)
    germ = jet_hopfield_retrieve(query, memories, config=cfg)
    d1 = contact_sq(query, memories[0], config=cfg)
    d2 = contact_sq(query, memories[1], config=cfg)
    return {
        "retrieved_value": germ[0],
        "retrieved_deriv": germ[1],
        "d2_j1": d1,
        "d2_j2": d2,
        "value_err": abs(germ[0] - 1.0),
    }


CONTACT_MEMORIES: tuple[tuple[float, float], ...] = (
    (1.0, 0.0),
    (0.0, 1.0),
    (0.5, 0.0),
)
CONTACT_QUERY = (0.6, 1.0)
VALUE_NEAREST = 2
CONTACT_NEAREST = 1


def contact_split(
    *,
    seeds: int = 5,
    noise: float = 1e-3,
    lam: float = 1.0,
    beta: float = 10.0,
) -> dict[str, object]:
    """G2: contact retrieve vs value-only retrieve on a three-memory probe."""
    jet_cfg = JetHopfieldConfig(jet_order=1, lam=lam, beta=beta)
    value_cfg = JetHopfieldConfig(jet_order=1, lam=0.0, beta=beta)
    jet_idx: list[int] = []
    value_idx: list[int] = []
    for seed in range(seeds):
        shift = noise * float(seed - 2)
        query = (CONTACT_QUERY[0] + shift, CONTACT_QUERY[1] + shift)
        jet_idx.append(nearest_index(query, CONTACT_MEMORIES, config=jet_cfg))
        value_idx.append(nearest_index(query, CONTACT_MEMORIES, config=value_cfg))
    return {
        "lam": lam,
        "beta": beta,
        "jet_indices": jet_idx,
        "value_indices": value_idx,
        "jet_contact": all(i == CONTACT_NEAREST for i in jet_idx),
        "value_other": all(i == VALUE_NEAREST for i in value_idx),
        "split": all(i == CONTACT_NEAREST for i in jet_idx)
        and all(j == VALUE_NEAREST for j in value_idx),
    }


__all__ = [
    "CONTACT_MEMORIES",
    "CONTACT_NEAREST",
    "CONTACT_QUERY",
    "DEFAULT_CONFIG",
    "DISCLAIMER",
    "JetHopfieldConfig",
    "VALUE_NEAREST",
    "contact_split",
    "contact_sq",
    "honesty_payload",
    "jet_hopfield_retrieve",
    "jet_hopfield_weights",
    "nearest_index",
    "worked_example",
]
