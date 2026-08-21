# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Logit gate over :class:`~omnibias.core.proof.observe.Observation` features.

The gate is a proposer. It never writes
:class:`~omnibias.core.proof.discovery.ExactCheck`. If it ranks SOS first and
SOS does not snap, the class loop tries the next sort.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from omnibias.core.proof.condition import ALL_CONDITION_SORTS
from omnibias.core.proof.observe import FEATURE_DIM, Observation

FeatureRow = Sequence[int]


def logit_gate_logits_numpy(
    weights: Sequence[Sequence[float]],
    bias: Sequence[float],
    features: FeatureRow,
) -> tuple[float, ...]:
    import numpy as np

    matrix = np.asarray(weights, dtype=float)
    offset = np.asarray(bias, dtype=float)
    vector = np.asarray(list(features), dtype=float)
    return tuple((matrix @ vector + offset).tolist())


def logit_gate_logits_torch(
    weights: Sequence[Sequence[float]],
    bias: Sequence[float],
    features: FeatureRow,
) -> tuple[float, ...]:
    import torch

    dtype = torch.get_default_dtype()
    matrix = torch.tensor(weights, dtype=dtype)
    offset = torch.tensor(bias, dtype=dtype)
    vector = torch.tensor(list(features), dtype=dtype)
    return tuple((matrix @ vector + offset).detach().tolist())


def logit_gate_logits_jax(
    weights: Sequence[Sequence[float]],
    bias: Sequence[float],
    features: FeatureRow,
) -> tuple[float, ...]:
    import jax.numpy as jnp

    matrix = jnp.asarray(weights)
    offset = jnp.asarray(bias)
    vector = jnp.asarray(list(features))
    return tuple(jnp.asarray(matrix @ vector + offset).tolist())


class LogitGate:
    """Linear logits on :meth:`Observation.features`. Softmax is ranking only."""

    def __init__(self, sorts: Sequence[str] | None = None) -> None:
        self.sorts = tuple(sorts) if sorts is not None else ALL_CONDITION_SORTS
        n_features = FEATURE_DIM
        self.weights: list[list[float]] = [
            [0.0] * n_features for _ in range(len(self.sorts))
        ]
        self.bias: list[float] = [0.0] * len(self.sorts)

    def train(
        self,
        pairs: Sequence[tuple[Observation | FeatureRow, str]],
    ) -> None:
        """Least-squares one-hot fit. Does not emit a certificate."""

        import numpy as np

        rows: list[list[float]] = []
        labels: list[list[float]] = []
        index = {sort: i for i, sort in enumerate(self.sorts)}
        for feat, sort in pairs:
            if isinstance(feat, Observation):
                row = [float(value) for value in feat.features()]
            else:
                row = [float(value) for value in feat]
            rows.append(row)
            onehot = [0.0] * len(self.sorts)
            if sort in index:
                onehot[index[sort]] = 1.0
            labels.append(onehot)
        design = np.asarray(rows, dtype=float)
        targets = np.asarray(labels, dtype=float)
        augmented = np.concatenate([design, np.ones((design.shape[0], 1))], axis=1)
        solution, *_ = np.linalg.lstsq(augmented, targets, rcond=None)
        self.weights = solution[:-1].T.tolist()
        self.bias = solution[-1].tolist()

    def logits(self, observation: Observation) -> tuple[float, ...]:
        return logit_gate_logits_numpy(self.weights, self.bias, observation.features())

    def propose(self, observation: Observation) -> tuple[str, ...]:
        scores = self.logits(observation)
        ranked = sorted(
            range(len(self.sorts)),
            key=lambda i: (-scores[i], self.sorts[i]),
        )
        return tuple(self.sorts[i] for i in ranked)

    def as_dict(self) -> dict[str, Any]:
        return {
            "sorts": list(self.sorts),
            "weights": self.weights,
            "bias": self.bias,
        }


__all__ = [
    "LogitGate",
    "logit_gate_logits_jax",
    "logit_gate_logits_numpy",
    "logit_gate_logits_torch",
]
