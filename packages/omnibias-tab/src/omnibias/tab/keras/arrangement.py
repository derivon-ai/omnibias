# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Keras 3 :class:`ArrangementClassifier` layer (``keras.ops``)."""

from __future__ import annotations

from typing import Any

from keras import initializers, layers, ops, saving
from omnibias.partition._core.params import region_code_matrix
from omnibias.partition.keras.weights import combine, partition_weights_arrays, prod_last_axis


@saving.register_keras_serializable(package="omnibias.tab")
class ArrangementClassifier(layers.Layer):
    """``H`` hyperplanes, soft cell membership, per-cell logits as a Keras layer."""

    def __init__(
        self,
        n_features: int,
        n_hyperplanes: int = 2,
        *,
        beta: float = 1.0,
        n_outputs: int = 1,
        task: str = "binary",
        learnable_beta: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.n_features = int(n_features)
        self.n_hyperplanes = int(n_hyperplanes)
        self.n_cells = 1 << self.n_hyperplanes
        self.n_outputs = int(n_outputs)
        self.task = str(task)
        self.beta_init = float(beta)
        self.learnable_beta = bool(learnable_beta)
        self.W = self.add_weight(
            name="W",
            shape=(self.n_hyperplanes, self.n_features),
            initializer="random_normal",
            trainable=True,
        )
        self.t = self.add_weight(
            name="t",
            shape=(self.n_hyperplanes,),
            initializer="zeros",
            trainable=True,
        )
        self.cell_logits = self.add_weight(
            name="cell_logits",
            shape=(self.n_cells, self.n_outputs),
            initializer="zeros",
            trainable=True,
        )
        self.beta = self.add_weight(
            name="beta",
            shape=(),
            initializer=initializers.Constant(self.beta_init),
            trainable=self.learnable_beta,
        )

    def call(self, X: Any) -> Any:
        rows = ops.reshape(X, (-1, self.n_features))
        weights = partition_weights_arrays(
            self.W, self.t, rows, self.beta, self.n_hyperplanes
        )
        cell = self.cell_logits
        logits = ops.broadcast_to(
            ops.expand_dims(cell, 0),
            (ops.shape(rows)[0], self.n_cells, self.n_outputs),
        )
        out = combine(weights, logits)
        return ops.reshape(out, ops.shape(X)[:-1] + (self.n_outputs,))

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "n_features": self.n_features,
                "n_hyperplanes": self.n_hyperplanes,
                "beta": float(self.beta_init),
                "n_outputs": self.n_outputs,
                "task": self.task,
                "learnable_beta": self.learnable_beta,
            }
        )
        return config


def _batched_arrangement_logits(
    W: Any,
    t: Any,
    cell_logits: Any,
    X: Any,
    beta: Any,
    n_hyperplanes: int,
) -> Any:
    """Soft arrangement logits ``(n, M, k)`` from stacked members (keras.ops)."""
    n_leaves = 1 << int(n_hyperplanes)
    codes = ops.convert_to_tensor(region_code_matrix(int(n_hyperplanes)), dtype=X.dtype)
    z = ops.einsum("nd,mhd->nmh", X, W) - ops.expand_dims(t, 0)
    b = ops.cast(beta, X.dtype)
    if ops.ndim(b) == 0:
        g = ops.sigmoid(b * z)
    else:
        g = ops.sigmoid(ops.reshape(b, (1, -1, 1)) * z)
    gexp = ops.expand_dims(g, 2)
    bexp = ops.reshape(codes, (1, 1, n_leaves, n_hyperplanes))
    factors = bexp * gexp + (1.0 - bexp) * (1.0 - gexp)
    weights = prod_last_axis(factors, int(n_hyperplanes))
    return ops.einsum("nml,mlk->nmk", weights, cell_logits)


@saving.register_keras_serializable(package="omnibias.tab")
class ArrangementBoosted(layers.Layer):
    """Additive ensemble of :class:`ArrangementClassifier` (one ``keras.ops`` combine).

    ``call`` is ``base + lr * sum_m member_m(X)`` with stacked ``W/t/cell/beta``,
    logits ``(..., k)``. Mirrors the torch :class:`omnibias.tab.torch.arrangement.ArrangementBoosted`
    batched kernel.
    """

    def __init__(
        self,
        members: list[ArrangementClassifier] | None = None,
        *,
        n_features: int | None = None,
        n_hyperplanes: int = 2,
        n_members: int = 2,
        n_outputs: int = 1,
        beta: float = 1.0,
        learning_rate: float = 0.3,
        base: float = 0.0,
        learnable_beta: bool = False,
        task: str = "binary",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        packed = list(members) if members is not None else []
        if not packed:
            if n_features is None:
                raise ValueError("n_features is required when members is omitted")
            member_kwargs: dict[str, Any] = {}
            if kwargs.get("dtype") is not None:
                member_kwargs["dtype"] = kwargs["dtype"]
            packed = [
                ArrangementClassifier(
                    int(n_features),
                    int(n_hyperplanes),
                    beta=float(beta),
                    n_outputs=int(n_outputs),
                    task=str(task),
                    learnable_beta=bool(learnable_beta),
                    **member_kwargs,
                )
                for _ in range(max(1, int(n_members)))
            ]
        self._n_members = len(packed)
        first = packed[0]
        self.n_features = int(first.n_features)
        self.n_hyperplanes = int(first.n_hyperplanes)
        self.n_outputs = int(first.n_outputs)
        self.task = str(first.task)
        self.learning_rate_init = float(learning_rate)
        self.base_init = float(base)
        for i, member in enumerate(packed):
            setattr(self, f"member_{i}", member)
        self.learning_rate = self.add_weight(
            name="learning_rate",
            shape=(),
            initializer=initializers.Constant(self.learning_rate_init),
            trainable=False,
        )
        self.base = self.add_weight(
            name="base",
            shape=(),
            initializer=initializers.Constant(self.base_init),
            trainable=False,
        )

    @property
    def members(self) -> list[ArrangementClassifier]:
        return [getattr(self, f"member_{i}") for i in range(self._n_members)]

    def build(self, input_shape: Any) -> None:
        for member in self.members:
            member.build(input_shape)
        super().build(input_shape)

    def call(self, X: Any) -> Any:
        rows = ops.reshape(X, (-1, self.n_features))
        packed = self.members
        if not packed:
            zeros = ops.zeros((ops.shape(rows)[0], self.n_outputs), dtype=X.dtype)
            out = zeros + ops.cast(self.base, X.dtype)
            return ops.reshape(out, ops.shape(X)[:-1] + (self.n_outputs,))
        W = ops.stack([m.W for m in packed], axis=0)
        t = ops.stack([m.t for m in packed], axis=0)
        cell = ops.stack([m.cell_logits for m in packed], axis=0)
        betas = ops.stack([ops.reshape(m.beta, ()) for m in packed], axis=0)
        contrib = _batched_arrangement_logits(
            W, t, cell, rows, betas, self.n_hyperplanes
        )
        lr = ops.cast(self.learning_rate, X.dtype)
        base = ops.cast(self.base, X.dtype)
        out = base + lr * ops.sum(contrib, axis=1)
        return ops.reshape(out, ops.shape(X)[:-1] + (self.n_outputs,))

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "members": [saving.serialize_keras_object(m) for m in self.members],
                "learning_rate": float(self.learning_rate_init),
                "base": float(self.base_init),
            }
        )
        return config

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> ArrangementBoosted:
        members_cfg = config.pop("members")
        members = [saving.deserialize_keras_object(c) for c in members_cfg]
        return cls(members, **config)


__all__ = ["ArrangementBoosted", "ArrangementClassifier"]
