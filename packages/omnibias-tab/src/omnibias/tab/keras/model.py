# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Keras 3 :class:`SoftTreeEnsemble` layer (``keras.ops``, not ``omnibias-keras``).

``call`` is the plugin API: ``(..., d) -> (..., k)``. Lives under
``omnibias.tab.keras`` so the keras core package stays core + ``keras.ops`` only.
"""

from __future__ import annotations

from typing import Any

from keras import initializers, layers, ops, saving
from omnibias.partition.keras.weights import prod_last_axis
from omnibias.tab._core.params import leaf_code_matrix


@saving.register_keras_serializable(package="omnibias.tab")
class SoftTreeEnsemble(layers.Layer):
    """Oblivious soft-tree ensemble as a Keras 3 layer."""

    def __init__(
        self,
        n_features: int,
        n_trees: int = 4,
        depth: int = 2,
        n_outputs: int = 1,
        *,
        beta: float = 1.0,
        task: str = "binary",
        learnable_beta: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.n_features = int(n_features)
        self.n_trees = int(n_trees)
        self.depth = int(depth)
        self.n_outputs = int(n_outputs)
        self.n_leaves = 1 << self.depth
        self.task = str(task)
        self.beta_init = float(beta)
        self.learnable_beta = bool(learnable_beta)
        self.W = self.add_weight(
            name="W",
            shape=(self.n_trees, self.depth, self.n_features),
            initializer="random_normal",
            trainable=True,
        )
        self.t = self.add_weight(
            name="t",
            shape=(self.n_trees, self.depth),
            initializer="zeros",
            trainable=True,
        )
        self.leaves = self.add_weight(
            name="leaves",
            shape=(self.n_trees, self.n_leaves, self.n_outputs),
            initializer="zeros",
            trainable=True,
        )
        self.b0 = self.add_weight(
            name="b0",
            shape=(self.n_outputs,),
            initializer="zeros",
            trainable=True,
        )
        self.beta = self.add_weight(
            name="beta",
            shape=(),
            initializer=initializers.Constant(self.beta_init),
            trainable=self.learnable_beta,
        )

    def build(self, input_shape: Any) -> None:
        super().build(input_shape)

    def call(self, X: Any) -> Any:
        rows = ops.reshape(X, (-1, self.n_features))
        codes = ops.convert_to_tensor(leaf_code_matrix(self.depth), dtype=X.dtype)
        z = ops.einsum("nd,mjd->nmj", rows, self.W) - ops.expand_dims(self.t, 0)
        g = ops.sigmoid(self.beta * z)
        gexp = ops.expand_dims(g, 2)
        bexp = ops.reshape(codes, (1, 1, self.n_leaves, self.depth))
        factors = bexp * gexp + (1.0 - bexp) * (1.0 - gexp)
        memberships = prod_last_axis(factors, self.depth)
        out = ops.einsum("nml,mlk->nk", memberships, self.leaves) + self.b0
        return ops.reshape(out, ops.shape(X)[:-1] + (self.n_outputs,))

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "n_features": self.n_features,
                "n_trees": self.n_trees,
                "depth": self.depth,
                "n_outputs": self.n_outputs,
                "beta": float(self.beta_init),
                "task": self.task,
                "learnable_beta": self.learnable_beta,
            }
        )
        return config


__all__ = ["SoftTreeEnsemble"]
