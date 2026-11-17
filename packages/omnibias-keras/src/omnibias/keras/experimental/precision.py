# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Model-local precision adapters and an explicit typed training objective.

Native imports are confined to this adapter. No global backend configuration or
Keras implementation is changed, and variable reads stay inside Keras scopes.
"""

from __future__ import annotations

import inspect
import math
from typing import Any

import keras
from keras import ops


def validate_precision(precision: str) -> str:
    if precision not in {"source", "strict"}:
        raise ValueError("precision must be 'source' or 'strict'.")
    return precision


class StrictNumerics:
    """Keras-compatible operations which preserve the declared compute dtype.

Custom strict rules receive this interface as ``numerics``. Unmodified shape
and indexing operations delegate to Keras; numerical overrides use the selected
backend's public API. Call it on backend tensors inside a Layer/Operation call.
    """

    precision = "strict"
    native: Any

    def __init__(self, compute_dtype: str) -> None:
        self.compute_dtype = keras.backend.standardize_dtype(compute_dtype)
        if self.compute_dtype not in {"float32", "float64"}:
            raise TypeError("Strict numerical operations require float32/float64.")
        self.backend = keras.backend.backend()
        if self.backend == "torch":
            import torch
            self.native = torch
        elif self.backend == "jax":
            import jax
            import jax.numpy as jnp
            if self.compute_dtype == "float64" and not jax.config.x64_enabled:
                raise ValueError("Strict float64 requires JAX_ENABLE_X64=1 before importing Keras.")
            self.native = jnp
        elif self.backend == "tensorflow":
            import tensorflow as tf
            self.native = tf
        else:
            raise NotImplementedError(f"Strict precision is unsupported on {self.backend!r}.")
        probe = ops.convert_to_tensor(0.0, dtype=self.compute_dtype)
        if keras.backend.standardize_dtype(probe.dtype) != self.compute_dtype:
            raise ValueError("Backend cannot preserve the requested dtype; enable JAX_ENABLE_X64=1 before importing Keras.")

    def __getattr__(self, name: str) -> Any:
        return getattr(ops, name)

    def tensor(self, value: Any) -> Any:
        return ops.convert_to_tensor(value, dtype=self.compute_dtype)

    def cast(self, value: Any, dtype: str) -> Any:
        if isinstance(value, (bool, int, float, complex)):
            return ops.convert_to_tensor(value, dtype=dtype)
        return ops.cast(value, dtype)

    def check(self, value: Any) -> Any:
        dtype = keras.backend.standardize_dtype(value.dtype)
        if dtype != self.compute_dtype:
            raise TypeError(f"Strict operation returned {dtype}; expected {self.compute_dtype}.")
        return value

    def tanh(self, value: Any) -> Any:
        return self.native.tanh(self.tensor(value))

    def sin(self, value: Any) -> Any:
        return self.native.sin(self.tensor(value))

    def cos(self, value: Any) -> Any:
        return self.native.cos(self.tensor(value))

    def matmul(self, left: Any, right: Any) -> Any:
        left, right = self.tensor(left), self.tensor(right)
        return self.native.linalg.matmul(left, right) if self.backend == "tensorflow" else self.native.matmul(left, right)

    def mean(self, value: Any, axis: Any = None, keepdims: bool = False) -> Any:
        value = self.tensor(value)
        if self.backend == "torch":
            return self.native.mean(value, dim=axis, keepdim=keepdims)
        if self.backend == "tensorflow":
            return self.native.reduce_mean(value, axis=axis, keepdims=keepdims)
        return self.native.mean(value, axis=axis, keepdims=keepdims)

    def logaddexp(self, left: Any, right: Any) -> Any:
        left, right = self.tensor(left), self.tensor(right)
        if self.backend == "tensorflow":
            return ops.logaddexp(left, right)
        return self.native.logaddexp(left, right)

    def softplus(self, value: Any) -> Any:
        value = self.tensor(value)
        return self.logaddexp(ops.zeros_like(value), value)

    def conv(self, inputs: Any, kernel: Any, **kwargs: Any) -> Any:
        return ops.conv(self.tensor(inputs), self.tensor(kernel), **kwargs)

    def divide_no_nan(self, numerator: Any, denominator: Any) -> Any:
        numerator, denominator = self.tensor(numerator), self.tensor(denominator)
        nonzero = denominator != 0
        safe = ops.where(nonzero, denominator, ops.ones_like(denominator))
        return ops.where(nonzero, numerator / safe, ops.zeros_like(numerator))


def strict_source_contract(source: Any, dtype: str) -> None:
    """Reject authored policies rather than silently rewriting their meaning."""
    if isinstance(source, keras.layers.Layer) and not isinstance(source, keras.Model):
        if keras.backend.standardize_dtype(source.compute_dtype) != dtype:
            raise ValueError(f"{source.name}: strict precision requires layer compute dtype {dtype}, received {source.compute_dtype}.")
    if type(source).__name__ == "Cast" and getattr(source, "dtype", dtype) != dtype:
        raise ValueError(f"{source.name}: an explicit cast to {source.dtype} violates strict {dtype} precision.")


def _aligned(left: Any, right: Any) -> tuple[Any, Any]:
    if len(left.shape) == len(right.shape) + 1 and left.shape[-1] == 1:
        if len(right.shape) == 1:
            right = ops.expand_dims(right, axis=-1)
        else:
            left = ops.squeeze(left, axis=-1)
    elif len(right.shape) == len(left.shape) + 1 and right.shape[-1] == 1:
        if len(left.shape) == 1:
            left = ops.expand_dims(left, axis=-1)
        else:
            right = ops.squeeze(right, axis=-1)
    return left, right


@keras.saving.register_keras_serializable(package="omnibias")
class PrecisionMeanSquaredError(keras.losses.Loss):
    """Typed MSE with feature and weighted batch reductions in ``dtype``.

Use with a strict JetModel/OperatorModel to bypass Keras CompileLoss's implicit
floatx cast. Implicit masks are unsupported; supply explicit sample weights.
``mean_with_sample_weight`` divides by the sum of broadcast effective weights.
Empty/zero-weight reductions return zero. External losses are never replaced.
    """

    def __init__(self, reduction: str | None = "sum_over_batch_size",
                 name: str = "precision_mean_squared_error", dtype: str = "float64") -> None:
        super().__init__(name=name, reduction=reduction, dtype=dtype)
        self.numerics = StrictNumerics(self.dtype)

    def call(self, y_true: Any, y_pred: Any) -> Any:
        y_true, y_pred = _aligned(self.numerics.tensor(y_true), self.numerics.tensor(y_pred))
        error = (y_pred - y_true) ** 2
        if len(error.shape) == 0:
            raise ValueError("PrecisionMeanSquaredError expects at least one output dimension.")
        numerator = ops.sum(error, axis=-1)
        return self.numerics.divide_no_nan(numerator, ops.shape(error)[-1])

    def __call__(self, y_true: Any, y_pred: Any, sample_weight: Any = None) -> Any:
        if any(getattr(value, "_keras_mask", None) is not None for value in (y_true, y_pred)):
            raise NotImplementedError("PrecisionMeanSquaredError requires explicit sample weights instead of implicit masks.")
        values = self.call(y_true, y_pred)
        weights = None
        if sample_weight is not None:
            weights = self.numerics.tensor(sample_weight)
            if len(weights.shape) == len(values.shape) + 1 and weights.shape[-1] == 1:
                weights = ops.squeeze(weights, axis=-1)
            elif len(weights.shape) == len(values.shape) - 1:
                weights = ops.expand_dims(weights, axis=-1)
            weights = ops.broadcast_to(weights, ops.shape(values))
            values = values * weights
        if self.reduction in {None, "none"}:
            return self.numerics.check(values)
        numerator = ops.sum(values)
        if self.reduction == "sum":
            return self.numerics.check(numerator)
        denominator = (ops.sum(weights) if self.reduction == "mean_with_sample_weight" and weights is not None
                       else ops.size(values))
        return self.numerics.check(self.numerics.divide_no_nan(numerator, denominator))

    def get_config(self) -> dict[str, Any]:
        return {"name": self.name, "reduction": self.reduction, "dtype": self.dtype}


def _same_tree(reference: Any, value: Any, name: str) -> Any:
    try:
        keras.tree.assert_same_structure(reference, value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Strict {name} must exactly match the model output structure.") from error
    return value


class PrecisionObjectiveMixin:
    """Public compile/loss hooks shared by the two transformed model types."""

    def compile(self, *args: Any, **kwargs: Any) -> None:
        bound = inspect.signature(keras.Model.compile).bind(self, *args, **kwargs)
        specification = bound.arguments.get("loss")
        leaves = keras.tree.flatten(specification)
        recognized = getattr(self, "precision", "source") == "strict" and any(
            isinstance(loss, PrecisionMeanSquaredError) for loss in leaves)
        objective = None
        if recognized:
            if any(loss is not None and not isinstance(loss, PrecisionMeanSquaredError) for loss in leaves):
                raise ValueError("Strict precision losses cannot be mixed with external loss objects.")
            if self.losses:
                raise NotImplementedError("Strict precision objectives do not support additional regularization/add_loss terms.")
            output = self.graph.output
            loss_tree = (keras.tree.map_structure(lambda _: specification, output)
                         if isinstance(specification, PrecisionMeanSquaredError)
                         else _same_tree(output, specification, "loss"))
            weights = bound.arguments.get("loss_weights")
            if weights is None:
                weights = keras.tree.map_structure(lambda _: 1.0, output)
            elif isinstance(weights, (int, float)) and not keras.tree.is_nested(output):
                weights = float(weights)
            else:
                _same_tree(output, weights, "loss_weights")
            for loss, weight in zip(keras.tree.flatten(loss_tree), keras.tree.flatten(weights), strict=True):
                if loss is not None and loss.dtype != self.compute_dtype:
                    raise ValueError("Strict loss dtype must match the model compute dtype.")
                if loss is not None and loss.reduction in {None, "none"}:
                    raise ValueError("Compiled strict objectives require a scalar loss reduction.")
                if not isinstance(weight, (int, float)) or not math.isfinite(weight):
                    raise ValueError("Strict loss weights must be finite scalar numbers.")
            objective = (loss_tree, weights)
        super().compile(*args, **kwargs)
        self._precision_objective = objective

    def compute_loss(self, x: Any = None, y: Any = None, y_pred: Any = None,
                     sample_weight: Any = None, training: bool = True) -> Any:
        objective = getattr(self, "_precision_objective", None)
        if objective is None:
            return super().compute_loss(x, y, y_pred, sample_weight, training=training)
        if self.losses:
            raise NotImplementedError("Strict precision objectives do not support additional regularization/add_loss terms.")
        loss_tree, weights = objective
        _same_tree(y_pred, y, "targets")
        if keras.tree.is_nested(sample_weight):
            _same_tree(y_pred, sample_weight, "sample_weights")
            sample_tree = sample_weight
        else:
            sample_tree = keras.tree.map_structure(lambda _: sample_weight, y_pred)
        result = None
        for loss, weight, target, prediction, samples in zip(
            keras.tree.flatten(loss_tree), keras.tree.flatten(weights),
            keras.tree.flatten(y), keras.tree.flatten(y_pred), keras.tree.flatten(sample_tree), strict=True
        ):
            if loss is None:
                continue
            value = loss(target, prediction, samples) * loss.numerics.tensor(weight)
            result = value if result is None else result + value
        if result is None:
            raise ValueError("Strict objectives require at least one nonempty loss.")
        return result

    def get_compile_config(self) -> dict[str, Any]:
        return super().get_compile_config()

    def compile_from_config(self, config: dict[str, Any]) -> None:
        self.compile(**keras.saving.deserialize_keras_object(config))
        if getattr(self, "optimizer", None) is not None and self.built:
            self.optimizer.build(self.trainable_variables)
