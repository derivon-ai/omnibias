# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Lift Functional graphs through public Keras cloning and ordinary ops."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from operator import index as integer_index
from typing import Any

from omnibias.core.jet_algebra import JetSpec
from omnibias.core.jet_series import activation, activation_derivative, multiply, reciprocal
from omnibias.keras.experimental.precision import (
    PrecisionObjectiveMixin,
    StrictNumerics,
    strict_source_contract,
    validate_precision,
)

import keras
from keras import ops

_CUSTOM_RULES: dict[type, Callable[..., Any]] = {}
_STRICT_RULES: dict[type, Callable[..., Any]] = {}
_SMOOTH = {
    "tanh", "sigmoid", "exp", "sin", "cos", "log", "sqrt", "reciprocal",
    "gaussian", "gaussian_normalized", "gaussian_rbf", "erf", "erfc",
    "gelu", "gelu_approx", "gelu_tanh", "softplus", "silu", "swish",
}
_ELEMENTWISE = {
    "Add", "Subtract", "Multiply", "Divide", "TrueDivide", "Negative",
    "Matmul", "Power", "Square",
}
_LINEAR = {
    "Reshape", "Transpose", "ExpandDims", "Squeeze", "GetItem", "Slice",
    "Concatenate", "Stack", "Sum", "Mean", "Cast", "Copy", "Identity",
    "BroadcastTo", "Repeat", "Tile", "Take", "Flip", "Moveaxis",
}


def register_jet_rule(operation_type: type, rule: Callable[..., Any], *,
                      strict_rule: Callable[..., Any] | None = None) -> None:
    """Register ``rule(source, coefficient_args, coefficient_kwargs, spec)``.

    Tensor arguments are coefficient lists; non-tensor arguments retain their
    original structure. The rule returns a coefficient list. Rules must use
    portable Keras ops, preserve parameter gradients, and be installed again
    before loading a saved model that uses their operation type. A strict rule
    receives the same four arguments plus keyword ``numerics``; it must use
    that numerical adapter, include the primal and support order zero.
    """
    if not isinstance(operation_type, type) or not callable(rule):
        raise TypeError("A Keras operation type and callable jet rule are required.")
    if strict_rule is not None and not callable(strict_rule):
        raise TypeError("strict_rule must be callable or None.")
    _CUSTOM_RULES[operation_type] = rule
    _STRICT_RULES.pop(operation_type, None)
    if strict_rule is not None:
        _STRICT_RULES[operation_type] = strict_rule


def _dtype(value: Any) -> str:
    result = keras.backend.standardize_dtype(value)
    if result not in {"float32", "float64"}:
        raise TypeError(f"Jet models support float32/float64, received {result}.")
    return result


def _check_shape(spec: JetSpec, shape: Any, dtype: Any) -> None:
    # Symbolic construction knows no batch size. Check the known dimensions
    # there, and the full concrete tensor again when the model is executed.
    dimensions = []
    for dimension in shape:
        try:
            dimensions.append(integer_index(dimension))
        except TypeError:
            dimensions.append(1)
    spec.check_tensor(dimensions, 8 if _dtype(dtype) == "float64" else 4)


def seed(value: Any, directions: Any, *, spec: JetSpec) -> Any:
    """Seed an affine spatial path, packing its normalized coefficients at axis 1.

    For a directional spec, ``directions`` has the shape of ``value``. For a
    mixed spec, provide one same-shaped direction per formal variable.
    """
    value = ops.convert_to_tensor(value)
    _dtype(value.dtype)
    if len(value.shape) < 1:
        raise ValueError("Keras jet values require a batch dimension.")
    vectors = [directions] if spec.mode == "directional" else list(directions)
    if len(vectors) != spec.dim:
        raise ValueError(f"Expected {spec.dim} seed directions.")
    vectors = [ops.convert_to_tensor(v) for v in vectors]
    for vector in vectors:
        if _dtype(vector.dtype) != _dtype(value.dtype):
            raise TypeError("Every seed direction must match the value dtype.")
        if tuple(vector.shape) != tuple(value.shape):
            raise ValueError("Every seed direction must match the value shape.")
    _check_shape(spec, value.shape, value.dtype)
    coefficients = [value]
    for alpha in spec.indices[1:]:
        coefficients.append(vectors[alpha.index(1)] if sum(alpha) == 1 else ops.zeros_like(value))
    return ops.stack(coefficients, axis=1)


def extract(value: Any, alpha: int | Sequence[int], *, spec: JetSpec,
            derivative: bool = False) -> Any:
    """Extract a coefficient, optionally multiplying by the multi-index factorial."""
    index = spec.index((alpha,) if isinstance(alpha, int) else tuple(alpha))
    result = value[:, index]
    return result * float(spec.factorials[index]) if derivative else result


@dataclass
class _Series:
    values: list[Any]
    primal_dtype: str | None = None


def _primal(series: _Series) -> Any:
    value = series.values[0]
    return ops.cast(value, series.primal_dtype) if series.primal_dtype is not None else value


def _map(value: Any, fn: Callable[[Any], Any]) -> Any:
    if isinstance(value, _Series):
        return fn(value)
    if isinstance(value, tuple):
        return tuple(_map(v, fn) for v in value)
    if isinstance(value, list):
        return [_map(v, fn) for v in value]
    if isinstance(value, dict):
        return {k: _map(v, fn) for k, v in value.items()}
    return value


def _encode(value: Any, tensors: list[Any], constants: list[Any],
            primal_dtypes: dict[int, str]) -> Any:
    if isinstance(value, keras.KerasTensor):
        tensors.append(value)
        return {"jet": len(tensors) - 1, "primal_dtype": primal_dtypes.get(id(value), value.dtype)}
    if isinstance(value, slice):
        return {"slice": [value.start, value.stop, value.step]}
    if value is Ellipsis:
        return {"ellipsis": True}
    if isinstance(value, tuple):
        return {"tuple": [_encode(v, tensors, constants, primal_dtypes) for v in value]}
    if isinstance(value, list):
        return {"list": [_encode(v, tensors, constants, primal_dtypes) for v in value]}
    if isinstance(value, dict):
        return {"dict": {k: _encode(v, tensors, constants, primal_dtypes) for k, v in value.items()}}
    if ops.is_tensor(value) or isinstance(value, keras.Variable) or (
        hasattr(value, "dtype") and hasattr(value, "shape")
    ):
        if isinstance(value, keras.Variable) and value.trainable:
            raise NotImplementedError(
                "Standalone trainable variable operands require a registered layer jet rule."
            )
        constants.append(value)
        return {"constant": len(constants) - 1}
    return {"literal": value}


def _decode(template: Any, inputs: list[Any], constants: list[Any], size: int) -> Any:
    if "jet" in template:
        tensor = inputs[template["jet"]]
        return _Series([tensor[:, k] for k in range(size)], template.get("primal_dtype"))
    if "constant" in template:
        return constants[template["constant"]]
    if "slice" in template:
        return slice(*template["slice"])
    if "ellipsis" in template:
        return Ellipsis
    if "tuple" in template:
        return tuple(_decode(v, inputs, constants, size) for v in template["tuple"])
    if "list" in template:
        return [_decode(v, inputs, constants, size) for v in template["list"]]
    if "dict" in template:
        return {k: _decode(v, inputs, constants, size) for k, v in template["dict"].items()}
    return template["literal"]


def _apply_input_policy(template: Any, dtype: str) -> None:
    """Canonicalize the cast which a source Layer.__call__ already performs."""
    if isinstance(template, dict):
        if "jet" in template:
            template["primal_dtype"] = dtype
        else:
            for value in template.values():
                _apply_input_policy(value, dtype)
    elif isinstance(template, (list, tuple)):
        for value in template:
            _apply_input_policy(value, dtype)


def _series(value: Any, size: int) -> list[Any]:
    return value.values if isinstance(value, _Series) else [value] + [0.0] * (size - 1)


def _activation_name(fn: Any) -> str:
    if fn is None:
        return "linear"
    name = keras.activations.serialize(fn)
    if name == "exponential":
        name = "exp"
    if not isinstance(name, str):
        raise NotImplementedError("Custom activation requires a registered jet rule.")
    if name == "linear" or name in _SMOOTH or name == "softmax":
        return name
    raise NotImplementedError(f"Unsupported smooth activation {name!r}.")


def _softmax(values: list[Any], spec: JetSpec, axis: Any, numerics: Any = None) -> list[Any]:
    ops = numerics or keras.ops
    axes = (axis,) if isinstance(axis, int) else tuple(axis)
    rank = len(values[0].shape)
    if any(a % rank == 0 for a in axes):
        raise NotImplementedError("Softmax across the batch dimension is unsupported.")
    shift = ops.stop_gradient(ops.max(values[0], axis=axis, keepdims=True))
    shifted = [values[0] - shift, *values[1:]]
    numerator = activation(shifted, spec, "exp", ops)
    denominator = [ops.sum(v, axis=axis, keepdims=True) for v in numerator]
    return multiply(numerator, reciprocal(denominator, spec, ops), spec, ops)


def _nonlinear(values: list[Any], spec: JetSpec, name: str, axis: Any = -1,
               backward: str = "native", numerics: Any = None) -> list[Any]:
    ops = numerics or keras.ops
    strict = isinstance(ops, StrictNumerics)
    if name == "linear":
        return values
    if name == "softmax":
        return _softmax(values, spec, axis, numerics=ops)
    if not strict and name in {"gelu", "gelu_approx", "gelu_tanh"} and keras.backend.backend() == "tensorflow":
        # tf.nn.gelu explicitly casts these Python constants through float32,
        # including for float64 inputs. Lift its actual numerical expression.
        if name == "gelu":
            scale = ops.cast(ops.convert_to_tensor(0.7071067811865476, dtype="float32"), values[0].dtype)
            cdf = _nonlinear([-scale * v for v in values], spec, "erfc", backward=backward)
        else:
            cubic_scale = ops.cast(ops.convert_to_tensor(0.044715, dtype="float32"), values[0].dtype)
            cube = multiply(multiply(values, values, spec, ops), values, spec, ops)
            argument = [0.7978845608028654 * (v + cubic_scale * c)
                        for v, c in zip(values, cube, strict=True)]
            cdf = _nonlinear(argument, spec, "tanh", backward=backward)
            cdf[0] = 1 + cdf[0]
        return [0.5 * value for value in multiply(values, cdf, spec, ops)]
    if backward == "analytic":
        @ops.custom_gradient
        def compose(packed: Any) -> Any:
            inputs = [packed[:, k] for k in range(spec.size)]
            output = ops.stack(activation(inputs, spec, name, ops), axis=1)

            def grad(*args: Any, upstream: Any = None) -> Any:
                if upstream is None:
                    (upstream,) = args
                    grad_inputs = inputs
                else:
                    # Torch passes the saved primal input explicitly. Re-slice
                    # it in backward so higher parameter derivatives stay linked.
                    grad_inputs = [args[0][:, k] for k in range(spec.size)]
                derivative = activation_derivative(grad_inputs, spec, name, ops)
                result = [ops.zeros_like(grad_inputs[0]) for _ in range(spec.size)]
                for out_index, pairs in enumerate(spec.pairs):
                    for left, right in pairs:
                        result[left] = result[left] + upstream[:, out_index] * derivative[right]
                return ops.stack(result, axis=1)

            return output, grad

        packed = compose(ops.stack(values, axis=1))
        result = [packed[:, k] for k in range(spec.size)]
    else:
        result = activation(values, spec, name, ops)
    # Keras' Torch adapter calls thresholded torch.nn.functional.softplus.
    # Respect that source primitive's identity branch, including curved paths.
    if not strict and name == "softplus" and keras.backend.backend() == "torch":
        result = [ops.where(values[0] > 20, incoming, outgoing)
                  for incoming, outgoing in zip(values, result, strict=True)]
    return result


def _merge_coefficient(value: Any, k: int) -> Any:
    if isinstance(value, _Series):
        return value.values[k]
    if isinstance(value, list):
        return [_merge_coefficient(v, k) for v in value]
    if isinstance(value, tuple):
        return tuple(_merge_coefficient(v, k) for v in value)
    return value if k == 0 else ops.zeros_like(value)


def _reject_batch_change(source: Any, args: Any) -> None:
    name = type(source).__name__
    first = args[0]
    if isinstance(first, (list, tuple)) and first:
        first = first[0]
    if not isinstance(first, _Series):
        return
    rank = len(first.values[0].shape)
    if name in {"Sum", "Mean", "Squeeze", "Softmax", "Take", "Repeat", "Flip", "Concatenate"}:
        axis = getattr(source, "axis", None)
        if axis is None or any(a % rank == 0 for a in ([axis] if isinstance(axis, int) else axis)):
            raise NotImplementedError(f"{name} may not reduce/change the batch axis.")
    if name in {"ExpandDims", "Stack"}:
        axis = getattr(source, "axis", 0)
        if axis % (rank + 1) == 0:
            raise NotImplementedError(f"{name} may not move the leading batch axis.")
    if name == "GetItem":
        key = args[1]
        key = key if isinstance(key, tuple) else (key,)
        first_key = key[0]
        if first_key is not Ellipsis and first_key != slice(None):
            raise NotImplementedError("Static slicing must preserve the complete batch axis.")
    if name in {"Transpose", "Moveaxis", "Swapaxes"}:
        axes = getattr(source, "axes", None)
        if name == "Transpose" and (axes is None or axes[0] % rank != 0):
            raise NotImplementedError("Transpose must preserve the batch axis at position zero.")
        if name == "Moveaxis":
            for axis in (source.source, source.destination):
                axes = (axis,) if isinstance(axis, int) else axis
                if any(a % rank == 0 for a in axes):
                    raise NotImplementedError("Moveaxis may not move the batch axis.")
    if name == "Tile":
        repeats = tuple(source.repeats)
        if len(repeats) > rank or (len(repeats) == rank and repeats[0] != 1):
            raise NotImplementedError("Tile may not repeat the batch dimension.")
    if name == "Slice":
        raise NotImplementedError("Use static indexing with a full batch slice for jet tensors.")


def _evaluate(source: Any, args: Any, kwargs: Any, spec: JetSpec,
              backward: str = "native", activation_name: str = "linear", numerics: Any = None) -> list[Any]:
    ops = numerics or keras.ops
    strict = isinstance(ops, StrictNumerics)
    custom = _CUSTOM_RULES.get(type(source))
    if custom is not None:
        if strict:
            custom = _STRICT_RULES.get(type(source))
            if custom is None:
                raise NotImplementedError(f"{source.name}: strict precision requires an explicit strict_rule.")
        options = {"numerics": ops} if strict else {}
        result = custom(source, _map(args, lambda s: s.values), _map(kwargs, lambda s: s.values), spec, **options)
        if len(result) != spec.size:
            raise ValueError("Custom jet rule returned an incorrect coefficient count.")
        if strict:
            for value in result:
                ops.check(value)
        return list(result)
    size = spec.size
    name = type(source).__name__
    if isinstance(source, keras.layers.Dense):
        if getattr(source, "lora_enabled", False) or getattr(source, "quantization_mode", None):
            raise NotImplementedError("Quantized/LoRA Dense requires a registered jet rule.")
        packed = ops.matmul(ops.stack(args[0].values, axis=1), source.kernel)
        values = [packed[:, k] for k in range(size)]
        if source.use_bias:
            values[0] = values[0] + source.bias
        return _nonlinear(values, spec, activation_name, backward=backward, numerics=ops)
    if isinstance(source, (keras.layers.Conv1D, keras.layers.Conv2D)):
        packed = ops.stack(args[0].values, axis=1)
        packed_shape = ops.shape(packed)
        flattened = ops.reshape(packed, (-1, *packed_shape[2:]))
        padding = source.padding
        if padding == "causal":
            left = source.dilation_rate[0] * (source.kernel_size[0] - 1)
            pads = ((0, 0), (left, 0), (0, 0)) if source.data_format == "channels_last" else (
                (0, 0), (0, 0), (left, 0))
            flattened = ops.pad(flattened, pads)
            padding = "valid"
        convolution = ops.conv(flattened, source.kernel, strides=source.strides,
                               padding=padding, data_format=source.data_format,
                               dilation_rate=source.dilation_rate)
        packed = ops.reshape(convolution, (packed_shape[0], size, *ops.shape(convolution)[1:]))
        values = [packed[:, k] for k in range(size)]
        if source.use_bias:
            shape = (1,) * (len(values[0].shape) - 1) + (source.filters,)
            if source.data_format == "channels_first":
                shape = (1, source.filters) + (1,) * (len(values[0].shape) - 2)
            values[0] = values[0] + ops.reshape(source.bias, shape)
        return _nonlinear(values, spec, activation_name, backward=backward, numerics=ops)
    if isinstance(source, keras.layers.Activation):
        return _nonlinear(args[0].values, spec, activation_name, backward=backward, numerics=ops)
    if isinstance(source, keras.layers.Softmax):
        if kwargs.get("mask") is not None:
            raise NotImplementedError("Masked softmax requires an explicit smooth jet rule.")
        return _softmax(args[0].values, spec, source.axis, numerics=ops)
    if isinstance(source, (keras.layers.Add, keras.layers.Subtract, keras.layers.Multiply,
                           keras.layers.Average)):
        operands = args[0]
        values = _series(operands[0], size)
        for operand in operands[1:]:
            other = _series(operand, size)
            if isinstance(source, keras.layers.Multiply):
                values = multiply(values, other, spec, ops)
            elif isinstance(source, keras.layers.Subtract):
                values = [a - b for a, b in zip(values, other, strict=True)]
            else:
                values = [a + b for a, b in zip(values, other, strict=True)]
        return [v / len(operands) for v in values] if isinstance(source, keras.layers.Average) else values
    if isinstance(source, (keras.layers.Reshape, keras.layers.Flatten, keras.layers.Permute,
                           keras.layers.Concatenate)):
        _reject_batch_change(source, args)
        if isinstance(source, keras.layers.Concatenate):
            return [source(_merge_coefficient(args[0], k), **kwargs) for k in range(size)]
        return [source(*_map(args, lambda s, k=k: s.values[k]), **_map(kwargs, lambda s, k=k: s.values[k]))
                for k in range(size)]
    if not isinstance(source, keras.Operation) or isinstance(source, keras.layers.Layer):
        raise NotImplementedError(f"Unsupported Keras layer {name}; register a jet rule.")
    if not type(source).__module__.startswith("keras.src.ops."):
        raise NotImplementedError(f"Custom operation {name} needs a registered jet rule.")
    if name in _ELEMENTWISE:
        a = _series(args[0], size)
        if name == "Negative":
            return [-v for v in a]
        if name == "Square":
            return multiply(a, a, spec, ops)
        b = _series(args[1], size)
        if name == "Add":
            return [x + y for x, y in zip(a, b, strict=True)]
        if name == "Subtract":
            return [x - y for x, y in zip(a, b, strict=True)]
        if name == "Multiply":
            if not isinstance(args[0], _Series):
                return [args[0] * value for value in b]
            if not isinstance(args[1], _Series):
                return [value * args[1] for value in a]
            return multiply(a, b, spec, ops)
        if name in {"Divide", "TrueDivide"}:
            if not isinstance(args[1], _Series):
                return [value / args[1] for value in a]
            return multiply(a, reciprocal(b, spec, ops), spec, ops)
        if name == "Matmul":
            # Constant operands avoid scalar-zero matmuls and unnecessary work.
            if not isinstance(args[0], _Series):
                return [ops.matmul(args[0], v) for v in b]
            if not isinstance(args[1], _Series):
                return [ops.matmul(v, args[1]) for v in a]
            return [sum((ops.matmul(a[i], b[j]) for i, j in pairs), ops.zeros_like(ops.matmul(a[0], b[0])))
                    for pairs in spec.pairs]
        if name == "Power":
            exponent = args[1]
            if isinstance(exponent, _Series) or not isinstance(exponent, (int, float)):
                raise NotImplementedError("Power requires a constant scalar exponent.")
            if int(exponent) == exponent:
                result = [ops.ones_like(a[0])] + [ops.zeros_like(a[0]) for _ in range(size - 1)]
                base = a if exponent >= 0 else reciprocal(a, spec, ops)
                power = abs(int(exponent))
                while power:
                    if power & 1:
                        result = multiply(result, base, spec, ops)
                    power //= 2
                    if power:
                        base = multiply(base, base, spec, ops)
                return result
            logarithm = activation(a, spec, "log", ops)
            return activation([exponent * v for v in logarithm], spec, "exp", ops)
    if name in _LINEAR:
        _reject_batch_change(source, args)
        if strict and name == "Mean":
            return [ops.mean(value, axis=source.axis, keepdims=source.keepdims) for value in args[0].values]
        if name in {"Concatenate", "Stack"}:
            return [source(_merge_coefficient(args[0], k), **kwargs) for k in range(size)]
        return [source(*_map(args, lambda s, k=k: s.values[k]), **_map(kwargs, lambda s, k=k: s.values[k]))
                for k in range(size)]
    if name == "Pad":
        width = source.pad_width
        if tuple(width[0]) != (0, 0):
            raise NotImplementedError("Padding may not change the batch dimension.")
        return [ops.pad(args[0].values[k], width, mode=source.mode,
                        **({"constant_values": kwargs.get("constant_values", 0) if k == 0 else 0}
                           if source.mode == "constant" else {}))
                for k in range(size)]
    if name in {"OnesLike", "ZerosLike", "FullLike"}:
        value = source(*_map(args, _primal), **_map(kwargs, _primal))
        return [value] + [ops.zeros_like(value) for _ in range(size - 1)]
    lower = name.lower()
    if lower == "softmax":
        return _softmax(args[0].values, spec, source.axis, numerics=ops)
    if lower in _SMOOTH:
        if lower == "gelu" and getattr(source, "approximate", False):
            lower = "gelu_approx"
        return _nonlinear(args[0].values, spec, lower, backward=backward, numerics=ops)
    raise NotImplementedError(f"Unsupported Keras operation {name}; register a jet rule.")


@keras.saving.register_keras_serializable(package="omnibias")
class JetRuleLayer(keras.layers.Layer):
    """Serializable per-call rule holding the original source layer's variables."""

    def __init__(self, source: Any, spec: JetSpec, args_template: Any,
                 kwargs_template: Any, constants: Sequence[Any] = (),
                 backward: str = "native", precision: str = "source", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.source = source
        self.spec = spec
        self.args_template = args_template
        self.kwargs_template = kwargs_template
        self.constants = list(constants)
        self.backward = backward
        self.precision = validate_precision(precision)
        self.numerics = StrictNumerics(self.compute_dtype) if precision == "strict" else None
        if self.numerics is not None:
            strict_source_contract(source, self.compute_dtype)
            if type(source) in _CUSTOM_RULES and type(source) not in _STRICT_RULES:
                raise NotImplementedError(f"{source.name}: strict precision requires an explicit strict_rule.")
        self.primal_output_dtype: str | None = None
        self.activation_name = "linear"
        if isinstance(source, (keras.layers.Dense, keras.layers.Conv1D,
                               keras.layers.Conv2D, keras.layers.Activation)):
            self.activation_name = _activation_name(source.activation)

    def call(self, inputs: Any) -> Any:
        inputs = list(inputs) if isinstance(inputs, (tuple, list)) else [inputs]
        for tensor in inputs:
            _dtype(tensor.dtype)
            if tensor.shape[1] != self.spec.size:
                raise ValueError(f"Expected {self.spec.size} Taylor coefficients at axis 1.")
            shape = (tensor.shape[0], *tensor.shape[2:])
            _check_shape(self.spec, shape, tensor.dtype)
        args = _decode(self.args_template, inputs, self.constants, self.spec.size)
        kwargs = _decode(self.kwargs_template, inputs, self.constants, self.spec.size)
        values = _evaluate(self.source, args, kwargs, self.spec, self.backward, self.activation_name, self.numerics)
        # Keep the source operation's primal implementation, including GELU variants.
        primal = values[0] if self.precision == "strict" else self.source(*_map(args, _primal), **_map(kwargs, _primal))
        if isinstance(primal, (tuple, list, dict)):
            raise NotImplementedError("Multi-output operations require a registered model rule.")
        if len(primal.shape) < 1 or primal.shape[0] != inputs[0].shape[0]:
            raise NotImplementedError("Jet operations must preserve their leading batch dimension.")
        values[0] = primal
        if self.numerics is not None:
            for value in values:
                self.numerics.check(value)
        self.primal_output_dtype = _dtype(primal.dtype)
        _check_shape(self.spec, primal.shape, primal.dtype)
        return ops.stack(values, axis=1)

    def get_config(self) -> dict[str, Any]:
        result = super().get_config()
        result.update(source=keras.saving.serialize_keras_object(self.source),
                      spec=self.spec.to_config(), args_template=self.args_template,
                      kwargs_template=self.kwargs_template,
                      backward=self.backward, precision=self.precision,
                      constants=[keras.saving.serialize_keras_object(v) for v in self.constants])
        return result

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> JetRuleLayer:
        config = dict(config)
        config["source"] = keras.saving.deserialize_keras_object(config["source"])
        config["spec"] = JetSpec.from_config(config["spec"])
        config["constants"] = [keras.saving.deserialize_keras_object(v) for v in config["constants"]]
        return cls(**config)


@keras.saving.register_keras_serializable(package="omnibias")
class JetModel(PrecisionObjectiveMixin, keras.Model):
    """A normal Keras Model wrapping a transformed, shared-variable graph."""

    def __init__(self, graph: keras.Model, spec: JetSpec, report: Sequence[dict[str, Any]] = (),
                 backward: str = "native", precision: str = "source", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.graph = graph
        self.spec = spec
        self._report = list(report)
        self.backward = backward
        self.precision = validate_precision(precision)

    def call(self, inputs: Any, training: bool | None = None) -> Any:
        return self.graph(inputs, training=training)

    def explain(self) -> dict[str, Any]:
        return {"spec": self.spec.explain(), "coefficient_axis": 1,
                "backward": self.backward, "precision": self.precision,
                "compute_dtype": self.compute_dtype, "operations": list(self._report),
                "precision_notes": [r["precision_note"] for r in self._report if "precision_note" in r],
                "budget_note": "Symbolic dimensions are checked when concrete; compiled dynamic shapes require a bounded caller batch."}

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(graph=keras.saving.serialize_keras_object(self.graph),
                      spec=self.spec.to_config(), report=self._report, backward=self.backward, precision=self.precision)
        return config

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> JetModel:
        config = dict(config)
        config["graph"] = keras.saving.deserialize_keras_object(config["graph"])
        config["spec"] = JetSpec.from_config(config["spec"])
        return cls(**config)


def jet_model(model: keras.Model, *, spec: JetSpec, backward: str = "native",
              precision: str = "source") -> JetModel:
    """Lift a built Functional/Sequential model without copying its variables.

    The returned model accepts the original input structure with coefficient
    axis 1 inserted after each batch axis. Parameter AD is retained.
    ``precision='source'`` preserves source numerical behavior; ``'strict'``
    evaluates supported mathematics in the declared compute dtype and rejects
    conflicting authored casts/policies. Unsupported operations fail while
    constructing the graph.
    """
    if backward not in {"native", "analytic"}:
        raise ValueError("backward must be 'native' or 'analytic'.")
    validate_precision(precision)
    if not hasattr(keras, "Operation") or "call_function" not in inspect.signature(
        keras.models.clone_model
    ).parameters:
        raise RuntimeError("jet_model requires Keras Operation and clone_model(call_function=...); tested with Keras 3.15.")
    if not isinstance(spec, JetSpec):
        raise TypeError("spec must be a JetSpec.")
    if not isinstance(model, keras.Model) or not getattr(model, "inputs", None):
        raise TypeError("jet_model requires a built Functional or Sequential model.")
    strict_dtype = _dtype(model.inputs[0].dtype)
    if precision == "strict" and any(_dtype(value.dtype) != strict_dtype for value in model.inputs):
        raise ValueError("Strict precision requires consistent floating input dtypes.")
    cache: dict[int, keras.Model] = {}
    wrappers: dict[tuple[Any, ...], JetRuleLayer] = {}
    primal_dtypes: dict[int, str] = {}
    model_output_dtypes: dict[int, Any] = {}
    report: list[dict[str, Any]] = []

    def lift(source_model: keras.Model) -> keras.Model:
        key = id(source_model)
        if key in cache:
            return cache[key]
        if isinstance(source_model, keras.Sequential):
            original = keras.Model(source_model.inputs, source_model.outputs)
        elif type(source_model).__name__ == "Functional":
            original = source_model
        else:
            raise TypeError("Subclassed/custom models require a registered jet rule.")

        def make_input(tensor: Any) -> Any:
            dtype = _dtype(tensor.dtype)
            return keras.Input(batch_shape=(tensor.shape[0], spec.size, *tensor.shape[1:]),
                               dtype=dtype, name=tensor.name.split(":")[0])

        inputs = keras.tree.map_structure(make_input, original.input)

        def transform(source: Any, *args: Any, **kwargs: Any) -> Any:
            if isinstance(source, keras.Model) and type(source) not in _CUSTOM_RULES:
                nested = lift(source)
                outputs = nested(*args, **kwargs)
                for tensor, dtype in zip(keras.tree.flatten(outputs),
                                         keras.tree.flatten(model_output_dtypes[id(source)]), strict=True):
                    primal_dtypes[id(tensor)] = dtype
                return outputs
            tensors: list[Any] = []
            constants: list[Any] = []
            args_template = _encode(args, tensors, constants, primal_dtypes)
            kwargs_template = _encode(kwargs, tensors, constants, primal_dtypes)
            if not tensors:
                raise NotImplementedError("A captured operation without spatial input needs an explicit rule.")
            dtype = _dtype(getattr(source, "compute_dtype", tensors[0].dtype))
            if precision == "strict":
                strict_source_contract(source, strict_dtype)
                dtype = strict_dtype
            if isinstance(source, keras.layers.Layer) and source.autocast:
                _apply_input_policy(args_template, dtype)
                _apply_input_policy(kwargs_template, dtype)
            key = (id(source), repr(args_template), repr(kwargs_template), tuple(map(id, constants)))
            wrapper = wrappers.get(key)
            if wrapper is None:
                wrapper = JetRuleLayer(source, spec, args_template, kwargs_template, constants,
                                       backward=backward, precision=precision, dtype=dtype,
                                       name=f"jet_{source.name}_{len(report)}")
                wrappers[key] = wrapper
            entry = {"name": source.name, "type": type(source).__name__,
                     "rule": "custom" if type(source) in _CUSTOM_RULES else "native_jet"}
            activation_fn = getattr(source, "activation", None)
            if type(source).__name__.lower() in {"tanh", "sin", "cos"}:
                activation_fn = source
            if type(source).__name__ == "Gelu" and getattr(source, "approximate", False):
                activation_fn = ops.tanh
            if activation_fn is not None and precision == "source":
                observed = _dtype(activation_fn(ops.zeros((1,), dtype=dtype)).dtype)
                if observed != dtype:
                    entry["precision_note"] = (
                        f"{source.name}: source activation converts {dtype} to {observed}; "
                        "jet accuracy inherits this upstream precision limit."
                    )
            if precision == "source" and (isinstance(source, keras.layers.Dense) or type(source).__name__ == "Matmul"):
                probe = ops.ones((1, 1), dtype=dtype)
                observed = _dtype(ops.matmul(probe, probe).dtype)
                if observed != dtype:
                    note = (f"{source.name}: source matmul computes {dtype} operands as {observed}; "
                            "a later bias can promote the output dtype without restoring precision.")
                    entry["precision_note"] = (entry.get("precision_note", "") + " " + note).strip()
            report.append(entry)
            output = wrapper(tensors)
            primal_dtypes[id(output)] = wrapper.primal_output_dtype or output.dtype
            return output

        transformed = keras.models.clone_model(original, input_tensors=inputs,
                                               clone_function=lambda source: source,
                                               call_function=transform)
        cache[key] = transformed
        model_output_dtypes[key] = keras.tree.map_structure(
            lambda tensor: primal_dtypes.get(id(tensor), tensor.dtype), transformed.output)
        return transformed

    graph = lift(model)
    dtype = _dtype(keras.tree.flatten(graph.inputs)[0].dtype)
    return JetModel(graph, spec, report, backward=backward, precision=precision, dtype=dtype, name=f"{model.name}_jet")
