# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Independent precision, higher parameter gradients and real training gates."""

from __future__ import annotations

import keras
import mpmath as mp
import numpy as np
import pytest
from keras import ops
from omnibias.core.jet_series import multiply
from omnibias.keras.experimental import (
    JetSpec,
    OperatorSpec,
    PrecisionMeanSquaredError,
    jet_model,
    operator_model,
    register_jet_rule,
    seed,
)


def array(value):
    return np.asarray(ops.convert_to_numpy(value))


def gradients(function, parameters):
    backend = keras.backend.backend()
    if backend == "jax":
        import jax
        gradient = jax.grad(function)
        first = gradient(parameters)
        second = jax.grad(lambda p: sum(ops.sum(v) for v in gradient(p)))(parameters)
    elif backend == "torch":
        import torch
        first = torch.autograd.grad(function(parameters), parameters, create_graph=True)
        second = torch.autograd.grad(sum(v.sum() for v in first), parameters)
    else:
        import tensorflow as tf
        with tf.GradientTape() as outer:
            with tf.GradientTape() as inner:
                value = function(parameters)
            first = inner.gradient(value, parameters)
            gradient_sum = sum(tf.reduce_sum(v) for v in first)
        second = outer.gradient(gradient_sum, parameters)
    return np.concatenate([array(v).reshape(-1) for v in first]), np.concatenate([array(v).reshape(-1) for v in second])


def dense_source():
    x = keras.Input((2,), dtype="float64")
    source = keras.Model(x, keras.layers.Dense(1, activation="tanh", dtype="float64")(x))
    source.set_weights([np.array([[.7], [.2]]), np.array([.1])])
    return source


@pytest.mark.parametrize("strategy", ["auto", "reference", "jet_native", "jet_analytic"])
def test_strict_ridge_independent_value_parameter_gradient_and_hvp(strategy):
    source = dense_source()
    points = ops.convert_to_tensor([[.1234567890123, .234567890123]], dtype="float64")
    if strategy.startswith("jet_"):
        spec = JetSpec.directional(2)
        model = jet_model(source, spec=spec, precision="strict", backward=strategy[4:])
        points = seed(points, ops.convert_to_tensor([[1., 0.]], dtype="float64"), spec=spec)
    else:
        model = operator_model(source, operator=OperatorSpec.mixed((2, 0)), strategy=strategy, precision="strict")
    model(points)
    parameters = [variable.value for variable in model.trainable_variables]
    def function(p):
        value = model.stateless_call(p, [], points)[0]
        return ops.sum(value[:, 2]*2 if strategy.startswith("jet_") else value)
    actual = array(function(parameters)).item()
    first, second = gradients(function, parameters)
    with mp.workdps(90):
        theta = tuple(mp.mpf(float(v)) for v in (.7, .2, .1))
        x, y = map(lambda v: mp.mpf(float(v)), (.1234567890123, .234567890123))
        def oracle(a, b, c):
            z = a * x + b * y + c
            return -2 * a * a * mp.tanh(z) / mp.cosh(z)**2
        expected = float(oracle(*theta))
        unit = [tuple(int(i == j) for i in range(3)) for j in range(3)]
        expected_first = [float(mp.diff(oracle, theta, alpha)) for alpha in unit]
        expected_second = [float(sum(mp.diff(oracle, theta, tuple(a + b for a, b in zip(left, right, strict=True)))
                                     for right in unit)) for left in unit]
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=1e-15)
    np.testing.assert_allclose(first, expected_first, rtol=4e-13, atol=2e-14)
    np.testing.assert_allclose(second, expected_second, rtol=8e-13, atol=3e-14)
    assert {id(v) for v in source.trainable_variables} == {id(v) for v in model.trainable_variables}
    assert model.explain()["precision"] == "strict"


@pytest.mark.parametrize("backward", ["native", "analytic"])
def test_strict_jet_primal_and_coefficients(backward):
    source = dense_source()
    spec = JetSpec.directional(8)
    model = jet_model(source, spec=spec, precision="strict", backward=backward)
    point = np.array([[.1234567890123, .234567890123]])
    direction = np.array([[.17, -.31]])
    result = array(model(seed(ops.convert_to_tensor(point), ops.convert_to_tensor(direction), spec=spec)))
    with mp.workdps(90):
        p = list(map(lambda v: mp.mpf(float(v)), point[0]))
        v = list(map(lambda v: mp.mpf(float(v)), direction[0]))
        a, b, c = map(lambda v: mp.mpf(float(v)), (.7, .2, .1))
        expected = [float(row) for row in mp.taylor(lambda t: mp.tanh(a*(p[0]+t*v[0])+b*(p[1]+t*v[1])+c), 0, 8)]
    np.testing.assert_allclose(result[0, :, 0], expected, rtol=8e-13, atol=2e-16)
    assert model.compute_dtype == "float64"


@pytest.mark.parametrize("strategy", ["auto", "reference"])
@pytest.mark.parametrize("primitive", ["tanh", "sin", "cos", "softplus", "mean"])
def test_strict_public_primitive_and_reduction_precision(primitive, strategy):
    x = keras.Input((2,), dtype="float64")
    state = ops.mean(x, axis=-1, keepdims=True) if primitive == "mean" else getattr(ops, primitive)(x)
    source = keras.Model(x, ops.exp(state))
    request = OperatorSpec.mixed((2, 0))
    model = operator_model(source, operator=request, strategy=strategy, precision="strict")
    point = np.array([[.1234567890123, .2718281828459]])
    with mp.workdps(90):
        a, b = map(lambda value: mp.mpf(float(value)), point[0])
        if primitive == "mean":
            expected = [[float(mp.exp((a+b)/2)/4)]]
        else:
            inner = (lambda z: mp.log(1+mp.exp(z))) if primitive == "softplus" else getattr(mp, primitive)
            expected = [[float(mp.diff(lambda z: mp.exp(inner(z)), a, 2)), 0.]]
    np.testing.assert_allclose(array(model(point)), expected, rtol=2e-12, atol=2e-14)


@pytest.mark.parametrize("approximate", [False, True])
@pytest.mark.parametrize("order", [0, 2, 8, 32])
def test_strict_gelu_preserves_variant_with_precise_constants(approximate, order):
    x = keras.Input((1,), dtype="float64")
    model = operator_model(keras.Model(x, ops.gelu(x, approximate=approximate)),
                           operator=OperatorSpec.mixed((order,)), precision="strict")
    values = [-8., -.1234567890123, .27, 8.]
    with mp.workdps(100):
        def formula(z):
            if approximate:
                return z * (1+mp.tanh(mp.sqrt(2/mp.pi)*(z+mp.mpf(.044715)*z**3)))/2
            return z * mp.erfc(-z/mp.sqrt(2))/2
        expected = [float(mp.diff(formula, mp.mpf(value), order)) for value in values]
    np.testing.assert_allclose(array(model(np.array(values)[:, None]))[:, 0], expected, rtol=2e-10, atol=0)


def test_strict_softplus_is_canonical_beyond_torch_threshold():
    x = keras.Input((1,), dtype="float64")
    source = keras.Model(x, ops.softplus(x))
    request = OperatorSpec.mixed((2,))
    strict = operator_model(source, operator=request, precision="strict")
    result = array(strict(np.array([[21.]]))).item()
    with mp.workdps(80):
        expected = float(mp.diff(lambda z: mp.log(1+mp.exp(z)), mp.mpf(21), 2))
    assert result == pytest.approx(expected, rel=2e-13, abs=0)
    if keras.backend.backend() == "torch":
        assert array(operator_model(source, operator=request)(np.array([[21.]]))).item() == 0


@pytest.mark.parametrize("frontend", ["jet", "operator"])
def test_strict_float32_declared_contract_and_typed_objective(frontend):
    x = keras.Input((1,), dtype="float32")
    source = keras.Model(x, ops.sin(x)*ops.exp(x))
    point = np.array([[.1234567890123]], dtype=np.float32)
    if frontend == "jet":
        spec = JetSpec.directional(2)
        model = jet_model(source, spec=spec, precision="strict")
        prediction = model(seed(ops.convert_to_tensor(point), ops.ones_like(ops.convert_to_tensor(point)), spec=spec))
        actual = array(prediction)[:, 2]*2
    else:
        model = operator_model(source, operator=OperatorSpec.mixed((2,)), precision="strict")
        prediction = model(point)
        actual = array(prediction)
    expected = 2*np.exp(point.astype(np.float64))*np.cos(point.astype(np.float64))
    np.testing.assert_allclose(actual, expected, rtol=4e-7)
    assert keras.backend.standardize_dtype(prediction.dtype) == "float32"
    model.compile(loss=PrecisionMeanSquaredError(dtype="float32"))
    value = model.compute_loss(y=ops.zeros_like(prediction), y_pred=prediction)
    assert keras.backend.standardize_dtype(value.dtype) == "float32"


@pytest.mark.parametrize("frontend", ["jet", "operator"])
def test_strict_rejects_declared_downcasts_and_mixed_layer_policy(frontend):
    def lift(source):
        return (jet_model(source, spec=JetSpec.directional(2), precision="strict") if frontend == "jet"
                else operator_model(source, operator=OperatorSpec.mixed((2,)), precision="strict"))
    x = keras.Input((1,), dtype="float64")
    with pytest.raises(ValueError, match="explicit cast"):
        lift(keras.Model(x, ops.cast(x, "float32")))
    with pytest.raises(ValueError, match="compute dtype"):
        lift(keras.Model(x, keras.layers.Dense(1, dtype="float32")(x)))


def test_strict_custom_rule_requires_opt_in_and_owns_primal():
    @keras.saving.register_keras_serializable(package="precision_test")
    class Cube(keras.layers.Layer):
        def call(self, inputs):
            return inputs**3
    def source_rule(source, args, kwargs, spec):
        return multiply(multiply(args[0], args[0], spec, ops), args[0], spec, ops)
    register_jet_rule(Cube, source_rule)
    x = keras.Input((1,), dtype="float64")
    source = keras.Model(x, Cube(dtype="float64")(x))
    with pytest.raises(NotImplementedError, match="strict_rule"):
        operator_model(source, operator=OperatorSpec.mixed((2,)), precision="strict")
    def strict_rule(source, args, kwargs, spec, *, numerics):
        return multiply(multiply(args[0], args[0], spec, numerics), args[0], spec, numerics)
    register_jet_rule(Cube, source_rule, strict_rule=strict_rule)
    model = operator_model(source, operator=OperatorSpec.mixed((2,)), precision="strict")
    np.testing.assert_allclose(array(model(np.array([[.1234567890123]]))), [[6*.1234567890123]], rtol=1e-14)
    register_jet_rule(Cube, source_rule)
    with pytest.raises(NotImplementedError, match="strict_rule"):
        jet_model(source, spec=JetSpec.directional(2), precision="strict")


@pytest.mark.parametrize("reduction", [None, "sum", "mean", "mean_with_sample_weight"])
def test_precision_loss_owns_all_reductions_and_serializes_dtype(reduction):
    loss = PrecisionMeanSquaredError(reduction=reduction)
    truth = np.array([[.12, -.31], [.23, .13]])
    prediction = np.array([[.1234567890123, -.2718281828459], [.3141592653589, .1414213562373]])
    weights = np.array([.37, .91])
    rows = np.mean((prediction-truth)**2, axis=-1)*weights
    expected = rows if reduction is None else np.sum(rows)
    if reduction == "mean":
        expected /= 2
    elif reduction == "mean_with_sample_weight":
        expected /= weights.sum()
    actual = loss(truth, prediction, weights)
    assert keras.backend.standardize_dtype(actual.dtype) == "float64"
    np.testing.assert_allclose(array(actual), expected, rtol=3e-14, atol=1e-18)
    loaded = keras.saving.deserialize_keras_object(keras.saving.serialize_keras_object(loss))
    assert loaded.dtype == "float64"
    np.testing.assert_array_equal(array(loaded(truth, prediction, weights)), array(actual))


def test_zero_weight_loss_and_unsupported_objectives():
    loss = PrecisionMeanSquaredError(reduction="mean_with_sample_weight")
    assert array(loss(np.zeros((2, 1)), np.ones((2, 1)), np.zeros(2))).item() == 0
    source = dense_source()
    model = operator_model(source, operator=OperatorSpec.mixed((2, 0)), precision="strict")
    with pytest.raises(ValueError, match="scalar loss"):
        model.compile(loss=PrecisionMeanSquaredError(reduction=None))
    with pytest.raises(ValueError, match="dtype"):
        model.compile(loss=PrecisionMeanSquaredError(dtype="float32"))
    with pytest.raises(ValueError, match="finite scalar"):
        model.compile(loss=PrecisionMeanSquaredError(), loss_weights=float("nan"))
    x = keras.Input((1,), dtype="float64")
    regularized = keras.Model(x, keras.layers.Dense(1, activation="tanh", dtype="float64", kernel_regularizer="l2")(x))
    regularized = operator_model(regularized, operator=OperatorSpec.mixed((2,)), precision="strict")
    with pytest.raises(NotImplementedError, match="regularization"):
        regularized.compile(loss=PrecisionMeanSquaredError())


@pytest.mark.parametrize("reduction", [None, "mean", "mean_with_sample_weight"])
def test_precision_loss_vector_targets_preserve_per_example_weights(reduction):
    truth = np.array([.1234567890123, .2718281828459])
    prediction = np.array([[.3141592653589], [.1414213562373]])
    weights = np.array([.37, .91])
    expected = (prediction[:, 0]-truth)**2*weights
    if reduction == "mean":
        expected = expected.sum()/2
    elif reduction == "mean_with_sample_weight":
        expected = expected.sum()/weights.sum()
    actual = PrecisionMeanSquaredError(reduction=reduction)(truth, prediction, weights)
    np.testing.assert_allclose(array(actual), expected, rtol=2e-14, atol=1e-17)


@pytest.mark.parametrize("masked_target", [False, True])
def test_precision_loss_refuses_implicit_masks(masked_target):
    values = [ops.ones((2, 1), dtype="float64"), ops.ones((2, 1), dtype="float64")]
    values[int(not masked_target)]._keras_mask = ops.ones((2,), dtype="bool")
    with pytest.raises(NotImplementedError, match="implicit masks"):
        PrecisionMeanSquaredError()(*values)


@pytest.mark.parametrize("pattern", ["scalar_cut", "separable", "affine_bottleneck", "apolar"])
def test_strict_structural_paths_preserve_accurate_primal(pattern):
    if pattern == "separable":
        dimension = 24
        x = keras.Input((dimension,), dtype="float64")
        output = ops.sin(ops.mean(ops.tanh(x*.3), axis=-1, keepdims=True))
        point = np.linspace(.01, .05, dimension)[None, :]
        request = OperatorSpec.mixed((1,)*dimension)
        with mp.workdps(100):
            a = mp.mpf(.3)
            p = list(map(lambda value: mp.mpf(float(value)), point[0]))
            s = sum(mp.tanh(a*v) for v in p)/dimension
            expected = mp.diff(mp.sin, s, dimension)*mp.fprod(a/dimension/mp.cosh(a*v)**2 for v in p)
    elif pattern == "scalar_cut":
        x = keras.Input((2,), dtype="float64")
        state = x[:, :1]+x[:, 1:]
        output = ops.sin(state)*ops.exp(state)
        point = np.array([[.1234567890123, .2718281828459]])
        request = OperatorSpec.mixed((2, 2))
        with mp.workdps(90):
            expected = mp.diff(lambda t: mp.sin(t)*mp.exp(t), sum(mp.mpf(float(v)) for v in point[0]), 4)
    elif pattern == "affine_bottleneck":
        x = keras.Input((3,), dtype="float64")
        dense = keras.layers.Dense(1, activation="tanh", dtype="float64")
        state = dense(x)
        output = ops.exp(state)+ops.cos(state)
        dense.set_weights([np.array([[.3], [.7], [-.2]]), np.array([.1])])
        point = np.array([[.1234567890123, .2718281828459, -.3141592653589]])
        request = OperatorSpec.mixed((2, 1, 1))
        with mp.workdps(90):
            weights = list(map(mp.mpf, (.3, .7, -.2)))
            z = sum(w*mp.mpf(float(v)) for w, v in zip(weights, point[0], strict=True))+mp.mpf(.1)
            expected = weights[0]**2*weights[1]*weights[2]*mp.diff(lambda t: mp.exp(mp.tanh(t))+mp.cos(mp.tanh(t)), z, 4)
    else:
        x = keras.Input((3,), dtype="float64")
        output = ops.tanh(x[:, :1]*x[:, 1:2]+x[:, 1:2]*x[:, 2:])
        point = np.array([[.1234567890123, .2718281828459, -.3141592653589]])
        request = OperatorSpec.laplacian(2)
        with mp.workdps(90):
            p = tuple(mp.mpf(float(v)) for v in point[0])
            def f(a, b, c):
                return mp.tanh(a*b+b*c)
            expected = sum(mp.diff(f, p, tuple(2*int(k == i)+2*int(k == j) for k in range(3)))
                           for i in range(3) for j in range(3))
    model = operator_model(keras.Model(x, output), operator=request, precision="strict")
    assert model.explain()["selected_strategy"] == pattern
    np.testing.assert_allclose(array(model(point)), [[float(expected)]], rtol=5e-11, atol=0)


@pytest.mark.parametrize("rank", [1, 2])
def test_strict_convolution_jet_and_symbolic_dtype(rank):
    shape = (3,)*rank+(1,)
    x = keras.Input(shape, dtype="float64")
    layer = (keras.layers.Conv1D if rank == 1 else keras.layers.Conv2D)(1, 2, activation="tanh", dtype="float64")
    source = keras.Model(x, layer(x))
    kernel = (np.array([.3, .7]) if rank == 1 else np.array([.1, .2, .3, .4])).reshape((2,)*rank+(1, 1))
    layer.set_weights([kernel, np.array([.1])])
    point = np.linspace(.1234567890123, .3141592653589, 3**rank).reshape((1,)+shape)
    spec = JetSpec.directional(2)
    model = jet_model(source, spec=spec, precision="strict")
    result = array(model(seed(ops.convert_to_tensor(point), ops.ones_like(ops.convert_to_tensor(point)), spec=spec)))
    expected = []
    with mp.workdps(90):
        for index in np.ndindex((2,)*rank):
            window = point[(0,)+tuple(slice(i, i+2) for i in index)+(0,)]
            z = sum(mp.mpf(float(a))*mp.mpf(float(b)) for a, b in zip(window.flat, kernel.flat, strict=True))+mp.mpf(.1)
            slope = sum(mp.mpf(float(v)) for v in kernel.flat)
            expected.append([float(mp.tanh(z)), float(slope/mp.cosh(z)**2), float(-slope**2*mp.tanh(z)/mp.cosh(z)**2)])
    expected = np.array(expected).T.reshape((1, 3)+(2,)*rank+(1,))
    np.testing.assert_allclose(result, expected, rtol=6e-13, atol=2e-15)
    assert model.graph.output.dtype == "float64"


def test_strict_structured_objective_weights_and_external_loss_delegation():
    x = keras.Input((1,), dtype="float64")
    source = keras.Model(x, {"a": ops.exp(x), "b": x*x})
    model = operator_model(source, operator=OperatorSpec.mixed((1,)), strategy="reference", precision="strict")
    losses = {"a": PrecisionMeanSquaredError(), "b": PrecisionMeanSquaredError()}
    model.compile(loss=losses, loss_weights={"a": .37, "b": .91})
    points = np.array([[.1234567890123], [.2718281828459]])
    prediction = model(points)
    target = {key: np.zeros((2, 1)) for key in prediction}
    samples = {"a": np.array([.17, .53]), "b": np.array([.71, .43])}
    actual = model.compute_loss(y=target, y_pred=prediction, sample_weight=samples)
    expected = .37*np.mean(np.exp(points[:, 0])**2*samples["a"])+.91*np.mean((2*points[:, 0])**2*samples["b"])
    np.testing.assert_allclose(array(actual), expected, rtol=3e-14, atol=2e-16)
    with pytest.raises(ValueError, match="mixed"):
        model.compile(loss={"a": PrecisionMeanSquaredError(), "b": "mse"})
    model.compile(loss="mse")
    assert model._precision_objective is None


@pytest.mark.filterwarnings("ignore:.*lru_cache.*:UserWarning:torch.*")
@pytest.mark.filterwarnings("ignore:.*non-leaf Tensor.*:UserWarning")
@pytest.mark.parametrize("frontend", ["operator", "jet_native", "jet_analytic"])
def test_actual_compiled_strict_objective_sgd_and_optimizer_continuation(tmp_path, frontend):
    source = dense_source()
    if frontend == "operator":
        model = operator_model(source, operator=OperatorSpec.mixed((2, 0)), precision="strict")
    else:
        spec = JetSpec.directional(2)
        model = jet_model(source, spec=spec, backward=frontend[4:], precision="strict")
    model.compile(optimizer=keras.optimizers.SGD(.125), loss=PrecisionMeanSquaredError(), jit_compile=True)
    point = np.array([[.1234567890123, .234567890123]])
    with mp.workdps(90):
        theta = tuple(mp.mpf(float(v)) for v in (.7, .2, .1))
        p = [mp.mpf(float(v)) for v in point[0]]
        def objective(a, b, c):
            z = a*p[0]+b*p[1]+c
            if frontend == "operator":
                return (-2*a*a*mp.tanh(z)/mp.cosh(z)**2)**2
            coefficients = [mp.tanh(z), a/mp.cosh(z)**2, -a*a*mp.tanh(z)/mp.cosh(z)**2]
            return sum(value**2 for value in coefficients)/3
        expected_loss = float(objective(*theta))
        gradient = np.array([float(mp.diff(objective, theta, tuple(int(i == j) for i in range(3)))) for j in range(3)])
    if frontend != "operator":
        point = seed(ops.convert_to_tensor(point), ops.convert_to_tensor([[1., 0.]], dtype="float64"), spec=spec)
    target = ops.zeros_like(model(point))
    value = model.compute_loss(y=target, y_pred=model(point))
    assert keras.backend.standardize_dtype(value.dtype) == "float64"
    np.testing.assert_allclose(array(value), expected_loss, rtol=3e-13)
    model.train_on_batch(point, target)
    actual = np.concatenate([v.reshape(-1) for v in source.get_weights()])
    np.testing.assert_allclose(actual, np.array([.7, .2, .1])-.125*gradient, rtol=3e-13, atol=1e-14)
    path = tmp_path / "strict.keras"
    model.save(path)
    loaded = keras.models.load_model(path)
    assert loaded.precision == "strict"
    assert loaded.loss.dtype == "float64"
    assert int(array(loaded.optimizer.iterations)) == 1
    np.testing.assert_array_equal(array(model(point)), array(loaded(point)))
    loaded.train_on_batch(point, target)
    assert int(array(loaded.optimizer.iterations)) == 2
    assert keras.backend.standardize_dtype(loaded.compute_loss(y=target, y_pred=loaded(point)).dtype) == "float64"


@pytest.mark.filterwarnings("ignore:.*lru_cache.*:UserWarning:torch.*")
@pytest.mark.filterwarnings("ignore:.*non-leaf Tensor.*:UserWarning")
def test_compiled_structured_weighted_objective_and_continuation(tmp_path):
    base = dense_source()
    source = keras.Model(base.input, {"a": base.output, "b": 2*base.output})
    model = operator_model(source, operator=OperatorSpec.mixed((2, 0)), precision="strict")
    model.compile(optimizer=keras.optimizers.SGD(.125),
                  loss={"a": PrecisionMeanSquaredError(reduction="mean_with_sample_weight"),
                        "b": PrecisionMeanSquaredError()},
                  loss_weights={"a": .37, "b": .91}, jit_compile=True)
    point = np.array([[.1234567890123, .234567890123], [-.2718281828459, .3141592653589]])
    target = {"a": np.zeros((2, 1)), "b": np.zeros((2, 1))}
    samples = {"a": np.array([.17, .53]), "b": np.array([.71, .43])}
    with mp.workdps(90):
        theta = tuple(mp.mpf(float(v)) for v in (.7, .2, .1))
        points = [[mp.mpf(float(v)) for v in row] for row in point]
        wa, wb = ([mp.mpf(float(v)) for v in samples[name]] for name in ("a", "b"))
        def objective(a, b, c):
            values = [-2*a*a*mp.tanh(a*x+b*y+c)/mp.cosh(a*x+b*y+c)**2 for x, y in points]
            return (mp.mpf(.37)*sum(w*v*v for w, v in zip(wa, values, strict=True))/sum(wa)
                    + mp.mpf(.91)*sum(w*(2*v)**2 for w, v in zip(wb, values, strict=True))/2)
        expected = float(objective(*theta))
        gradient = np.array([float(mp.diff(objective, theta, tuple(int(i == j) for i in range(3)))) for j in range(3)])
    np.testing.assert_allclose(array(model.compute_loss(y=target, y_pred=model(point), sample_weight=samples)),
                               expected, rtol=4e-13)
    model.train_on_batch(point, target, sample_weight=samples)
    actual = np.concatenate([v.reshape(-1) for v in base.get_weights()])
    np.testing.assert_allclose(actual, np.array([.7, .2, .1])-.125*gradient, rtol=5e-13, atol=1e-14)
    path = tmp_path / "structured_strict.keras"
    model.save(path)
    loaded = keras.models.load_model(path)
    assert int(array(loaded.optimizer.iterations)) == 1
    loaded.train_on_batch(point, target, sample_weight=samples)
    model.train_on_batch(point, target, sample_weight=samples)
    for got, wanted in zip(loaded.get_weights(), model.get_weights(), strict=True):
        np.testing.assert_array_equal(got, wanted)
