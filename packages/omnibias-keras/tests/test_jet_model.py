# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Independent algebra, training and serialization regressions for model jets."""

from __future__ import annotations

import math

import keras
import mpmath as mp
import numpy as np
import pytest
from keras import ops
from omnibias.keras.experimental import JetSpec, extract, jet_model, register_jet_rule, seed


def _array(value):
    return ops.convert_to_numpy(value)


@pytest.mark.parametrize("backward", ["native", "analytic"])
@pytest.mark.parametrize("order", [2, 4, 8, 16, 32])
def test_nonlinear_path_against_high_precision(order, backward):
    x = keras.Input((1,), dtype="float64")
    source = keras.Model(x, ops.tanh(x) + x * x)
    spec = JetSpec.directional(order)
    model = jet_model(source, spec=spec, backward=backward)
    coefficients = np.zeros((1, order + 1, 1))
    coefficients[0, :3, 0] = [.3, .4, -.1]
    result = _array(model(coefficients))[0, :, 0]
    with mp.workdps(90):
        def curve(t):
            return mp.mpf(.3) + mp.mpf(.4) * t + mp.mpf(-.1) * t * t
        expected = np.array([float(c) for c in mp.taylor(lambda t: mp.tanh(curve(t)) + curve(t)**2, 0, order)])
    if model.explain()["precision_notes"]:
        # Keras 3.15's JAX tanh explicitly computes a float32 primal even
        # with x64 enabled. Bound that observed initialization error only.
        np.testing.assert_allclose(result[0], expected[0], rtol=0, atol=2e-8)
        np.testing.assert_allclose(result[1:], expected[1:], rtol=2e-6, atol=2e-8)
    else:
        np.testing.assert_allclose(result, expected, rtol=2e-10, atol=2e-15)


def test_selected_mixed_polynomial_and_factorials():
    x = keras.Input((2,), dtype="float64")
    a, b = x[:, :1], x[:, 1:]
    source = keras.Model(x, a**2 * b**2 + 3 * a)
    spec = JetSpec.mixed([(2, 2)])
    model = jet_model(source, spec=spec)
    values = ops.convert_to_tensor([[.2, -.3], [.5, .7]], dtype="float64")
    packed = seed(values, [ops.ones_like(values) * ops.convert_to_tensor([1., 0.], dtype="float64"),
                           ops.ones_like(values) * ops.convert_to_tensor([0., 1.], dtype="float64")], spec=spec)
    result = model(packed)
    np.testing.assert_allclose(_array(extract(result, (2, 2), spec=spec)), 1., atol=1e-14)
    np.testing.assert_allclose(_array(extract(result, (2, 2), spec=spec, derivative=True)), 4., atol=1e-14)
    assert model.explain()["spec"]["coefficients"] == 9


@pytest.mark.parametrize("backward", ["native", "analytic"])
def test_training_shared_weights_and_serialization(tmp_path, backward):
    x = keras.Input((1,), dtype="float64")
    dense = keras.layers.Dense(1, activation="tanh", dtype="float64")
    source = keras.Model(x, dense(x) + dense(x))
    dense.kernel.assign([[.8]])
    dense.bias.assign([.2])
    spec = JetSpec.directional(2)
    model = jet_model(source, spec=spec, backward=backward)
    assert len(model.trainable_variables) == len(source.trainable_variables) == 2
    assert all(a is b for a, b in zip(model.trainable_variables, source.trainable_variables, strict=True))
    packed = seed(ops.convert_to_tensor([[.3], [-.4]], dtype="float64"),
                  ops.ones((2, 1), dtype="float64"), spec=spec)
    residual = keras.Model(model.graph.input, model.graph.output[:, 2])
    residual.compile(optimizer=keras.optimizers.SGD(.001), loss="mse", jit_compile=False)
    before = np.array(dense.kernel.numpy(), copy=True)
    residual.train_on_batch(packed, np.zeros((2, 1)))
    assert not np.array_equal(before, dense.kernel.numpy())
    reference = _array(model(packed))
    path = tmp_path / "jet.keras"
    model.save(path)
    restored = keras.saving.load_model(path)
    np.testing.assert_allclose(_array(restored(packed)), reference, rtol=1e-13, atol=1e-14)
    assert len(restored.trainable_variables) == 2
    assert restored.explain()["backward"] == backward


@pytest.mark.parametrize("backward", ["native", "analytic"])
def test_residual_parameter_gradient_against_independent_formula(backward):
    x = keras.Input((1,), dtype="float64")
    dense = keras.layers.Dense(1, activation="tanh", dtype="float64")
    source = keras.Model(x, dense(x))
    dense.kernel.assign([[.7]])
    dense.bias.assign([.1])
    model = jet_model(source, spec=JetSpec.directional(2), backward=backward)
    points = np.array([[.2], [-.3]])
    packed = seed(points, np.ones_like(points), spec=model.spec)
    residual = keras.Model(model.graph.input, model.graph.output[:, 2])
    rate = 1e-4
    residual.compile(optimizer=keras.optimizers.SGD(rate), loss="mse", jit_compile=False)
    residual.train_on_batch(packed, np.zeros_like(points))
    measured = (.7 - float(dense.kernel.numpy()[0, 0])) / rate
    def exact_loss(weight):
        tangent = np.tanh(weight * points + .1)
        coefficients = -tangent * (1 - tangent**2) * weight**2
        return np.mean(coefficients**2)
    h = 1e-5
    reference = (exact_loss(.7 + h) - exact_loss(.7 - h)) / (2 * h)
    assert measured == pytest.approx(reference, rel=2e-6, abs=1e-10)


def test_nested_sequential_and_input_dependence(tmp_path):
    inner = keras.Sequential([keras.Input((2,), dtype="float64"),
                              keras.layers.Dense(2, activation="tanh", dtype="float64")])
    x = keras.Input((2,), dtype="float64")
    source = keras.Model(x, inner(x) * inner(x))
    model = jet_model(source, spec=JetSpec.directional(2))
    assert len(model.trainable_variables) == 2
    assert all(a is b for a, b in zip(model.weights, source.weights, strict=True))
    packed = seed(np.array([[.2, -.1]]), np.array([[.3, .4]]), spec=model.spec)
    result = _array(model(packed))
    values = _array(source(np.array([[.2, -.1]])))
    np.testing.assert_allclose(result[:, 0], values, rtol=1e-14)
    path = tmp_path / "nested.keras"
    model.save(path)
    restored = keras.saving.load_model(path)
    np.testing.assert_allclose(_array(restored(packed)), result, atol=1e-14)
    assert len(restored.trainable_variables) == 2


def test_shared_layer_after_primal_dtype_change_roundtrips(tmp_path):
    x = keras.Input((2,), dtype="float64")
    dense = keras.layers.Dense(2, dtype="float64")
    source = keras.Model(x, dense(x) + dense(ops.tanh(x)))
    model = jet_model(source, spec=JetSpec.directional(2))
    packed = seed(np.ones((2, 2)), np.ones((2, 2)), spec=model.spec)
    expected = _array(model(packed))
    path = tmp_path / "shared_policy.keras"
    model.save(path)
    restored = keras.saving.load_model(path)
    assert len(restored.trainable_variables) == 2
    np.testing.assert_allclose(_array(restored(packed)), expected, atol=1e-14)


@pytest.mark.parametrize("approximate", [False, True])
def test_gelu_semantics(approximate):
    x = keras.Input((1,), dtype="float64")
    source = keras.Model(x, ops.gelu(x, approximate=approximate))
    model = jet_model(source, spec=JetSpec.directional(4))
    packed = seed(np.array([[.4]]), np.array([[.8]]), spec=model.spec)
    result = _array(model(packed))[0, :, 0]
    with mp.workdps(60):
        def function(t):
            z = mp.mpf(.4) + mp.mpf(.8) * t
            if approximate:
                coefficient = float(np.float32(.044715)) if keras.backend.backend() == "tensorflow" else .044715
                return z * (1 + mp.tanh(mp.sqrt(2 / mp.pi) * (z + mp.mpf(coefficient) * z**3))) / 2
            if keras.backend.backend() == "tensorflow":
                return z * mp.erfc(-z * mp.mpf(float(np.float32(1 / math.sqrt(2))))) / 2
            return z * (1 + mp.erf(z / mp.sqrt(2))) / 2
        expected = [float(c) for c in mp.taylor(function, 0, 4)]
    if model.explain()["precision_notes"]:
        np.testing.assert_allclose(result, expected, rtol=2e-6, atol=2e-8)
    else:
        np.testing.assert_allclose(result, expected, rtol=2e-12, atol=1e-14)


def test_stable_softmax_and_linear_shape_operations():
    x = keras.Input((2,), dtype="float64")
    value = ops.softmax(ops.concatenate([x, x + 2], axis=-1), axis=-1)
    source = keras.Model(x, ops.sum(ops.reshape(value, (-1, 2, 2)), axis=-1))
    model = jet_model(source, spec=JetSpec.directional(3))
    packed = seed(np.array([[1000., 1001.]]), np.array([[.3, -.2]]), spec=model.spec)
    result = _array(model(packed))
    assert np.isfinite(result).all()
    np.testing.assert_allclose(result[:, 1:], 0., atol=2e-14)


@pytest.mark.parametrize("kind", ["conv1d", "causal", "conv2d"])
def test_convolution_affine_rule(kind):
    shape = (4, 1) if kind != "conv2d" else (3, 3, 1)
    x = keras.Input(shape, dtype="float64")
    layer = (keras.layers.Conv2D(2, 2, dtype="float64") if kind == "conv2d" else
             keras.layers.Conv1D(2, 2, padding="causal" if kind == "causal" else "valid", dtype="float64"))
    source = keras.Model(x, layer(x))
    spec = JetSpec.directional(2)
    model = jet_model(source, spec=spec)
    value = np.arange(math.prod(shape), dtype=np.float64).reshape((1, *shape)) / 10
    direction = np.ones_like(value)
    result = _array(model(seed(value, direction, spec=spec)))
    np.testing.assert_allclose(result[:, 0], _array(source(value)), atol=1e-14)
    np.testing.assert_allclose(result[:, 1], _array(source(direction) - source(np.zeros_like(value))), atol=1e-14)
    np.testing.assert_array_equal(result[:, 2], 0.)


@pytest.mark.parametrize("factory", [lambda x: keras.layers.Dropout(.2)(x),
                                      lambda x: ops.relu(x), lambda x: ops.sum(x)])
def test_unsupported_operations_fail_during_lifting(factory):
    x = keras.Input((2,), dtype="float64")
    with pytest.raises((NotImplementedError, ValueError), match="Unsupported|batch"):
        jet_model(keras.Model(x, factory(x)), spec=JetSpec.directional(2))


def test_custom_rule():
    class Cube(keras.layers.Layer):
        def call(self, x):
            return x**3
    from omnibias.core.jet_series import multiply
    register_jet_rule(Cube, lambda source, args, kwargs, spec: multiply(
        multiply(args[0], args[0], spec, ops), args[0], spec, ops))
    x = keras.Input((1,), dtype="float64")
    model = jet_model(keras.Model(x, Cube(dtype="float64")(x)), spec=JetSpec.directional(3))
    result = _array(model(seed(np.array([[2.]]), np.array([[1.]]), spec=model.spec)))
    np.testing.assert_allclose(result[0, :, 0], [8., 12., 6., 1.], atol=1e-14)


def test_padding_and_batch_axis_refusal():
    x = keras.Input((2, 2), dtype="float64")
    transposed = ops.transpose(x, (0, 2, 1))
    source = keras.Model(x, ops.pad(transposed, ((0, 0), (1, 0), (0, 0)), constant_values=7))
    model = jet_model(source, spec=JetSpec.directional(2))
    result = _array(model(seed(np.ones((1, 2, 2)), np.ones((1, 2, 2)), spec=model.spec)))
    np.testing.assert_array_equal(result[:, 0, 0], 7.)
    np.testing.assert_array_equal(result[:, 1:, 0], 0.)
    np.testing.assert_array_equal(result[:, 1, 1:], 1.)
    with pytest.raises((NotImplementedError, ValueError), match="batch"):
        jet_model(keras.Model(x, ops.transpose(x, (1, 0, 2))), spec=model.spec)


def test_seed_dtype_and_shape_validation():
    spec = JetSpec.directional(2)
    with pytest.raises(TypeError, match="dtype"):
        seed(np.zeros((2, 1), dtype="float64"), np.ones((2, 1), dtype="float32"), spec=spec)
    with pytest.raises(ValueError, match="shape"):
        seed(np.zeros((2, 1)), np.ones((1, 1)), spec=spec)


def test_order32_derivative_extraction_uses_float_scale():
    spec = JetSpec.directional(32)
    result = extract(ops.ones((1, 33, 1), dtype="float64"), 32, spec=spec, derivative=True)
    np.testing.assert_allclose(_array(result), float(math.factorial(32)), rtol=1e-15)


@pytest.mark.parametrize("factory", [
    lambda x: ops.take(x, [1, 0], axis=0),
    lambda x: ops.moveaxis(x, 0, 1),
    lambda x: ops.stack([x, x], axis=0),
    lambda x: ops.concatenate([x, x], axis=0),
    lambda x: keras.layers.Concatenate(axis=0)([x, x]),
    lambda x: ops.flip(x, axis=0),
])
def test_equal_sized_axes_do_not_hide_batch_reindexing(factory):
    x = keras.Input(batch_shape=(2, 2), dtype="float64")
    with pytest.raises((NotImplementedError, ValueError), match="batch"):
        jet_model(keras.Model(x, factory(x)), spec=JetSpec.directional(2))


def test_float32_dtype_is_retained():
    x = keras.Input((1,), dtype="float32")
    source = keras.Model(x, ops.exp(x / 2) + x * x)
    model = jet_model(source, spec=JetSpec.directional(3))
    packed = seed(np.array([[.2]], dtype="float32"), np.array([[.4]], dtype="float32"), spec=model.spec)
    result = model(packed)
    assert keras.backend.standardize_dtype(result.dtype) == "float32"
    np.testing.assert_allclose(_array(result)[:, 0], _array(source(np.array([[.2]], dtype="float32"))), rtol=1e-7)


def test_tensor_constant_matmul_and_multi_input_roundtrip(tmp_path):
    x = keras.Input((2,), dtype="float64", name="x")
    y = keras.Input((2,), dtype="float64", name="y")
    matrix = np.array([[.2, .3], [.4, -.1]])
    source = keras.Model({"x": x, "y": y}, {"product": ops.matmul(x, matrix) * y, "sum": x + y})
    spec = JetSpec.directional(2)
    model = jet_model(source, spec=spec)
    a, b = np.array([[.2, -.3]]), np.array([[.7, .4]])
    da, db = np.array([[.1, .2]]), np.array([[.3, .5]])
    inputs = {"x": seed(a, da, spec=spec), "y": seed(b, db, spec=spec)}
    result = model(inputs)
    np.testing.assert_allclose(_array(result["product"])[:, 2], (da @ matrix) * db, atol=1e-14)
    path = tmp_path / "multi.keras"
    model.save(path)
    loaded = keras.saving.load_model(path)
    for key, tensor in result.items():
        np.testing.assert_allclose(_array(loaded(inputs)[key]), _array(tensor), atol=1e-14)


@pytest.mark.parametrize("merge", [lambda values: ops.concatenate(values, axis=-1),
                                   lambda values: ops.stack(values, axis=-1),
                                   lambda values: keras.layers.Concatenate(axis=-1)(values)])
def test_affine_merge_constant_columns_have_zero_higher_coefficients(merge):
    x = keras.Input(batch_shape=(2, 1), dtype="float64")
    constant = np.full((2, 1), 7., dtype="float64")
    model = jet_model(keras.Model(x, merge([x, constant])), spec=JetSpec.directional(2))
    packed = seed(np.array([[.2], [.3]]), np.ones((2, 1)), spec=model.spec)
    result = _array(model(packed))
    np.testing.assert_array_equal(result[:, 0, ..., -1], 7.)
    np.testing.assert_array_equal(result[:, 1:, ..., -1], 0.)


@pytest.mark.parametrize("backward", ["native", "analytic"])
def test_torch_softplus_threshold_source_semantics(backward):
    if keras.backend.backend() != "torch":
        pytest.skip("Thresholded identity branch belongs to the Torch source primitive.")
    x = keras.Input((1,), dtype="float64")
    model = jet_model(keras.Model(x, ops.softplus(x)), spec=JetSpec.directional(3), backward=backward)
    packed = np.array([[[21.], [.7], [.2], [-.1]]])
    np.testing.assert_array_equal(_array(model(packed)), packed)


@pytest.mark.parametrize("backward", ["native", "analytic"])
@pytest.mark.filterwarnings(
    # Torch 2.14's Dynamo inspects .grad while classifying Keras loss tensors.
    # This upstream introspection warning is unrelated to gradient correctness.
    r"ignore:The .grad attribute of a Tensor that is not a leaf Tensor is being accessed.*:UserWarning:torch\..*"
)
@pytest.mark.filterwarnings(
    # Keras' own pure dtype-promotion metadata cache is inlined by Dynamo.
    r"ignore:Dynamo detected a call to a `functools\.lru_cache`-wrapped function at 'dtypes\.py.*:UserWarning"
)
def test_compiled_training_and_optimizer_roundtrip(tmp_path, backward):
    x = keras.Input((1,), dtype="float64")
    dense = keras.layers.Dense(1, activation="sigmoid", dtype="float64")
    model = jet_model(keras.Model(x, dense(x)), spec=JetSpec.directional(2), backward=backward)
    packed = seed(np.array([[.2], [-.1]]), np.ones((2, 1)), spec=model.spec)
    target = np.zeros((2, 3, 1))
    model.compile(optimizer=keras.optimizers.Adam(.001), loss="mse", jit_compile=True)
    loss = model.train_on_batch(packed, target)
    assert np.isfinite(loss)
    path = tmp_path / "compiled_jet.keras"
    model.save(path)
    restored = keras.saving.load_model(path)
    assert int(_array(restored.optimizer.iterations)) == 1
    np.testing.assert_allclose(_array(restored(packed)), _array(model(packed)), rtol=1e-12, atol=1e-14)
    assert np.isfinite(restored.train_on_batch(packed, target))
    assert int(_array(restored.optimizer.iterations)) == 2


def test_analytic_parameter_second_gradient_matches_native():
    if keras.backend.backend() != "torch":
        pytest.skip("Torch grad-grad regression; ordinary gradients covered on every backend.")
    import torch
    class ScalarScale(keras.layers.Layer):
        def build(self, shape):
            self.factor = self.add_weight(shape=(), initializer=keras.initializers.Constant(.7))

        def call(self, value):
            return value * self.factor

    # Avoid Keras/Torch's float64 matmul -> float32 promotion so this is a
    # strict float64 test of the analytic transpose rule and its second AD.
    register_jet_rule(ScalarScale, lambda source, args, kwargs, spec:
                      [v * source.factor for v in args[0]])
    x = keras.Input((1,), dtype="float64")
    scale = ScalarScale(dtype="float64")
    source = keras.Model(x, ops.sigmoid(scale(x)))
    spec = JetSpec.directional(3)
    packed = seed(np.array([[.2]]), np.array([[.4]]), spec=spec)
    answers = []
    for backward in ("native", "analytic"):
        model = jet_model(source, spec=spec, backward=backward)
        loss = model(packed)[:, 3].square().sum()
        first = torch.autograd.grad(loss, scale.factor.value, create_graph=True)[0]
        second = torch.autograd.grad(first.sum(), scale.factor.value)[0]
        answers.append((first.detach().numpy(), second.detach().numpy()))
    for native, analytic in zip(answers[0], answers[1], strict=True):
        np.testing.assert_allclose(analytic, native, rtol=2e-11, atol=1e-14)
