# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Independent values, structural refusals and training for operator models."""

from __future__ import annotations

from fractions import Fraction
from itertools import product
from math import factorial

import keras
import numpy as np
import pytest
from keras import ops
from omnibias.core.operator_spec import OperatorPlanningBudget
from omnibias.keras.experimental import (
    JetBudget,
    JetSpec,
    OperatorSpec,
    operator_model,
    register_jet_rule,
)


def array(value):
    return ops.convert_to_numpy(value)


@pytest.mark.parametrize("order", [4, 24, 32])
@pytest.mark.parametrize("outer", [False, True])
def test_high_order_structural_proof_precedes_mixed_layout(order, outer):
    x = keras.Input((order,), dtype="float64")
    value = x[:, :1]
    for i in range(1, order):
        value = value + x[:, i:i + 1]
    output = ops.exp(value) + value**3 if outer else ops.exp(value)
    source = keras.Model(x, output)
    operator = OperatorSpec.mixed((1,) * order, budget=JetBudget(max_coefficients=1200))
    model = operator_model(source, operator=operator)
    assert model.explain()["selected_strategy"] == ("scalar_cut" if outer else "read_once")
    assert model.explain()["coefficient_count"] == order + 1
    points = np.full((2, order), .01)
    np.testing.assert_allclose(array(model(points)), np.exp(.01 * order), rtol=3e-12, atol=1e-13)
    if order >= 24:
        with pytest.raises(ValueError, match="budget|coefficients"):
            operator_model(source, operator=operator, strategy="reference")


@pytest.mark.parametrize("strategy", ["auto", "reference"])
def test_mixed_product_repeated_orders_and_zero_safe(strategy):
    x = keras.Input((2,), dtype="float64")
    a, b = x[:, :1], x[:, 1:]
    source = keras.Model(x, ops.exp(a * b))
    model = operator_model(source, operator=OperatorSpec.mixed((2, 2)), strategy=strategy)
    points = np.array([[.2, -.3], [0., 0.], [0., .4]])
    product = points[:, :1] * points[:, 1:]
    expected = np.exp(product) * (product**2 + 4 * product + 2)
    np.testing.assert_allclose(array(model(points)), expected, rtol=2e-13, atol=1e-14)


def test_inactive_coordinates_and_polynomial_zero_order():
    x = keras.Input((2,), dtype="float64")
    source = keras.Model(x, ops.exp(x[:, :1]))
    points = np.array([[.2, .3]])
    missing = operator_model(source, operator=OperatorSpec.mixed((0, 1)))
    np.testing.assert_array_equal(array(missing(points)), [[0.]])
    operator = OperatorSpec.polynomial({(0, 0): Fraction(3, 2), (2, 0): -2, (0, 1): 7})
    model = operator_model(source, operator=operator)
    np.testing.assert_allclose(array(model(points)), -.5 * np.exp(.2), rtol=1e-13)


@pytest.mark.parametrize("kind", ["mixed", "directional", "laplacian", "polynomial"])
def test_ridge_contractions_against_exponential_formula(kind):
    x = keras.Input((3,), dtype="float64")
    dense = keras.layers.Dense(2, dtype="float64")
    head = keras.layers.Dense(1, dtype="float64")
    source = keras.Model(x, head(ops.exp(dense(x))))
    weight = np.array([[.3, -.2], [.4, .1], [-.5, .6]])
    bias, readout = np.array([.1, -.2]), np.array([[.7], [-.3]])
    dense.kernel.assign(weight)
    dense.bias.assign(bias)
    head.kernel.assign(readout)
    head.bias.assign([.25])
    points = np.array([[.2, -.3, .1], [.4, .1, -.2]])
    directions = np.array([[.3, -.1, .2], [.1, .2, -.4]])
    if kind == "mixed":
        operator = OperatorSpec.mixed((2, 1, 1))
        factor = weight[0]**2 * weight[1] * weight[2]
    elif kind == "directional":
        operator = OperatorSpec.directional(4)
        factor = (directions @ weight)**4
    elif kind == "laplacian":
        operator = OperatorSpec.laplacian(power=2)
        factor = (weight**2).sum(axis=0)**2
    else:
        operator = OperatorSpec.polynomial({(2, 0, 0): 2, (0, 0, 0): Fraction(1, 3)})
        factor = 2 * weight[0]**2 + 1 / 3
    model = operator_model(source, operator=operator)
    assert model.explain()["selected_strategy"] == "ridge"
    assert all(a is b for a, b in zip(model.trainable_variables, source.trainable_variables, strict=True))
    inputs = [points, directions] if kind == "directional" else points
    expected = (np.exp(points @ weight + bias) * factor) @ readout
    if kind == "polynomial":
        expected += .25 / 3
    # Keras 3.15 Torch ops.matmul explicitly uses float32 for float64 operands.
    tolerance = 8e-7 if keras.backend.backend() == "torch" else 3e-12
    np.testing.assert_allclose(array(model(inputs)), expected, rtol=tolerance, atol=2e-9 if tolerance > 1e-8 else 1e-14)


def test_wide_ridge_laplacian_does_not_expand_symbol():
    x = keras.Input((1024,), dtype="float64")
    dense = keras.layers.Dense(2, dtype="float64", use_bias=False,
                               kernel_initializer=keras.initializers.Constant(.01))
    source = keras.Model(x, ops.exp(dense(x)))
    operator = OperatorSpec.laplacian(power=3, budget=JetBudget(max_coefficients=100))
    model = operator_model(source, operator=operator)
    assert model.explain()["selected_strategy"] == "ridge"
    np.testing.assert_allclose(array(model(np.zeros((1, 1024)))), (.01**2 * 1024)**3, rtol=5e-7)


@pytest.mark.parametrize("method", ["recurrence", "polynomial"])
def test_activation_selection_and_directional_input(method):
    x = keras.Input((1,), dtype="float64")
    source = keras.Model(x, ops.sigmoid(x))
    model = operator_model(source, operator=OperatorSpec.directional(3), activation_method=method)
    points, directions = np.array([[.2]]), np.array([[.7]])
    s = 1 / (1 + np.exp(-points))
    expected = s * (1 - s) * (1 - 6 * s + 6 * s**2) * directions**3
    np.testing.assert_allclose(array(model([points, directions])), expected, rtol=3e-12, atol=1e-14)


def test_overlap_falls_back_and_batch_reindex_is_rejected():
    x = keras.Input((2,), dtype="float64")
    a, b = x[:, :1], x[:, 1:]
    source = keras.Model(x, ops.exp(a * b + a))
    model = operator_model(source, operator=OperatorSpec.mixed((1, 1)))
    assert model.explain()["selected_strategy"] == "reference"
    points = np.array([[.2, -.3]])
    expected = np.exp(.2 * (.7)) * (1 + .2 * .7)
    np.testing.assert_allclose(array(model(points)), expected, rtol=2e-13)
    batch_mixing = keras.Model(x, ops.sum(x, axis=0))
    with pytest.raises(NotImplementedError, match="batch"):
        operator_model(batch_mixing, operator=OperatorSpec.mixed((1, 0)))


@pytest.mark.parametrize("strategy", ["auto", "reference"])
def test_shared_nested_layers_training_and_optimizer_roundtrip(tmp_path, strategy):
    inner = keras.Sequential([keras.Input((1,), dtype="float64"),
                              keras.layers.Dense(1, activation="sigmoid", dtype="float64")])
    x = keras.Input((1,), dtype="float64")
    source = keras.Model(x, inner(x) + inner(x))
    source.layers[1].layers[0].kernel.assign([[.7]])
    model = operator_model(source, operator=OperatorSpec.mixed((2,)), strategy=strategy)
    assert len(model.trainable_variables) == 2
    assert all(a is b for a, b in zip(model.trainable_variables, source.trainable_variables, strict=True))
    points = np.array([[.2], [-.3]])
    model.compile(optimizer=keras.optimizers.Adam(.001), loss="mse", jit_compile=False)
    before = np.array(source.trainable_variables[0].numpy(), copy=True)
    model.train_on_batch(points, np.zeros_like(points))
    assert not np.array_equal(before, source.trainable_variables[0].numpy())
    expected = array(model(points))
    path = tmp_path / "operator.keras"
    model.save(path)
    restored = keras.saving.load_model(path)
    assert len(restored.trainable_variables) == 2
    np.testing.assert_allclose(array(restored(points)), expected, rtol=1e-12, atol=1e-14)
    assert int(array(restored.optimizer.iterations)) == 1
    restored.train_on_batch(points, np.zeros_like(points))
    assert int(array(restored.optimizer.iterations)) == 2


@pytest.mark.parametrize("strategy", ["auto", "reference"])
def test_ridge_save_and_parameter_gradient_formula(tmp_path, strategy):
    x = keras.Input((1,), dtype="float64")
    dense = keras.layers.Dense(1, activation="sigmoid", dtype="float64")
    source = keras.Model(x, dense(x))
    dense.kernel.assign([[.7]])
    dense.bias.assign([.1])
    model = operator_model(source, operator=OperatorSpec.mixed((2,)), strategy=strategy)
    points = np.array([[.2], [-.3]])
    rate = 1e-4
    model.compile(optimizer=keras.optimizers.SGD(rate), loss="mse", jit_compile=False)
    model.train_on_batch(points, np.zeros_like(points))
    measured = (.7 - float(dense.kernel.numpy()[0, 0])) / rate
    def loss(weight):
        s = 1 / (1 + np.exp(-(weight * points + .1)))
        derivative = s * (1 - s) * (1 - 2 * s) * weight**2
        return np.mean(derivative**2)
    step = 1e-5
    expected = (loss(.7 + step) - loss(.7 - step)) / (2 * step)
    assert measured == pytest.approx(expected, rel=3e-6, abs=1e-10)
    path = tmp_path / "ridge.keras"
    model.save(path)
    restored = keras.saving.load_model(path)
    np.testing.assert_allclose(array(restored(points)), array(model(points)), rtol=1e-12)


def test_sequential_ridge_and_invalid_interface():
    source = keras.Sequential([keras.Input((2,), dtype="float64"),
                               keras.layers.Dense(3, activation="sigmoid", dtype="float64"),
                               keras.layers.Dense(1, dtype="float64")])
    model = operator_model(source, operator=OperatorSpec.directional(2))
    assert model.explain()["selected_strategy"] == "ridge"
    assert array(model([np.ones((2, 2)), np.ones((2, 2))])).shape == (2, 1)
    with pytest.raises(ValueError, match="activation_method"):
        operator_model(source, operator=OperatorSpec.directional(2), activation_method="unknown")


@pytest.mark.parametrize("order", [24, 32])
@pytest.mark.parametrize("reduction", ["sum", "mean"])
def test_vectorized_separable_scalar_cut(order, reduction):
    x = keras.Input((order,), dtype="float64")
    source = keras.Model(x, ops.sin(getattr(ops, reduction)(ops.tanh(x * .8 + .1), axis=-1)))
    model = operator_model(source, operator=OperatorSpec.mixed((1,) * order,
                           budget=JetBudget(max_coefficients=1200)))
    assert model.explain()["selected_strategy"] == "separable"
    points = np.full((2, order), .02)
    t = array(ops.tanh(ops.convert_to_tensor(points * .8 + .1, dtype="float64")))
    derivative = .8 * (1 - t.astype("float64")**2)
    if reduction == "mean":
        derivative /= order
    # n is divisible by four, so the outer derivative equals its source
    # primal. This also preserves Keras mean's backend-specific accumulation.
    expected = array(source(points)).astype("float64") * np.prod(derivative, axis=-1)
    # These Keras sin/tanh kernels may compute float32 despite float64 inputs.
    # Bound that measured primitive precision, including a length-32 product.
    tolerance = 8e-6 if t.dtype == np.float32 else 3e-11
    np.testing.assert_allclose(array(model(points)), expected, rtol=tolerance, atol=0)


@pytest.mark.parametrize("strategy", ["auto", "reference"])
@pytest.mark.parametrize("kind", ["reshape", "softmax", "transpose", "slice"])
def test_all_strategies_reject_batch_mixing(strategy, kind):
    x = keras.Input(batch_shape=(2, 2), dtype="float64")
    if kind == "reshape":
        y = ops.reshape(x, (1, 4))
    elif kind == "softmax":
        y = ops.softmax(x, axis=0)
    elif kind == "transpose":
        y = ops.transpose(x)
    else:
        y = x[1:, :]
    source = keras.Model(x, y)
    with pytest.raises(NotImplementedError, match="batch"):
        operator_model(source, operator=OperatorSpec.mixed((1, 1)), strategy=strategy)


@pytest.mark.parametrize("kind", ["mixed", "laplacian", "directional", "polynomial"])
def test_affine_bottleneck_runtime_weight_transport(kind):
    x = keras.Input((4,), dtype="float64")
    dense = keras.layers.Dense(1, dtype="float64")
    latent = dense(x)
    source = keras.Model(x, ops.exp(latent * latent))
    weights = np.array([[.3], [-.2], [.4], [.1]])
    dense.kernel.assign(weights)
    dense.bias.assign([.1])
    if kind == "mixed":
        operator = OperatorSpec.mixed((1, 1, 1, 1))
        factor = weights.prod()
    elif kind == "laplacian":
        operator = OperatorSpec.laplacian(2)
        factor = float((weights**2).sum())**2
    elif kind == "directional":
        operator = OperatorSpec.directional(4)
        factor = float(weights.sum())**4
    else:
        operator = OperatorSpec.polynomial({(1, 1, 1, 1): 3, (0, 0, 0, 0): 2})
        factor = 3 * weights.prod()
    model = operator_model(source, operator=operator)
    assert model.explain()["selected_strategy"] == "affine_bottleneck"
    assert model.explain()["coefficient_count"] == 5
    points = np.array([[.2, -.1, .3, .4]])
    z = points @ weights + .1
    fourth = np.exp(z*z) * (12 + 48*z*z + 16*z**4)
    expected = fourth * factor
    if kind == "polynomial":
        expected += 2 * np.exp(z*z)
    inputs = [points, np.ones_like(points)] if kind == "directional" else points
    tolerance = 1e-6 if keras.backend.backend() == "torch" else 1e-12
    np.testing.assert_allclose(array(model(inputs)), expected, rtol=tolerance, atol=1e-13)


def test_bottleneck_transport_retains_parameter_gradients(tmp_path):
    x = keras.Input((3,), dtype="float64")
    dense = keras.layers.Dense(1, dtype="float64", use_bias=False)
    z = dense(x)
    model = operator_model(keras.Model(x, ops.exp(z*z)), operator=OperatorSpec.laplacian(2))
    weights = np.array([[.3], [-.2], [.4]])
    dense.kernel.assign(weights)
    points = np.array([[.2, -.1, .3]])
    rate = 1e-6
    model.compile(optimizer=keras.optimizers.SGD(rate), loss="mse", jit_compile=False)
    model.train_on_batch(points, np.zeros((1, 1)))
    measured = (weights - dense.kernel.numpy()) / rate
    def loss(weight):
        z = points @ weight
        value = np.exp(z*z) * (12 + 48*z*z + 16*z**4) * ((weight**2).sum())**2
        return np.mean(value**2)
    expected = np.zeros_like(weights)
    for index in range(3):
        plus, minus = weights.copy(), weights.copy()
        plus[index] += 1e-5
        minus[index] -= 1e-5
        expected[index] = (loss(plus) - loss(minus)) / 2e-5
    np.testing.assert_allclose(measured, expected, rtol=2e-6, atol=1e-8)
    path = tmp_path / "transport.keras"
    model.save(path)
    restored = keras.saving.load_model(path)
    np.testing.assert_allclose(array(restored(points)), array(model(points)), rtol=1e-12)


@pytest.mark.parametrize("strategy", ["auto", "reference"])
@pytest.mark.filterwarnings(r"ignore:The .grad attribute of a Tensor that is not a leaf Tensor is being accessed.*:UserWarning:torch\..*")
@pytest.mark.filterwarnings(r"ignore:Dynamo detected a call to a `functools\.lru_cache`-wrapped function at 'dtypes\.py.*:UserWarning")
def test_compiled_operator_training_continuation(tmp_path, strategy):
    x = keras.Input((1,), dtype="float64")
    dense = keras.layers.Dense(1, activation="sigmoid", dtype="float64")
    model = operator_model(keras.Model(x, dense(x)), operator=OperatorSpec.mixed((2,)), strategy=strategy)
    points = np.array([[.2], [-.3]])
    model.compile(optimizer=keras.optimizers.Adam(.001), loss="mse", jit_compile=True)
    assert np.isfinite(model.train_on_batch(points, np.zeros_like(points)))
    path = tmp_path / "compiled.keras"
    model.save(path)
    restored = keras.saving.load_model(path)
    assert int(array(restored.optimizer.iterations)) == 1
    assert np.isfinite(restored.train_on_batch(points, np.zeros_like(points)))
    assert int(array(restored.optimizer.iterations)) == 2


@pytest.mark.parametrize("power", [1, 2])
def test_weighted_apolar_fallback_and_independent_polynomial(power):
    x = keras.Input((2,), dtype="float64")
    a, b = x[:, :1], x[:, 1:]
    source = keras.Model(x, ops.exp(a*a + 3*a*b + 2*b*b))
    model = operator_model(source, operator=OperatorSpec.laplacian(power))
    assert model.explain()["selected_strategy"] == "apolar"
    expected = 6 if power == 1 else 112
    np.testing.assert_allclose(array(model(np.zeros((1, 2)))), expected, rtol=2e-13)
    reference = operator_model(source, operator=OperatorSpec.laplacian(power), strategy="reference")
    points = np.array([[.2, -.1], [.1, .3]])
    np.testing.assert_allclose(array(model(points)), array(reference(points)), rtol=5e-13)


def test_inhomogeneous_readout_preserves_final_affine_bias():
    x = keras.Input((2,), dtype="float64")
    first = keras.layers.Dense(2, activation="sigmoid", dtype="float64")
    last = keras.layers.Dense(1, dtype="float64", bias_initializer=keras.initializers.Constant(3.))
    source = keras.Model(x, last(first(x)))
    points = np.array([[.2, -.1]])
    for operator in [OperatorSpec.polynomial({(0, 0): 2}), OperatorSpec.laplacian(0)]:
        model = operator_model(source, operator=operator)
        scale = 2 if operator.mode == "polynomial" else 1
        np.testing.assert_allclose(array(model(points)), scale * array(source(points)), rtol=2e-13)


@pytest.mark.filterwarnings(r"ignore:The .grad attribute of a Tensor that is not a leaf Tensor is being accessed.*:UserWarning:torch\..*")
@pytest.mark.filterwarnings(r"ignore:Dynamo detected a call to a `functools\.lru_cache`-wrapped function at 'dtypes\.py.*:UserWarning")
def test_apolar_compiled_parameter_gradient_and_save(tmp_path):
    x = keras.Input((2,), dtype="float64")
    dense = keras.layers.Dense(2, use_bias=False, dtype="float64")
    z = dense(x)
    a, b = z[:, :1], z[:, 1:]
    source = keras.Model(x, ops.exp(a*a + 3*a*b + 2*b*b))
    model = operator_model(source, operator=OperatorSpec.laplacian(2))
    assert model.explain()["selected_strategy"] == "apolar"
    weight = np.array([[.3, -.2], [.4, .1]])
    dense.kernel.assign(weight)
    rate = 1e-6
    model.compile(optimizer=keras.optimizers.SGD(rate), loss="mse", jit_compile=True)
    model.train_on_batch(np.zeros((1, 2)), np.zeros((1, 1)))
    measured = (weight - dense.kernel.numpy()) / rate
    def loss(w):
        matrix = w @ np.array([[1., 1.5], [1.5, 2.]]) @ w.T
        return (4 * np.trace(matrix)**2 + 8 * np.trace(matrix @ matrix))**2
    expected = np.zeros_like(weight)
    for i in range(2):
        for j in range(2):
            plus, minus = weight.copy(), weight.copy()
            plus[i, j] += 1e-5
            minus[i, j] -= 1e-5
            expected[i, j] = (loss(plus) - loss(minus)) / 2e-5
    np.testing.assert_allclose(measured, expected, rtol=3e-6, atol=1e-8)
    path = tmp_path / "apolar.keras"
    model.save(path)
    restored = keras.saving.load_model(path)
    np.testing.assert_allclose(array(restored(np.zeros((1, 2)))), array(model(np.zeros((1, 2)))), rtol=1e-13)
    assert np.isfinite(restored.train_on_batch(np.zeros((1, 2)), np.zeros((1, 1))))


def test_polynomial_provider_reference_and_shape_dtype_checks():
    x = keras.Input((2,), dtype="float64")
    a, b = x[:, :1], x[:, 1:]
    source = keras.Model(x, ops.sigmoid(a*b + a))
    points = np.array([[.2, -.3]])
    operator = OperatorSpec.mixed((2, 1))
    recurrence = operator_model(source, operator=operator, strategy="reference")
    polynomial = operator_model(source, operator=operator, strategy="reference", activation_method="polynomial")
    np.testing.assert_allclose(array(polynomial(points)), array(recurrence(points)), rtol=2e-13, atol=1e-14)
    directional = operator_model(source, operator=OperatorSpec.directional(2))
    with pytest.raises(TypeError, match="dtype"):
        directional([ops.ones((1, 2), dtype="float64"), ops.ones((1, 2), dtype="float32")])


@pytest.mark.parametrize("strategy", ["auto", "reference"])
def test_equal_size_broadcast_cannot_hide_batch_coupling(strategy):
    x = keras.Input(batch_shape=(2, 2), dtype="float64")
    source = keras.Model(x, x + ops.sum(x, axis=-1))
    with pytest.raises(NotImplementedError, match="batch"):
        operator_model(source, operator=OperatorSpec.mixed((1, 1)), strategy=strategy)


def test_ellipsis_cannot_hide_batch_indexing():
    x = keras.Input(batch_shape=(2, 2), dtype="float64")
    source = keras.Model(x, x[..., 0, :])
    with pytest.raises(NotImplementedError, match="batch"):
        operator_model(source, operator=OperatorSpec.mixed((1, 1)))


def test_apolar_selection_precedes_reference_layout(monkeypatch):
    x = keras.Input((2,), dtype="float64")
    a, b = x[:, :1], x[:, 1:]
    source = keras.Model(x, ops.exp(a*a + 3*a*b + 2*b*b))
    monkeypatch.setattr(JetSpec, "mixed", lambda *args, **kwargs: pytest.fail("allocated reference mixed layout"))
    model = operator_model(source, operator=OperatorSpec.laplacian(2))
    assert model.explain()["selected_strategy"] == "apolar"
    np.testing.assert_allclose(array(model(np.zeros((1, 2)))), 112, rtol=1e-13)


def test_custom_class_name_cannot_impersonate_builtin_rule():
    class Sin(keras.layers.Layer):
        def call(self, value):
            return value * value
    x = keras.Input((1,), dtype="float64")
    source = keras.Model(x, Sin(dtype="float64")(x))
    with pytest.raises(NotImplementedError, match="registered"):
        operator_model(source, operator=OperatorSpec.mixed((1,)))
    from omnibias.core.jet_series import multiply
    register_jet_rule(Sin, lambda source, args, kwargs, spec: multiply(args[0], args[0], spec, ops))
    model = operator_model(source, operator=OperatorSpec.mixed((1,)))
    assert model.explain()["selected_strategy"] == "reference"
    np.testing.assert_allclose(array(model(np.array([[.3]]))), .6, atol=1e-14)


def test_smaller_apolar_plan_survives_reference_product_budget():
    terms = {alpha: factorial(7) // np.prod([factorial(value) for value in alpha])
             for alpha in product(range(8), repeat=4) if sum(alpha) == 7}
    terms = {alpha: int(coefficient) for alpha, coefficient in terms.items()}
    operator = OperatorSpec.polynomial(terms, budget=JetBudget(max_multiplication_terms=3000))
    x = keras.Input((4,), dtype="float64")
    a, b, c, d = [x[:, i:i + 1] for i in range(4)]
    source = keras.Model(x, ops.exp(a*b + a*c + b*d))
    model = operator_model(source, operator=operator)
    assert model.explain()["selected_strategy"] == "apolar"
    assert model.explain()["coefficient_count"] == 8
    points = np.array([[.1, -.2, .3, .4]])
    expected = np.exp(-.07) * factorial(7) * sum(.5**(7 - 2*j) * 3**j /
                                               (factorial(7 - 2*j) * factorial(j)) for j in range(4))
    np.testing.assert_allclose(array(model(points)), expected, rtol=2e-12)
    with pytest.raises(ValueError, match="budget"):
        operator_model(source, operator=operator, strategy="reference")


@pytest.mark.parametrize("strategy", ["auto", "reference"])
def test_derivative_only_training_keeps_zero_final_bias_gradient(strategy):
    source = keras.Sequential([keras.Input((2,), dtype="float64"),
                               keras.layers.Dense(3, activation="sigmoid", dtype="float64"),
                               keras.layers.Dense(1, bias_initializer="ones", dtype="float64")])
    model = operator_model(source, operator=OperatorSpec.mixed((1, 1)), strategy=strategy)
    model.compile(optimizer=keras.optimizers.SGD(.001), loss="mse", jit_compile=False)
    model.train_on_batch(np.zeros((1, 2)), np.zeros((1, 1)))
    np.testing.assert_array_equal(source.layers[-1].bias.numpy(), [1.])


def test_aggregate_schedule_budget_and_scope():
    x = keras.Input((1,), dtype="float64")
    source = keras.Model(x, ops.exp(x))
    operator = OperatorSpec.mixed((4,), planning_budget=OperatorPlanningBudget(max_planning_work=100))
    with pytest.raises(ValueError, match="planning"):
        operator_model(source, operator=operator)
    model = operator_model(source, operator=OperatorSpec.mixed((4,)))
    assert model.explain()["estimated_schedule_work"] > 100
    assert "backend/compiler" in model.explain()["schedule_work_scope"]


@pytest.mark.parametrize("coefficient", [Fraction(10**400), Fraction(1, 10**400)])
def test_unrepresentable_operator_coefficient_rejected(coefficient):
    x = keras.Input((1,), dtype="float64")
    source = keras.Model(x, ops.exp(x))
    with pytest.raises(ValueError, match="floating-point"):
        operator_model(source, operator=OperatorSpec.polynomial({(1,): coefficient}))
