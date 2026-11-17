# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Compare frontends and parameter pullbacks in isolated backend processes."""

import importlib.util
import json
import os
import subprocess
import sys

import numpy as np
import pytest

SCRIPT = r'''
import json
import os
import numpy as np
from omnibias.core.jet_algebra import JetSpec

x = np.array([[0.2, -0.3], [-0.1, 0.4]], dtype=np.float64)
w = np.array([[0.4, -0.2, 0.1], [0.3, 0.5, -0.4]], dtype=np.float64)
b = np.array([0.1, -0.1, 0.2], dtype=np.float64)
d0 = np.array([[0.6, -0.2], [0.1, 0.3]], dtype=np.float64)
d1 = np.array([[0.2, 0.1], [-0.3, 0.4]], dtype=np.float64)
specs = [JetSpec.directional(4), JetSpec.mixed([(2, 1)])]
result = []
if os.environ['JET_FRONTEND'] == 'torch':
    import torch
    from omnibias.torch.experimental import compile_jet, seed_jet
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(w))
            self.bias = torch.nn.Parameter(torch.tensor(b))
        def forward(self, x):
            z = x @ self.weight + self.bias
            y = torch.nn.functional.gelu(z, approximate='none')
            return (y*y + torch.exp(0.25*z)).sum(-1, keepdim=True)
    for spec in specs:
        model = Model()
        point = torch.tensor(x)
        directions = {(1,): torch.tensor(d0)} if spec.dim == 1 else {
            (1, 0): torch.tensor(d0), (0, 1): torch.tensor(d1)}
        program = compile_jet(model, (point,), spec=spec)
        output = program(seed_jet(point, spec, directions))
        loss = output.square().sum()
        gradients = torch.autograd.grad(loss, tuple(model.parameters()))
        result.append({'output': output.detach().numpy().tolist(),
                       'gradients': [g.detach().numpy().tolist() for g in gradients]})
else:
    import keras
    from keras import ops
    from omnibias.keras.experimental import jet_model, seed
    for spec in specs:
        inputs = keras.Input((2,), dtype='float64')
        dense = keras.layers.Dense(3, dtype='float64')
        z = dense(inputs)
        y = ops.gelu(z, approximate=False)
        model = keras.Model(inputs, ops.sum(y*y + ops.exp(0.25*z), axis=-1, keepdims=True))
        dense.set_weights([w, b])
        lifted = jet_model(model, spec=spec)
        point = ops.convert_to_tensor(x)
        directions = ops.convert_to_tensor(d0) if spec.dim == 1 else [
            ops.convert_to_tensor(d0), ops.convert_to_tensor(d1)]
        packed = seed(point, directions, spec=spec)
        lifted(packed)  # Build before Keras' stateless parameter-gradient API.
        if keras.backend.backend() == 'torch':
            import torch
            output = lifted(packed)
            gradients = torch.autograd.grad(ops.sum(output*output),
                                            [v.value for v in lifted.trainable_variables])
        elif keras.backend.backend() == 'jax':
            import jax
            variables = [v.value for v in lifted.trainable_variables]
            def loss(parameters):
                output, _ = lifted.stateless_call(parameters, [], packed)
                return ops.sum(output*output)
            gradients = jax.grad(loss)(variables)
            output = lifted(packed)
        else:
            import tensorflow as tf
            with tf.GradientTape() as tape:
                output = lifted(packed)
                loss = ops.sum(output*output)
            gradients = tape.gradient(loss, lifted.trainable_variables)
        result.append({'output': np.moveaxis(ops.convert_to_numpy(output), 1, 0).tolist(),
                       'gradients': [ops.convert_to_numpy(g).tolist() for g in gradients]})
print('JET_PARITY=' + json.dumps(result))
'''


def _run(frontend, backend):
    environment = dict(os.environ, KERAS_BACKEND=backend, JAX_ENABLE_X64="1",
                       CUDA_VISIBLE_DEVICES="", JET_FRONTEND=frontend,
                       OMP_NUM_THREADS="1")
    result = subprocess.run([sys.executable, "-c", SCRIPT], env=environment,
                            text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stderr
    line = next(line for line in result.stdout.splitlines() if line.startswith("JET_PARITY="))
    return json.loads(line.removeprefix("JET_PARITY="))


@pytest.fixture(scope="module")
def torch_reference():
    pytest.importorskip("torch")
    return _run("torch", "torch")


@pytest.mark.parametrize("backend", ["torch", "jax", "tensorflow"])
def test_compiler_frontends_agree_on_values_and_parameter_pullbacks(backend, torch_reference):
    if importlib.util.find_spec("keras") is None or importlib.util.find_spec(backend) is None:
        pytest.skip(f"Keras/{backend} not installed")
    # The source primitives themselves differ: Keras 3.15/Torch result_type
    # demotes matmul's float64 operands to float32; TensorFlow GELU uses
    # float32-rounded constants. The frontends must preserve these semantics.
    # Core mathematical recurrences have separate 80+ digit reference gates.
    rtol, atol = {"torch": (8e-7, 2e-8), "tensorflow": (3e-7, 2e-9),
                  "jax": (3e-11, 3e-13)}[backend]
    for reference, actual in zip(torch_reference, _run("keras", backend), strict=True):
        np.testing.assert_allclose(actual["output"], reference["output"], rtol=rtol, atol=atol)
        for got, wanted in zip(actual["gradients"], reference["gradients"], strict=True):
            np.testing.assert_allclose(got, wanted, rtol=rtol, atol=atol)
