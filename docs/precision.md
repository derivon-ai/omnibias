# Precision: float32, TF32 and half precision

Three things decide how accurate a planned derivative is: the dtype of the inputs, the precision the hardware gives matrix
products of that dtype, and the derivative order. The planner fixes the second and checks the first against the third, and
`explain()` reports what it did. The numbers below are measured, not promised: they are the error of each program against a
50-digit oracle on small instances, and against the same program in float64 on realistic ones. They describe benign random
weights (see [Limits](#limits)); they do not bound the error of a trained network.

## The policy

```python
import jax
import jax.numpy as jnp
import numpy as np

from omnibias.core.planner import PlanCache
from omnibias.jax.operators import compile_operator

rng = np.random.default_rng(0)
params = [
    (jnp.asarray(rng.standard_normal((4, 8)) / 2, jnp.float32), jnp.zeros(8, jnp.float32)),
    (jnp.asarray(rng.standard_normal((8, 1)) / 3, jnp.float32), jnp.zeros(1, jnp.float32)),
]


def model(p, x):
    for W, b in p[:-1]:
        x = jnp.tanh(x @ W + b)
    W, b = p[-1]
    return (x @ W + b)[..., 0]


x = jnp.asarray(rng.standard_normal((6, 4)), jnp.float32)
op = compile_operator(model, (params, x), operator="laplacian", cache=PlanCache())
print(op(params, x).dtype, op.explain()["precision"])
assert op.explain()["precision"]["matmul"] == "highest"

# Half precision is refused for derivative order two and above, with the reason.
half_params = jax.tree_util.tree_map(lambda a: a.astype(jnp.bfloat16), params)
try:
    compile_operator(model, (half_params, x.astype(jnp.bfloat16)), operator="laplacian", cache=PlanCache())
except ValueError as err:
    assert "refused" in str(err)
else:
    raise AssertionError("bfloat16 at order 2 must be refused")
```

- `precision="highest"` (the default) computes float32 matrix products in float32 on every device. Matrix engines on recent GPUs
  otherwise give float32 products a 10-bit mantissa (TF32), an error of about 1e-3 per product before any derivative
  amplifies it. `"tf32"` opts in to that speed and `"default"` leaves the framework's own setting. In PyTorch the setting is
  a process-wide flag that the front end sets for the duration of a call and restores; threads that need different modes
  must be serialized.
- `float16` and `bfloat16` inputs are refused for derivative order two and above (the Laplacian and everything built on it);
  `allow_low_precision=True` runs them anyway. First-order quantities (divergence, `log|det(I + J)|`, the first-order Gram rows)
  are allowed and reported in `explain()["precision"]["notes"]`.
- `compute_dtype="float64"` (JAX with `jax_enable_x64`; PyTorch) evaluates a float32 problem in float64 and returns the input
  dtype.

## Float32 with the default policy

Relative error is the maximum absolute error over the outputs divided by the maximum absolute value of the oracle. Float32
epsilon is 1.19e-7. Weights and points are drawn in float64 and rounded to float32 before the oracle sees them, so the table
reports the error of the computation, not of rounding the inputs. The oracle is `mpmath` differentiating the network by finite
differences at 50 digits, so no derivative formula of this package is involved. Instances are small (`D` of 2 to 4, `H` of 3
to 8, depth 1 to 4) so that the oracle is affordable; the realistic sizes are in the last table.

| operator | hidden layers | JAX CPU | JAX A100 | PyTorch CPU | PyTorch A100 |
|---|---:|---:|---:|---:|---:|
| Δ | 1 | 3.5e-07 | 4.2e-07 | 3.7e-08 | 2.8e-07 |
| Δ | 2 | 4.5e-07 | 4.8e-07 | 3.6e-07 | 3.4e-07 |
| Δ | 3 | 3.5e-07 | 5.3e-07 | 1.6e-07 | 2.6e-07 |
| Δ | 4 | 2.2e-07 | 2.3e-07 | 6.8e-07 | 2.2e-07 |
| Δ² | 1 | 6.8e-08 | 2.0e-07 | 2.5e-07 | 2.5e-07 |
| Δ² | 2 | 2.9e-08 | 2.8e-07 | 3.5e-07 | 2.9e-08 |
| Δ² | 3 | 5.1e-08 | 3.2e-07 | 1.5e-07 | 1.5e-07 |
| Δ³ | 1 | 1.5e-07 | 1.0e-07 | 2.2e-06 | 2.2e-06 |
| Δ³ | 2 | 1.6e-07 | 1.7e-07 | 2.6e-06 | 2.6e-06 |
| div | 1 | 7.0e-08 | 2.8e-07 | 1.2e-07 | 1.8e-07 |
| div | 2 | 4.3e-07 | 7.7e-07 | 9.9e-08 | 2.2e-07 |
| div | 3 | 1.0e-07 | 1.5e-07 | 1.0e-07 | 1.5e-07 |
| log|det(I+J)| | 1 | 3.9e-07 | 4.0e-07 | 6.6e-07 | 4.0e-07 |
| log|det(I+J)| | 2 | 8.3e-07 | 4.4e-07 | 4.4e-07 | 4.4e-07 |
| log|det(I+J)| | 3 | 1.9e-07 | 2.0e-07 | 2.0e-07 | 1.1e-07 |
| Gram, Δ residual | 1 | 2.9e-07 | 2.9e-07 | 1.0e-07 | 1.2e-07 |
| Gram, Δ residual | 2 | 1.2e-07 | 3.2e-07 | 1.8e-07 | 1.8e-07 |
| Gram, Δ residual | 3 | 2.3e-07 | 2.0e-07 | 1.6e-07 | 1.1e-07 |
| Gram, Δ² residual | 1 | 3.0e-07 | 1.3e-07 | 4.0e-08 | 4.0e-08 |
| Gram, Δ² residual | 2 | 2.0e-07 | 2.9e-07 | 8.5e-08 | 1.8e-07 |

On these instances every default program is between 3e-8 and 8.3e-7 from the oracle, except the third-order Laplacian on PyTorch
(2.2e-6 and 2.6e-6, up to 22 float32 epsilons). The range over every feasible program, forced one at a time, is wider than the
range over the default:

| operator | hidden layers | JAX CPU | JAX A100 | PyTorch CPU | PyTorch A100 |
|---|---:|---:|---:|---:|---:|
| Δ | 1 | 3e-07–1e-06 | 3e-07–1e-06 | 4e-08–6e-08 | 2e-07–3e-07 |
| Δ | 2 | 3e-07–1e-06 | 3e-07–6e-07 | 3e-07–4e-07 | 3.4e-07 |
| Δ | 3 | 3e-07–5e-07 | 2e-07–5e-07 | 2e-08–2e-07 | 2e-07–3e-07 |
| Δ | 4 | 1e-07–3e-07 | 2e-07–6e-07 | 1e-07–7e-07 | 2e-07–2e-07 |
| Δ² | 1 | 6e-08–1e-07 | 2e-07–2e-07 | 7e-08–2e-07 | 2e-07–2e-07 |
| Δ² | 2 | 3e-08–1e-07 | 3e-07–5e-07 | 2e-07–3e-07 | 3e-08–2e-07 |
| Δ² | 3 | 5e-08–1e-07 | 1e-07–3e-07 | 5e-08–2e-07 | 1e-07–2e-07 |
| Δ³ | 1 | 1e-07–1e-06 | 1e-07–1e-06 | 7e-08–2e-06 | 7e-08–2e-06 |
| Δ³ | 2 | 2e-07–7e-07 | 6e-08–3e-07 | 2e-07–3e-06 | 5e-08–3e-06 |
| div | 1 | 7e-08–8e-08 | 2.8e-07 | 8e-08–1e-07 | 1.8e-07 |
| div | 2 | 2e-07–4e-07 | 4e-07–8e-07 | 5e-08–1e-07 | 2e-07–3e-07 |
| div | 3 | 1e-07–2e-07 | 2e-07–3e-07 | 1e-07–2e-07 | 1.5e-07 |
| log|det(I+J)| | 1 | 4e-07–9e-07 | 4e-07–9e-07 | 5e-07–7e-07 | 4e-07–8e-07 |
| log|det(I+J)| | 2 | 8.3e-07 | 1e-07–4e-07 | 4e-07–6e-07 | 4e-07–8e-07 |
| log|det(I+J)| | 3 | 9e-08–2e-07 | 2e-07–2e-07 | 2e-07–2e-07 | 1e-07–2e-07 |
| Gram, Δ residual | 1 | 1e-07–4e-07 | 2e-07–4e-07 | 1e-07–1e-07 | 1.2e-07 |
| Gram, Δ residual | 2 | 1e-07–1e-07 | 3e-07–3e-07 | 1e-07–2e-07 | 2e-07–2e-07 |
| Gram, Δ residual | 3 | 8e-08–2e-07 | 2e-07–2e-07 | 1e-07–2e-07 | 1.1e-07 |
| Gram, Δ² residual | 1 | 2e-07–3e-07 | 1e-07–2e-07 | 4.0e-08 | 4.0e-08 |
| Gram, Δ² residual | 2 | 1e-07–2e-07 | 2e-07–3e-07 | 8.5e-08 | 1.8e-07 |

The choice of program changes the float32 error by up to a factor of 60 (third-order Laplacian, PyTorch A100: 5e-8 to 3e-6), and no
program is uniformly the most accurate.

## TF32 matrix engines on an A100

| operator | hidden layers | JAX highest | JAX tf32 | PyTorch highest | PyTorch tf32 |
|---|---:|---:|---:|---:|---:|
| Δ | 1 | 4.2e-07 | 4.2e-07 | 2.8e-07 | 5.3e-04 |
| Δ | 2 | 4.8e-07 | 4.8e-07 | 3.4e-07 | 3.3e-03 |
| Δ | 3 | 5.3e-07 | 5.3e-07 | 2.6e-07 | 9.8e-04 |
| Δ | 4 | 2.3e-07 | 2.3e-07 | 2.2e-07 | 1.1e-03 |
| Δ² | 1 | 2.0e-07 | 2.0e-07 | 2.5e-07 | 7.4e-04 |
| Δ² | 2 | 2.8e-07 | 2.8e-07 | 2.9e-08 | 9.9e-04 |
| Δ² | 3 | 3.2e-07 | 3.2e-07 | 1.5e-07 | 2.1e-04 |
| Δ³ | 1 | 1.0e-07 | 1.0e-07 | 2.2e-06 | 5.5e-04 |
| Δ³ | 2 | 1.7e-07 | 1.7e-07 | 2.6e-06 | 8.7e-05 |
| div | 1 | 2.8e-07 | 2.8e-07 | 1.8e-07 | 2.9e-04 |
| div | 2 | 7.7e-07 | 7.7e-07 | 2.2e-07 | 7.1e-04 |
| div | 3 | 1.5e-07 | 1.5e-07 | 1.5e-07 | 7.2e-04 |
| log|det(I+J)| | 1 | 4.0e-07 | 4.0e-07 | 4.0e-07 | 3.1e-04 |
| log|det(I+J)| | 2 | 4.4e-07 | 4.4e-07 | 4.4e-07 | 3.9e-04 |
| log|det(I+J)| | 3 | 2.0e-07 | 2.0e-07 | 1.1e-07 | 1.0e-03 |
| Gram, Δ residual | 1 | 2.9e-07 | 2.9e-07 | 1.2e-07 | 2.3e-04 |
| Gram, Δ residual | 2 | 3.2e-07 | 3.2e-07 | 1.8e-07 | 3.8e-04 |
| Gram, Δ residual | 3 | 2.0e-07 | 2.0e-07 | 1.1e-07 | 3.6e-04 |
| Gram, Δ² residual | 1 | 1.3e-07 | 1.3e-07 | 4.0e-08 | 1.2e-04 |
| Gram, Δ² residual | 2 | 2.9e-07 | 2.9e-07 | 1.8e-07 | 4.0e-04 |

The PyTorch `tf32` column is the cost of the opt-in: 8.7e-5 to 3.3e-3 against 3e-8 to 2.6e-6 under `highest`, that is 33 to
34,000 times larger in every row (median about 3,000). **The JAX `tf32` column is identical to the `highest` column to every printed
digit.** At these tiny shapes XLA did not use the TF32 matrix engine for the products at all, so the table says nothing about
JAX's TF32 error; at realistic sizes (last table) the JAX `tf32` error is 1.9e-4 to 9.8e-4, the same order as PyTorch's. The
policy is therefore not a no-op in JAX, and a small-shape benchmark is not evidence of what it does at training sizes.

## Half precision and float32 computed in float64

| operator | hidden layers | bfloat16, default policy | bfloat16, `allow_low_precision` | JAX float32 inputs, `compute_dtype="float64"` | JAX float64 |
|---|---:|---:|---:|---:|---:|
| Δ | 1 | refused | 4.1e-03 | 3.7e-08 | 2.9e-16 |
| Δ | 2 | refused | 1.8e-02 | 2.4e-08 | 3.1e-16 |
| Δ | 3 | refused | 6.9e-03 | 2.2e-08 | 1.9e-16 |
| Δ | 4 | refused | 7.3e-03 | 8.7e-09 | 1.8e-16 |
| Δ² | 1 | refused | 2.3e-02 | 3.5e-08 | 4.4e-16 |
| Δ² | 2 | refused | 1.1e-02 | 2.9e-08 | 2.4e-16 |
| Δ² | 3 | refused | 8.7e-03 | 1.2e-08 | 3.8e-16 |
| Δ³ | 1 | refused | 9.2e-03 | 1.4e-08 | 6.2e-16 |
| Δ³ | 2 | refused | 7.7e-02 | 5.1e-08 | 2.1e-16 |
| div | 1 | 9.7e-03 | 9.7e-03 | 2.3e-08 | 3.5e-16 |
| div | 2 | 1.4e-02 | 1.4e-02 | 2.2e-08 | 3.2e-16 |
| div | 3 | 8.5e-03 | 8.5e-03 | 4.8e-08 | 2.4e-16 |
| log|det(I+J)| | 1 | error | error | 2.6e-08 | 1.0e-15 |
| log|det(I+J)| | 2 | error | error | 4.8e-08 | 1.5e-15 |
| log|det(I+J)| | 3 | error | error | 1.3e-08 | 1.8e-16 |
| Gram, Δ residual | 1 | refused | 1.4e-02 | 2.5e-08 | 2.3e-16 |
| Gram, Δ residual | 2 | refused | 3.0e-02 | 3.4e-08 | 4.0e-16 |
| Gram, Δ residual | 3 | refused | 6.4e-03 | 2.7e-08 | 2.5e-16 |
| Gram, Δ² residual | 1 | refused | 4.7e-03 | 2.9e-08 | 7.0e-16 |
| Gram, Δ² residual | 2 | refused | 2.2e-02 | 2.0e-08 | 6.4e-17 |

`bfloat16` under the default policy is refused for every operator of derivative order two and above (`refused`); first-order
divergence runs and is 8.5e-3 to 1.4e-2. With `allow_low_precision=True` the errors are 4e-3 to 7.7e-2, which is one or two correct
digits at best. `log|det(I + J)|` does not run in `bfloat16` (`error`): its determinant step
is not supported in `bfloat16` by the backends tested, so it is unsupported rather than inaccurate. Computing float32 inputs in float64
(`compute_dtype="float64"`) gives 8.7e-9 to 5.1e-8, at the rounding of the result to float32, 3 to 35 times below float32
arithmetic, at the price of float64 throughput.

## Realistic sizes

Oracle cost rules out `mpmath` here, so the reference is the same planned program in float64 on the same float32-rounded
weights and points (at the small sizes the float64 program agrees with the oracle to about 1e-15, far below every float32 error
below). Cases: Laplacian (`D = 256`, `H = 256`, 4 hidden layers), bi-Laplacian (`D = 24`, `H = 128`, 3), divergence and
log-determinant (`D = 64`, `H = 128`, 3), Gram of a Laplacian residual (`D = 16`, `H = 128`, 3; 512 + 128 rows).

| case | mode | JAX CPU | JAX A100 | PyTorch CPU | PyTorch A100 |
|---|---|---:|---:|---:|---:|
| lap1_D256 | float32 | 7.5e-07 | 9.1e-07 | 7.4e-07 | 3.3e-07 |
| lap1_D256 | float32_tf32 | 7.5e-07 | 9.8e-04 | 7.4e-07 | 1.0e-03 |
| lap1_D256 | bfloat16_allowed | 1.0e-02 | 8.6e-03 | 1.1e-02 | 1.1e-02 |
| lap2_D24 | float32 | 4.8e-07 | 5.5e-07 | 6.0e-07 | 5.3e-07 |
| lap2_D24 | float32_tf32 | 4.8e-07 | 7.6e-04 | 6.0e-07 | 1.1e-03 |
| lap2_D24 | bfloat16_allowed | 3.1e-02 | 1.5e-02 | 3.1e-02 | 3.1e-02 |
| div_D64 | float32 | 3.3e-06 | 4.5e-07 | 3.0e-06 | 4.6e-07 |
| div_D64 | float32_tf32 | 3.3e-06 | 7.6e-04 | 3.0e-06 | 4.9e-04 |
| div_D64 | bfloat16_allowed | 8.2e-03 | 8.7e-03 | 8.2e-03 | 8.2e-03 |
| logdet_D64 | float32 | 1.7e-06 | 1.7e-06 | 1.4e-06 | 1.9e-06 |
| logdet_D64 | float32_tf32 | 1.7e-06 | 5.8e-04 | 1.4e-06 | 5.5e-04 |
| logdet_D64 | bfloat16_allowed | error | error | error | error |
| gram2_D16 | float32 | 2.5e-07 | 2.4e-07 | 2.0e-07 | 1.4e-07 |
| gram2_D16 | float32_tf32 | 2.5e-07 | 1.9e-04 | 2.0e-07 | 3.3e-04 |
| gram2_D16 | bfloat16_allowed | 5.9e-03 | 4.5e-03 | 5.8e-03 | 5.8e-03 |

On a CPU `float32_tf32` is bit-identical to `float32` because there is no TF32 engine. On the A100 `highest` gives 1.4e-7 to
1.9e-6 and `tf32` 1.9e-4 to 1.1e-3 in both frameworks.

## Limits

- The weights are random and benign: entries of order `1/sqrt(fan_in)` and biases of order 0.2. A trained network with large
  weights, saturated activations or an ill-conditioned Jacobian amplifies rounding more, and a float32 table here is not a
  bound for it. Use `compute_dtype="float64"` or measure.
- The accuracy of the Gram matrix and of the damped Gauss-Newton step is a different question from the accuracy of the
  residual rows; the step has error about `kappa * eps`, and the conditioning report (see the
  [planner](operator-planner.md)) makes it visible.
- One run of each cell on one device model. The error is deterministic given the program; which program the planner picks
  depends on the device and the problem size, so the default column mixes programs.
- Only the A100 and CPU were measured. Other GPU generations have the same TF32 hardware setting and different default kernels;
  the policy applies there, the numbers have not been measured there.
- Raw results (one record per case, program and mode): small instances
  [JAX CPU float32](benchmarks/precision/jax_cpu_f32.json), [JAX CPU float64](benchmarks/precision/jax_cpu_x64.json),
  [JAX A100 float32](benchmarks/precision/jax_a100_f32.json), [JAX A100 float64](benchmarks/precision/jax_a100_x64.json),
  [PyTorch CPU](benchmarks/precision/torch_cpu.json), [PyTorch A100](benchmarks/precision/torch_a100.json); realistic sizes
  [JAX CPU](benchmarks/precision/large_jax_cpu.json), [JAX A100](benchmarks/precision/large_jax_a100.json),
  [PyTorch CPU](benchmarks/precision/large_torch_cpu.json), [PyTorch A100](benchmarks/precision/large_torch_a100.json).
