# Experimental bias-collapse differentiation compiler

The installed `omnibias-torch` and `omnibias-keras` packages now include
experimental graph transformations for directional and selected mixed Taylor
jets. They emit trainable tensor programs, with the original model's parameters
still participating in ordinary reverse-mode training. The initial optimization
range is orders 2–32. This is a supported-primitive compiler, not a universal
Python differentiator or a framework fork.

The implementation uses three shared modules: `omnibias.core.jet_algebra` for
coefficient layouts and budgets, `jet_systems` for exact rational differential
systems, and `jet_series` for backend-independent tensor recurrences. Existing
`ActivationSpec` constructors remain unchanged. Its Riccati metadata can be
adapted with `DifferentialSystemSpec.from_activation(spec)`.

## PyTorch: transform, seed, train

`compile_jet` captures a `torch.nn.Module` with `torch.export.export`, inspects
spatial dependence, and constructs an ordinary FX graph. Capture examples are
ordinary tensors. At execution, selected positional tensor arguments are
replaced with tensors whose **leading axis is the coefficient axis**.

```python
import torch
from omnibias.torch.experimental import (
    JetSpec, compile_jet, directional_seed, extract_derivative,
)

torch.manual_seed(7)
model = torch.nn.Sequential(
    torch.nn.Linear(2, 4), torch.nn.Tanh(), torch.nn.Linear(4, 1),
).double()
x = torch.tensor([[0.2, -0.1], [0.1, 0.3]], dtype=torch.float64)
direction = torch.ones_like(x, requires_grad=True)
spec = JetSpec.directional(4)
program = compile_jet(model, (x,), spec=spec)
packed = directional_seed(x, direction, spec)
output = program(packed)                  # (5, batch, output_features)
fourth = extract_derivative(output, spec, 4)
loss = fourth.square().mean()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
optimizer.zero_grad()
loss.backward()
assert direction.grad is not None
assert all(p.grad is not None for p in model.parameters())
optimizer.step()                          # updates the original model
report = program.explain()
assert report["spec"]["mode"] == "directional"
```

For accelerated execution, wrap the result with
`torch.compile(program, fullgraph=True)`. Capture and backend compilation are
separate costs; the first compiled call can be expensive. The generated
`program.graph_module` remains inspectable. The experimental frontend is tested
on Torch 2.14; this does not change the requirements of existing APIs.

Use `backward="analytic"` in `compile_jet` to opt into activation pullbacks.
Native reverse mode is the default and reference implementation. Both modes
preserve gradients into original parameters and trainable directions. Analytic
activation forward and slope graphs are emitted before runtime compilation;
the pullback applies the transpose multiplication schedule and supports eager
higher parameter gradients. For higher derivatives after compilation, use the
explicit [parameter HVP or hybrid interfaces](operator-compiler.md#compiling-high-orders-and-parameter-higher-gradients).
Ordinary backend-compiled backward remains subject to the backend's own support.

`argnums=(0,)` selects spatially varying positional inputs. Other tensor inputs
and model parameters remain runtime values with ordinary training gradients.
Parameters and buffers are referenced by the generated module, not detached or
cached by value. Shared parameters retain their identity. Static program
structure is required. Declared dynamic dimensions need finite upper bounds:

```python
dynamic_program = compile_jet(
    model, (x,), spec=spec,
    dynamic_shapes=({0: torch.export.Dim("batch", min=1, max=16)},),
)
assert dynamic_program(packed).shape == output.shape
```

## Selected mixed derivatives and nonlinear paths

`JetSpec.mixed(targets)` retains the union of the targets' divisor closures.
The basis is ordered by total degree, then lexicographically. Inspect
`spec.indices`, `spec.factorials`, and `spec.index(alpha)` rather than assuming
axis positions. Every coefficient is normalized by the multi-index factorial.
Formal directions need not be coordinate vectors; they may themselves be
trainable.

```python
from omnibias.torch.experimental import seed_jet

class Polynomial(torch.nn.Module):
    def forward(self, value):
        return value[..., :1] ** 2 * value[..., 1:2]

mixed = JetSpec.mixed([(2, 1)])
point = torch.tensor([[2.0, 3.0]], dtype=torch.float64)
jet = seed_jet(point, mixed, {
    (1, 0): torch.tensor([[1.0, 0.0]], dtype=point.dtype),
    (0, 1): torch.tensor([[0.0, 1.0]], dtype=point.dtype),
})
mixed_program = compile_jet(Polynomial(), (point,), spec=mixed)
answer = extract_derivative(mixed_program(jet), mixed, (2, 1))
torch.testing.assert_close(answer, torch.tensor([[2.0]], dtype=point.dtype))
assert mixed.size == 6
```

`seed_jet` also accepts higher normalized input coefficients for nonlinear
paths. Keras users can assemble such paths with `keras.ops.stack(..., axis=1)`.
For a nonlinear path, the output is the derivative of the composition along
that path; it is not simply a repeated directional derivative at the base point.

## Keras: preserve batching and original variables

Choose `KERAS_BACKEND` before importing Keras. `jet_model` supports built
Functional and Sequential models, including supported nested models and shared
layers. It uses public cloning APIs and, by default, portable `keras.ops`. Its coefficient
axis is **immediately after the batch axis**. `extract` returns a normalized
coefficient unless `derivative=True` is explicitly requested.

```python
import os
os.environ.setdefault("KERAS_BACKEND", "torch")
import keras
from omnibias.keras.experimental import jet_model, seed, extract

inputs = keras.Input(shape=(2,), dtype="float64")
hidden = keras.layers.Dense(3, activation="tanh", dtype="float64")(inputs)
base = keras.Model(inputs, keras.layers.Dense(1, dtype="float64")(hidden))
kspec = JetSpec.directional(2)
lifted = jet_model(base, spec=kspec)
kx = keras.ops.convert_to_tensor([[0.2, -0.1], [0.1, 0.3]], dtype="float64")
kpacked = seed(kx, keras.ops.ones_like(kx), spec=kspec)
koutput = lifted(kpacked)                  # (batch, 3, output_features)
second = extract(koutput, 2, spec=kspec, derivative=True)
assert tuple(second.shape) == (2, 1)
assert {id(v) for v in lifted.trainable_variables} == {
    id(v) for v in base.trainable_variables
}
lifted.compile(optimizer=keras.optimizers.SGD(1e-3), loss="mse")
lifted.train_on_batch(kpacked, keras.ops.zeros_like(koutput))
lifted.save("derivative_model.keras")
restored = keras.models.load_model("derivative_model.keras")
restored.train_on_batch(kpacked, keras.ops.zeros_like(koutput))
```

A residual-specific loss can select the desired coefficient and apply its
factorial within the loss. Native Keras training, saving, and loading remain
available. Loading creates a restored set of variables; it does not reconnect
to a separate original model in memory. `backward="analytic"` enables the
activation pullback, while other rules retain native AD. Custom layers and
operations need `register_jet_rule`; reinstall those registrations before
loading a saved custom model. Arbitrary subclassed model tracing is excluded.

The frontend is tested on Keras 3.15 with Torch 2.14, JAX 0.10, and TensorFlow
2.21 CPU in separate processes, including compiled training and optimizer
restoration. TensorFlow validation used an isolated optional environment.
Enable `JAX_ENABLE_X64=1` before constructing JAX arrays. Keras 3.15/JAX source
trigonometric activations downcast float64 internally, and Keras/Torch's type
promotion can compute float64 matmul operands in float32 before promoting the
result again. TensorFlow GELU uses some float32-rounded constants; the frontend
preserves that formula in the default `precision="source"` mode. The lifted model preserves the source primal and
reports precision limitations in `explain()`; float64 storage alone does not
establish float64 accuracy for such a source operation. Cross-backend tests use
explicit tolerances for these source differences, while core recurrences have
separate high-precision mathematical reference tests.

### Explicit numerical precision

Both `jet_model` and `operator_model` accept `precision="strict"`. This opt-in
mode evaluates the supported mathematical operations in the declared input and
layer compute dtype. Model-local numerical adapters use the selected backend's
public tensor operations for affected primitives, including matrix multiplication,
trigonometric functions, means, and stable softplus. They do not patch Keras or
change global dtype settings. Original variables remain shared with the source
model, and parameter gradients and higher gradients remain connected.

Strict mode supports float32 and float64. All floating inputs and layer compute
policies must agree; an authored cast to another dtype or a lower-precision layer
policy is rejected. JAX float64 requires `JAX_ENABLE_X64=1` before importing
Keras. Symbolic output specifications retain the declared dtype as well.

The selected exact or tanh-approximate GELU formula is preserved, with constants
constructed at the working precision and stable tail evaluation. Strict softplus
is the smooth mathematical function `log(1 + exp(x))`, evaluated stably; the
Torch source adapter's internal threshold approximation is retained only in
`precision="source"`. An explicitly authored cast or custom threshold does not
become a different formula: unsupported source nodes still require an explicit
rule. Strict mode improves the numerical primitive contract; it does not promise
accurate derivatives for arbitrarily ill-conditioned functions or inputs.

Keras' ordinary compiled loss aggregation can cast even a float64 loss to its
global float type. Use `PrecisionMeanSquaredError` with a strict transformed model
when the **training objective and gradients** must retain their compute precision:

```python
from omnibias.keras.experimental import (
    OperatorSpec, PrecisionMeanSquaredError, operator_model,
)

residual = operator_model(
    base, operator=OperatorSpec.mixed((2, 0)), precision="strict",
)
residual.compile(
    optimizer=keras.optimizers.SGD(1e-3),
    loss=PrecisionMeanSquaredError(dtype="float64"),
    jit_compile=True,
)
target = keras.ops.zeros((2, 1), dtype="float64")
objective = residual.compute_loss(y=target, y_pred=residual(kx))
assert keras.backend.standardize_dtype(objective.dtype) == "float64"
residual.train_on_batch(kx, target)
residual.save("strict_residual.keras")
continued = keras.models.load_model("strict_residual.keras")
continued.train_on_batch(kx, target)
```

The registered loss owns feature reduction, sample weighting, and batch
reduction. Its reduction options are `sum_over_batch_size` (default), `mean`,
`sum`, `mean_with_sample_weight`, and `none`/`None`. Weighted means divide by the
sum of broadcast effective weights; a zero denominator returns zero. Vector
targets with column predictions retain per-example weighting. Nonscalar
reductions are available when calling the loss directly; compiled strict
objectives require a scalar reduction.

The transformed models use Keras' public `compute_loss` hook only when strict
mode is paired with this loss. A single loss applies to every output; a nested
loss tree must exactly match the output tree and may contain `None` for skipped
outputs. Optional finite scalar loss weights and per-output sample weights must
match that structure; a single sample-weight tensor applies to every output.
The objective uses typed reductions and typed addition throughout. Implicit
masks, additional regularization or `add_loss` terms, and mixing this loss with
external losses in one objective are explicitly rejected. Supply explicit
sample weights for masks. External losses used on their own retain ordinary
Keras behavior. Metric trackers and the loss number reported by `train_on_batch`
may use Keras' reporting dtype; `compute_loss` and the training gradient use the
strict objective. The tested acceptance path uses native SGD on CPU.

Precision mode and loss dtype serialize with the model's public configuration,
including optimizer continuation. Loading creates new restored variables;
constructing a transformed model reuses the original variables without copies.

Custom rules opt in separately:

```python
from omnibias.core.jet_series import multiply
from omnibias.keras.experimental import register_jet_rule

@keras.saving.register_keras_serializable(package="example")
class Cubic(keras.layers.Layer):
    def call(self, value):
        return value**3

def source_rule(source, coefficient_args, coefficient_kwargs, spec):
    value = coefficient_args[0]
    return multiply(multiply(value, value, spec, keras.ops), value, spec, keras.ops)

def strict_rule(source, coefficient_args, coefficient_kwargs, spec, *, numerics):
    value = coefficient_args[0]
    return multiply(multiply(value, value, spec, numerics), value, spec, numerics)

register_jet_rule(Cubic, source_rule, strict_rule=strict_rule)
```

The strict callback receives the same four arguments as a source rule plus the
`numerics` keyword. It must support order zero, preserve batch independence,
return the declared dtype, and retain parameter gradients. Strict construction
refuses a source-only custom rule. Reinstall registrations before loading custom
models; registering a source rule again without `strict_rule` clears any earlier
strict callback. No tensor parameter values are stored in recurrence metadata.

## Supported semantics and resource limits

The rule sets cover addition, subtraction, multiplication, reciprocal/division,
constant scalar powers, linear/Dense, matmul, Conv1D/Conv2D, common static
reshapes and slicing, concatenation/stacking, sum/mean reductions, and smooth
activation families. Spatially varying multiplication or matrix operands use
coefficient convolution; spatially constant operands use affine rules.
Torch atom handling distinguishes exact GELU from its tanh approximation and
preserves softplus threshold semantics. Softmax uses a shifted exponential jet
and reciprocal normalizer, without exponentiating large unshifted logits.
Supported activation systems include tanh, sigmoid, exp, sin/cos, Gaussian,
erf/erfc, exact/approximate GELU, log, sqrt, softplus, and SiLU; Gaussian
expressions can also lower through their ordinary smooth primitives.

Keras operations must preserve the leading batch dimension. Reductions over
batch, moving batch into a feature position, and slicing away batch entries are
outside this contract. Conv, shape, indexing, and activation variants are
accepted only where an explicit rule exists. Unsupported nodes produce
construction-time diagnostics naming the operation (Torch also reports its
source stack). Neither frontend silently falls back to finite differences or
nested spatial autodiff. Mutation, randomness, data-dependent control flow and
indexing, discontinuities, complex/integer jets, and mixed-precision jets are
unsupported. Log/sqrt/reciprocal require their smooth mathematical domains.

```python
from omnibias.core.jet_algebra import JetBudget

budget = JetBudget(
    max_coefficients=128,
    max_multiplication_terms=4096,
    max_tensor_bytes=32 * 1024 * 1024,
)
bounded = JetSpec.directional(32, budget=budget)
assert bounded.size == 33
assert bounded.multiplication_terms == 561
```

Layout and product-work limits are checked before constructing schedules. The
default coefficient ceiling reuses `get_multi_index_budget()`; default product
work is one million ordered pairs, and default bytes are 256 MiB per packed
tensor. A byte bound is **not a bound on total process memory, temporaries,
compilation, or backward tape**. Keras checks known symbolic dimensions and
concrete eager shapes; backend tracing with an unknown batch cannot establish a
complete runtime allocation bound. Resource errors identify the violated
quantity. Narrowing target derivatives, order, batch, or layer width reduces
state. Both frontends expose an `explain()` report.

## Mathematical contracts

For n+1 equally spaced coalescing biases, the centered normalized difference is

    C[n,h](z) = h^(-n) sum(j=0..n) (-1)^(n-j) choose(n,j)
               sigma(z + (j-n/2)h).

Its limit is the n-th derivative, because the signed binomial moments vanish
below degree n and equal n! at degree n. Repeated integration gives an average
of sigma^(n) at a shift with mean zero and variance n h²/12; consequently the
centered truncation error is at most n h²/24 times the supremum of
|sigma^(n+2)| on the stencil. Sample perturbations of size eta can instead be
amplified by 2^n eta / |h|^n. Production kernels evaluate the symbolic collapsed
limit, not differences at vanishing floating-point separations.

For distinct c[j], R[t]/product(t-h*c[j]) has the fixed basis 1,t,...,t^n.
For nonzero h it represents independent evaluations at the roots. At h=0 it
becomes R[t]/(t^(n+1)), the Taylor algebra. Divided-difference coordinates retain
the derivative information at collision. The center b0 is an ordinary bias:
b0+epsilon is not nilpotent when b0 is nonzero. Repeated identical dual
directions can compress the 2^n square-free dual basis to its n+1-dimensional
permutation-invariant subalgebra. Distinct mixed directions generally need
additional coefficients. Shared ridge weights preserve the normal direction
of level sets, not their number or existence.

Stored coefficients satisfy u[alpha] = partial^alpha u(0) / alpha!.
Multiplication sums u[beta]v[gamma] for beta+gamma=alpha. An affine map with
spatially constant parameters applies its bias only to coefficient zero. Each
primitive preserves the input Taylor class modulo the truncation ideal, so
induction proves correctness for compositions of supported primitives.

For S'(z)=F(z,S), the degree operator E gives

    |alpha| S[alpha] = [F(u,S) E(u)][alpha].

Since E(u)[0]=0, positive-degree state coefficients depend only on already
computed lower degrees. Fixed polynomial auxiliary systems and cached
intermediate products give quadratic directional arithmetic in the order and
linear forward coefficient state per lane. The implementation never allocates
a full tensor of coefficient pairs. Exact GELU carries the normal CDF and
density; its negative-tail CDF initializer is erfc(-z/sqrt(2))/2.

The activation variation is delta y = sigma'(u) delta u. For a directional
algebra its reverse rule is

    bar_u[i] = sum(k=i..n) bar_y[k] * sigma_prime_series[k-i].

For selected mixed coefficients the rule transposes the corresponding product
schedule. Training an order-n residual normally needs activation smoothness
through n+1. Smoothness and algebraic exactness do not remove floating-point
roundoff, cancellation, underflow, or overflow. There is no cross-backend
bit-identity guarantee or global numerical error certificate for these jets.

## Advanced experiment: operator-directed state

For `L(p)=[P(partial)p](0)`, its multiplication annihilator
I={q: L(qr)=0 for every r} defines a sufficient quotient algebra. Any
compositional scalar-algebra representation rho with linear readout L must
satisfy ker(rho) contained in I: rho(q)=0 implies L(qr)=0. Thus it needs at least
dim(R[t]/I) states. This is a state lower bound within that representation
class, not a minimum-runtime theorem.

`omnibias.core.apolar.apolar_algebra` builds bounded small homogeneous examples
using exact `Fraction` arithmetic. Monomial, basis, multiplication-work, order,
and rational-bit limits constrain construction. General symbolic construction
is outside the tensor runtime and is not automatically selected by the two
graph frontends.

```python
from omnibias.core.apolar import apolar_algebra

laplace = apolar_algebra({(2, 0): 1, (0, 2): 1})
ex, ey = laplace.generators
assert laplace.size == 4
assert laplace.multiply(ex, ex) == laplace.multiply(ey, ey)
assert not any(laplace.multiply(ex, ey))
assert laplace.apply(laplace.multiply(ex, ex)) == 2
```

The exact Laplacian requires D+2 scalar-algebra states. The biharmonic state
contains (u, gradient, packed symmetric Hessian, gradient of Laplacian,
biharmonic), with 2+2D+D(D+1)/2 components. For powers Delta^k the corresponding
dimension is 2 choose(D+k-1,k-1)+choose(D+k-1,k). The experimental Torch
`biharmonic_*` operations implement product, affine, and activation rules;
`mlp_biharmonic` specializes an MLP's first affine step without allocating the
zero input Hessians. The returned Hessian axis is upper-triangle row-major.

```python
from omnibias.torch.experimental import mlp_biharmonic

weights = torch.tensor([[0.5, -0.3]], dtype=torch.float64, requires_grad=True)
state = mlp_biharmonic(point, [(weights, None, "tanh")])
assert state.hessian.shape == (1, 1, 3)
state.biharmonic.square().sum().backward()
assert weights.grad is not None
```

With g=gradient(u), H=Hessian(u), q=gradient(Delta u), r=Delta²u,
ell=trace(H), and s=g·g, composition by sigma uses

    g_new = sigma' g
    H_new = sigma' H + sigma'' ggᵀ
    q_new = sigma' q + sigma''(ell g + 2Hg) + sigma''' s g
    r_new = sigma' r + sigma''(ell² + 2||H||² + 4g·q)
            + sigma'''(2ell s + 4g·Hg) + sigma'''' s².

This avoids third- and fourth-order derivative tensors and costs O(D²) per
scalar activation. It does not establish a runtime speedup over existing
contraction methods; those comparisons remain workload-dependent.

## Validation and reproducible measurements

Tests cover coefficient normalization and associativity, nonlinear incoming
paths, mixed targets, independent high-precision activation derivatives
including GELU tails through order 32, parameter/direction gradients,
original-variable optimizer updates, shared layers, supported primitives,
dynamic Torch shapes, native/analytic pullbacks, serialization, refusal cases,
and genuinely multivariate biharmonic identities.

Run the isolated CPU benchmark with:

```bash
CUDA_VISIBLE_DEVICES='' uv run python benchmarks/jet_compiler.py \
  --orders 2 4 8 16 32 --repeats 5 --timeout 90
```

It compares generated eager/Inductor/native/analytic paths, handwritten
omnibias jets, nested AD, JAX Taylor mode, and optional `jet-for-pytorch`.
Missing methods and timeouts are reported explicitly; nested AD defaults to a
bounded maximum order. Timings separate capture, first call including backend
compilation, warm evaluation, backward where measurable, and fixed-state
optimizer steps. The harness records versions, source hashes, peak worker RSS,
and an independent 80-digit reference. RSS includes the worker runtime and is
not live tensor memory. The companion operator benchmark is:

```bash
CUDA_VISIBLE_DEVICES='' uv run python benchmarks/biharmonic_compiler.py \
  --dims 2 4 --methods packed packed-inductor support nested folx torch-jet \
  --repeats 2 --timeout 20
```

It compares the packed state against existing omnibias support contractions,
nested Torch Hessians, nested forward Laplacians using optional `folx`, and
the exact collapsed `jet.bilaplacian(..., collapsed=True, randomization=None)`
construction from optional `jet-for-pytorch`. The collapsed baseline also has
an independent parameter-gradient comparison against the packed state.
The workload is a fixed one-hidden-layer tanh field with runtime parameters;
an independent 80-digit ridge formula checks the result. Eager and compiled
execution regimes are labeled separately. These are bounded CPU experiments,
not a GPU or deep-network performance survey. Optional comparison packages are
never installed by either script. No universal speedup, exponential AD lower
bound, or novelty claim follows from these initial tests.

The algebra and recurrence principles build on established
[divided differences](https://arxiv.org/abs/math/0502036),
[Weil-algebra AD](https://arxiv.org/abs/2106.14153), and
[Taylor-mode collapse](https://arxiv.org/abs/2505.13644).
[torch-jet](https://github.com/f-dangel/torch-jet) supplies a related comparison
implementation; [apolarity of quadratic powers](https://arxiv.org/abs/2208.07921)
provides the relevant quadratic-power dimension theory. The systems research
question is whether choosing coefficient states, recurrences, and pullbacks
together improves measured end-to-end workloads.
