# Structural derivative compiler

`compile_operator` transforms a supported PyTorch tensor program and a derivative
request into an ordinary trainable tensor program. It accepts primal inputs and
returns **raw derivatives**, without a public coefficient axis. `operator_model`
provides the corresponding interface for built Keras Functional/Sequential models.
Both are experimental extensions; arbitrary Python execution is outside capture.

The compiler looks for mathematical structure before allocating a generic jet.
An order-32 mixed request can therefore succeed on a separable expression even
when its generic 2³²-coefficient layout would exceed the resource budget.
When a reduction cannot be proved, the supported program uses bounded Taylor
propagation. Unsupported operations or excessive resource requests raise errors;
there is no finite-difference or nested spatial-autodiff fallback.

## Ordinary inputs and trainable derivatives

```python
import torch
from omnibias.torch.experimental import OperatorSpec, compile_operator

class Separable(torch.nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(dimension, dtype=torch.float64))

    def forward(self, x):
        return torch.sin(torch.tanh(x * self.weight).sum(-1, keepdim=True))

model = Separable(24)
x = torch.full((2, 24), 0.1, dtype=torch.float64)
program = compile_operator(
    model, (x,), operator=OperatorSpec.mixed((1,) * 24), batch_ndim=1,
)
derivative = program(x)
assert derivative.shape == (2, 1)
derivative.square().mean().backward()
assert model.weight.grad is not None
report = program.explain()
```

The selected input has shape `(*batch, coordinates)`. `batch_ndim=0` means one
coordinate vector; `batch_ndim=1` means a batch of independent vectors. Batch
reductions, reordering, or other operations whose independence is not proved are
refused. Use `argnum` to select a different positional tensor. Nonselected inputs
and model parameters stay live and spatially constant for this derivative query.
Capture examples must use distinct tensor objects for distinct argument roles,
and cannot be the model's parameter or buffer objects themselves. Use separate
example tensors to prevent export from merging those roles. Runtime arguments
may share tensor identities; hybrid backward preserves that sharing.

Directional requests supply a runtime direction, held fixed for the requested
spatial derivative. Gradients into that direction remain available for training:

```python
directional = compile_operator(
    model, (x,), operator=OperatorSpec.directional(4), batch_ndim=1,
)
direction = torch.ones_like(x, requires_grad=True)
fourth = directional(x, direction=direction)
assert fourth.shape == model(x).shape
```

Other constructors are `OperatorSpec.laplacian(power=k)` and
`OperatorSpec.polynomial({multi_index: integer_or_fraction, ...})`. Polynomial
requests can combine different derivative orders, including a zeroth-order
term. `mixed(alpha)` requests **one** mixed derivative; it differs from
`JetSpec.mixed(targets)`, which constructs a coefficient layout for several
targets. Internal Taylor coefficients remain divided by their multi-index
factorials; operator results include the factorial conversion.

`program.compile(fullgraph=True)` compiles the emitted tensor program in place
using PyTorch's standard module API and preserves module hooks.
`torch.compile(program, fullgraph=True)` also works; the in-place form avoids
an additional module wrapper on the validated Torch version.
The existing `compile_jet` interface remains appropriate when callers need a
complete tower or supply nonlinear incoming Taylor paths. An operator request
does not change that existing contract.

## Compiling high orders and parameter higher gradients

Both Torch frontends accept `kernel_lowering="auto"` (the default), `"inline"`,
`"regional"`, or `"packed"`. Automatic lowering keeps small activation kernels
inline and selects reusable regional compilation for directional kernels whose
estimated inline graph exceeds 2,048 nodes. If the public regional API is absent,
it selects packed coefficient reductions. Resource estimates include region
bodies; wrapping a kernel does not evade a planning budget. `explain()` records
the selected lowering. Parameter values are runtime inputs, not cache keys.

Eager programs support higher parameter gradients. Ordinary `program.compile()`
supports training through its first backward, but the installed Torch AOT runtime
does not support differentiating that compiled backward a second time. Two
explicit interfaces cover higher gradients while preserving the original model
parameters:

```python
# Compile the forward; recompute a differentiable native backward.
hybrid = program.hybrid(backend="aot_eager")
hybrid_value = hybrid(x)

# Compile the Hessian-vector product of a scalar derivative loss.
hvp = program.compile_parameter_hvp(
    lambda output: output.square().mean(), backend="aot_eager",
)
tangents = {name: torch.ones_like(p) for name, p in program.named_parameters()}
parameter_hvp = hvp(tangents, x)
```

Omit `backend="aot_eager"` to use the default Inductor backend. The hybrid wrapper
itself stays outside outer fullgraph compilation; its forward kernel receives the
compile options. It supports ordinary `autograd.grad(..., create_graph=True)`
and double backward, with additional recomputation cost. Its parameter hooks and
outer wrapper hooks retain their normal behavior. Side-effecting hooks inside
the replayed canonical graph are rejected.

For HVPs, `loss_fn(outputs, *loss_args)` must return a scalar tensor. Pass targets
as `hvp(tangents, *args, loss_args=(target,), **kwargs)`. Directional operators
still receive `direction=...`. The Hessian acts on the program's unique named
parameters while inputs, directions and buffers are held fixed. The result uses
the same parameter-name mapping, including zeros for unused parameters; `.grad`
fields are untouched. Parameter gradients can include the spatial derivative's
full coefficient computation without nesting spatial autodiff.

Both interfaces retain an inline native graph independently of an earlier
`program.compile()` call. HVP transformation precedes backend compilation and
does not traverse regional annotations or an analytic custom backward wrapper.
`explain()` states these distinct execution capabilities.

## What the compiler proves

For a ridge field and a homogeneous degree-n operator P,

    f(x) = sum_h c_h sigma(w_h·x + b_h)
    P(∂)f = sum_h c_h P(w_h) sigma^(n)(w_h·x + b_h).

This follows from `D^alpha sigma(w·x+b) = w^alpha sigma^(|alpha|)(w·x+b)`.
The contraction avoids constructing the derivative tensor. Consecutive affine
maps retain trainable weights; suitable factorized ridge norms use a Gram
matrix rather than forming the large effective weight matrix.

For disjoint scalar branches, restrict the requested multi-index to each node's
selected coordinate support: beta = alpha|support(v). Propagate a differential
polynomial B_v satisfying

    D^beta H(v(x)) = sum_k B_v[k] H^(k)(v(x))

for every smooth scalar outer function H. Coordinate leaves use `B_xi=t^alpha_i`.
Disjoint sums convolve the two polynomials. Disjoint products use

    B_(uv)[k+l-j] += B_u[k] B_v[l] binom(k,j) binom(l,j) j!
                    × v^(k-j) u^(l-j).

Every exponent is nonnegative, so the rule remains valid at zero factors.
Unary regions apply a Bell-polynomial transform using their normalized derivative
towers. These rules follow by applying the two disjoint differential functionals
to `H(u+v)` or `H(uv)`, followed by the ordinary chain/product rules.

For a node of requested order n, its primal and n positive-degree operator
coefficients suffice. Bell-transform scratch can be quadratic and arithmetic
cubic in n; this is not a constant-work claim. The state reduction requires
proven disjoint dependencies or a scalar intermediate through which all relevant
paths pass. Overlapping dependencies are not treated as disjoint merely because
a weight or derivative is currently zero.

The common example is

    ∂_1 ... ∂_D F(sum_i u_i(x_i)) = F^(D)(sum_i u_i(x_i)) prod_i u_i'(x_i).

Here the generic mixed layout has 2ᴰ entries, but the recognized expression needs
branch first derivatives and an outer directional tower. Shared trainable
parameters across branches do not violate spatial independence.

For an affine bottleneck `z=A x+b`, the operator becomes `P(Aᵀ∂_z)`. In particular,

    Δ_x → sum_ab (AAᵀ)_ab ∂_za ∂_zb.

The metric `AAᵀ` remains in the parameter graph. It must not be replaced by an
ordinary latent Laplacian or cached as a detached constant. Torch uses packed
weighted Laplacian/biharmonic states for suitable chains and bounded latent jets
for other supported contractions. General small homogeneous operators may use
an exact apolar quotient when it reduces coefficient state. Neither minimum
algebra dimension nor a smaller state guarantees the shortest runtime.

## Keras models

Select the backend before importing Keras. The operator model reuses the original
variables and accepts ordinary model inputs; a directional model accepts
`[points, directions]`. The batch axis remains the ordinary Keras batch axis.

```python
import keras
from omnibias.keras.experimental import OperatorSpec, operator_model

inputs = keras.Input((3,), dtype="float32")
hidden = keras.layers.Dense(4, activation="sigmoid", dtype="float32")(inputs)
outputs = keras.layers.Dense(1, dtype="float32")(hidden)
source = keras.Model(inputs, outputs)
laplacian = operator_model(source, operator=OperatorSpec.laplacian(), precision="strict")
points = keras.ops.ones((2, 3), dtype="float32")
assert tuple(laplacian(points).shape) == (2, 1)
assert {id(v) for v in source.trainable_variables} == {
    id(v) for v in laplacian.trainable_variables
}
```

`precision="source"` remains the compatibility default. Use `precision="strict"`
for model-local FP32/FP64 arithmetic and `PrecisionMeanSquaredError` when the
training objective must retain that precision. The
[precision and training contract](jet-compiler.md#explicit-numerical-precision)
documents supported policies, custom rules, objective trees, and reporting
metrics.

Saving/loading the transformed model preserves its configuration and training
variables. Arbitrary subclass tracing is outside this frontend. Recognition is
conservative and can differ between frontends because their captured operation
representations differ. `explain()` reports the selected strategy and fallback.

## Numerical and resource contracts

`activation_method="recurrence"` is the default. It uses the shared differential
systems rather than evaluating high-degree tanh/sigmoid polynomials directly.
`activation_method="polynomial"` explicitly requests the original fast providers
where their activation semantics match. A general fallback can still require
recurrences; this is reported. Polynomial evaluation can lose substantial
accuracy at orders 16–32. Neither method eliminates floating-point conditioning.

The recurrence initializes tanh's slope as
`4*sigmoid(2*z)*sigmoid(-2*z)` and sigmoid's slope as
`sigmoid(z)*sigmoid(-z)`. These equal `1-tanh(z)**2` and
`sigmoid(z)*(1-sigmoid(z))` in exact arithmetic, but avoid subtracting nearly
equal floating-point numbers in the tails. Higher recurrence coefficients and
analytic pullbacks use the same stable initialization. This matters even for
high derivatives whose values greatly exceed the first derivative. Overflow,
underflow and conditioning still impose limits.

Preserve FP32/FP64, the source GELU approximation, Softplus parameters, dtype,
device, and framework activation semantics. Keras source-mode precision
limitations described in the [jet guide](jet-compiler.md) also apply here.
Strict mode uses a local numerical adapter for the supported operations;
float64 storage alone does not establish float64 arithmetic in source mode.

`OperatorSpec` accepts `JetBudget` and `OperatorPlanningBudget`. They bound
coefficient state, multiplication schedules, tensor size, symbolic work and
generated graph size. Compact requests are constructed before any mixed closure.
`explain()` gives strategy, structural rules, provider and resource information.
Forward-state estimates exclude allocator workspaces and reverse-mode tape unless
explicitly stated otherwise.
Torch caps emitted FX nodes incrementally, including a reference subgraph.
Keras bounds its source graph and aggregate owned coefficient schedules; the
report excludes backend compiler expansion and registered custom-rule internals
from that schedule estimate.

## Reproducible comparisons

The [v2 measured results](operator-compiler-v2-results.md) include successful
cells, baseline wins, resource refusals and the exact-source evidence archive.
The [historical v1 results](operator-compiler-results.md) remain available.

```bash
CUDA_VISIBLE_DEVICES='' uv run python benchmarks/operator_compiler.py \
  --families ridge --orders 2 4 6 8 --methods compiler manual folx jax-nested \
  --execution compiled --replicates 3 --repeats 20 --timeout 120

CUDA_VISIBLE_DEVICES='' uv run python benchmarks/operator_compiler.py \
  --families separable --orders 4 8 16 24 32 --methods compiler reference manual \
  --execution compiled --replicates 3 --repeats 20 --output operator_mixed.json

CUDA_VISIBLE_DEVICES='' uv run python benchmarks/operator_compiler.py \
  --families coupled --orders 2 4 8 16 32 \
  --methods compiler reference manual jax-jet torch-jet \
  --replicates 3 --repeats 20 --output operator_coupled.json
```

The bottleneck family measures weighted Laplacian/biharmonic transport. Choose
`--execution eager` for a separate eager comparison. Optional folx and torch-jet
dependencies are never installed by the harness. Every method receives runtime
coordinates and parameters. Reference coordinate seeding occurs inside its
compiled function, exposing the same structural zeros to backend optimization.

Cells first pass an independent 80-digit reference. Timings separate capture,
compilation/first execution, warm derivatives, backward, and derivative-loss SGD.
Weights reset outside each training sample. Each cell records exact source
snapshots and individual samples. Protocol v2 generates the independent oracle
once per exact request before starting engine workers, records its cost
separately, and reports total elapsed time. The engine's time and memory budgets
include imports, capture, compilation, and training checks, and exclude that
separately reported oracle. `--cache cold` starts with empty isolated compiler
caches and permits normal reuse within a worker; `--cache disabled` explicitly
disables backend caches for diagnostics. The default retains persistent disk
caches. Process RSS is not a tensor-allocation measurement. An additional untimed Torch
profiler pass records tracked CPU tensor-storage peaks; XLA reports its static
compiled-buffer estimate separately. Failed, refused and timed-out
cells receive no speedup. Completed stages of a timed-out cell remain labeled as
partial observations.

The historical ~5,000× ratios concern particular ridge polylaplacian comparisons
in the [performance record](performance.md). They are not guarantees for these
new compiler paths, general programs, training, or every Taylor-mode baseline.
The mathematical identities build on established differentiation theory; automatic
recognition and measured execution improvements are the contribution to assess.
