# Operator planner: choose the program from the descriptor

`compile_operator` and `compile_residual` take a model and a request (an operator, or a
residual with the quantities it demands) and return a callable. Between the two sits a
**planner**, in the way a convolution library selects an algorithm for a layer
descriptor: it lists the programs that can serve the request, refuses the ones that
cannot and says why, ranks the rest by a per-device cost model (or by timing them) and
keeps the plan in a cache. `plan.explain()` shows every step. The planner changes which
program runs, not what is computed: every program it can choose is exact
for the request, and a refusal names the reason.

Both backends use one planner, `omnibias.core.planner`, and one registry of programs. The
front ends are `omnibias.jax.operators` and `omnibias.torch.operators` (`compile_operator`,
`compile_residual`). They are stable in the sense of [stability](stability.md): the plan
format is versioned, the old `experimental` names are aliases on a published schedule, and a
conformance suite checks any engine. The programs the planner chooses among stay experimental,
and the registry may gain or retire programs; the result of a request does not change.

## A request and its plan

```python
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from omnibias.jax.operators import compile_operator


def init(widths, seed=0):
    rng = np.random.default_rng(seed)
    return [
        (jnp.asarray(rng.standard_normal((a, b)) / np.sqrt(a)), jnp.asarray(0.1 * rng.standard_normal(b)))
        for a, b in zip(widths[:-1], widths[1:], strict=True)
    ]


def mlp(params, x):
    a = x
    for W, b in params[:-1]:
        a = jnp.tanh(a @ W + b)
    W, b = params[-1]
    return (a @ W + b)[..., 0]


D = 24
params = init([D, 6, 6, 1])
x = jnp.asarray(np.random.default_rng(1).standard_normal((8, D)))

op = compile_operator(mlp, (params, x), operator="laplacian")
laplacian = op(params, x)

plan = op.explain()
print(plan["selected_strategy"], [c["name"] for c in plan["candidates"]])


def dense_laplacian(z):
    return jnp.trace(jax.hessian(lambda q: mlp(params, q[None])[0])(z))


np.testing.assert_allclose(np.asarray(laplacian), np.asarray(jax.vmap(dense_laplacian)(x)), rtol=1e-9)
```

`explain()` returns the descriptor, the ranked `candidates` with their predicted seconds
and compile costs, the `refused` programs with reasons, the memory budget and the library
versions, and after the first call the measured time. `mode="find"` times the leading
candidates on the example and keeps the fastest, as `cudnnFind` does; the default
`mode="heuristic"` trusts the cost model. `force="name"` selects a program that is feasible.
`memory_budget` (bytes) refuses programs predicted to exceed it. `cache` is a `PlanCache`
or a path, so a second process starts from the saved plan. `bucket="pow2"` pads the batch
so that new batch sizes reuse a compiled executable.

## Models that are not dense chains

A *dense chain* is `dot -> bias -> activation` repeated and an affine readout. The chain
programs (projected Laplacian, forward Laplacian, polylaplacian, Woodbury solves, stream
Gram) need that structure. A literal scale such as the `w0` of a SIREN layer is folded
into the weights; skip connections, gates and attention are not chains.

When the input enters the model **only through products with parameter matrices**
(`x @ W`), every input derivative lies in the row space of the stacked matrices. The
planner then runs a generic exact engine, Taylor-mode jets or nested differentiation,
along `min(rows, q)` directions instead of the `q` columns of the operator: the
`subspace_jet` and `subspace_nested` programs.

```python
def residual_mlp(params, x):
    (W1, b1), *blocks, (Wo, bo) = params
    h = jnp.tanh(x @ W1 + b1)
    for W, b in blocks:
        h = h + jnp.tanh(h @ W + b)
    return (h @ Wo + bo)[..., 0]


skip_params = init([D, 5, 5, 5, 1], seed=2)
skip = compile_operator(residual_mlp, (skip_params, x), operator="laplacian")
assert skip.algorithm == "subspace_jet" and not skip.matched
print(skip.explain()["input_products"])


def dense_skip(z):
    return jnp.trace(jax.hessian(lambda q: residual_mlp(skip_params, q[None])[0])(z))


np.testing.assert_allclose(
    np.asarray(skip(skip_params, x)), np.asarray(jax.vmap(dense_skip)(x)), rtol=1e-9
)

# A model that also reads x elementwise is refused the reduction, with the reason.
def reads_x(params, x):
    return residual_mlp(params, x) + 0.3 * x[..., 0] ** 2


other = compile_operator(reads_x, (skip_params, x), operator="laplacian")
assert other.algorithm == "taylor_jet"
print({r["algorithm"]: r["reason"] for r in other.explain()["refused"]})
```

The basis of the retained directions is held constant under differentiation (it is
computed from `stop_gradient` weights). First-order parameter gradients of the operator are
exact, because the dropped directions are annihilated by the first-layer matrices; second-order
parameter derivatives of the operator are not. The programs are refused, with the reading
operation named, when the input is read anywhere else, and when `rows >= q` (no reduction).

## PyTorch

```python
import torch
from torch import nn

from omnibias.torch.operators import compile_operator


class Residual(nn.Module):
    def __init__(self, d=24, h=5, blocks=2):
        super().__init__()
        self.inp = nn.Linear(d, h)
        self.blocks = nn.ModuleList(nn.Linear(h, h) for _ in range(blocks))
        self.out = nn.Linear(h, 1)

    def forward(self, x):
        h = torch.tanh(self.inp(x))
        for blk in self.blocks:
            h = h + torch.tanh(blk(h))
        return self.out(h)[..., 0]


torch.manual_seed(0)
module = Residual().double()
xt = torch.randn(8, 24, dtype=torch.float64)
torch_op = compile_operator(module, xt, operator="laplacian")
assert torch_op.algorithm == "subspace_jet"

reference = torch.func.vmap(
    lambda z: torch.trace(torch.func.hessian(lambda q: module(q[None])[0])(z))
)(xt)
torch.testing.assert_close(torch_op(xt), reference, rtol=1e-9, atol=1e-9)

(torch_op(xt) ** 2).sum().backward()
assert module.inp.weight.grad is not None
```

The PyTorch planner reads a layer list, an `nn.Sequential` of `nn.Linear` and activation
layers, or any `torch.fx`-traceable module. A module that fx cannot trace runs on the
generic engines, and `explain()` says why. The projected `Delta^2` and `Delta^3` are ported
(`projected_deep_bilaplacian` and `projected_deep_trilaplacian` in `omnibias.torch.experimental`;
they store the Hessians densely, where the JAX programs use a packed layout). The projected
residual streams are not ported to PyTorch, and `compile_residual` refuses `projected_streams`
there with that reason.

## Residuals and Gauss-Newton

`compile_residual(model, blocks, quantity=...)` plans the residual Gram `K = J J^T`, the
damped Gauss-Newton step or the NTK diagonal for blocks of `(points, Residual(fn, demand))`.
A residual states which derivative quantities it needs (`value`, `grad`, `grad_sq`,
`lap`, `bilap`, `directional`); the planner routes it to an engine that provides them and
refuses the others with the missing quantity. The step reports its conditioning:
`conditioning="report"` (the default) never changes a result; `"floor"` raises the damping
to the smallest value that meets a requested step tolerance and says by how much.

```python
from omnibias.jax.operators import Residual, compile_residual

pts = jnp.asarray(np.random.default_rng(3).standard_normal((9, D)))
poisson = Residual(lambda q: q.lap[:, 0] + q.u[:, 0], ["value", "lap"])
gn = compile_residual(mlp, [(pts, poisson)], example_params=params, quantity="gauss_newton")
r, K, step = gn(params, damping=1e-3)
assert K.shape == (9, 9)
print(gn.explain()["selected_strategy"], gn.explain()["conditioning"]["status"])
```

## Planning for training

A forward evaluation and a training step are different objectives: the backward pass adds
work, launches and memory that the forward ranking does not see. `differentiate` states the
objective, and the planner ranks, and `mode="find"` times, that one.

| `differentiate` | the callable is used for | ranked and timed as |
|---|---|---|
| `"none"` (default) | a forward evaluation | forward work and launches |
| `"params"` | value-and-gradient into the parameters, as in a training step | forward plus backward work, launches and peak memory |
| `"params2"` | a second derivative into the parameters (Newton and curvature methods) | three times the forward work; **not validated against measurements** |

`expected_uses` is the number of calls the compile cost is amortised over: an integer, or
`"auto"`, which is `uses_hint` when a training API passes its step count, else 1,000 for
`differentiate="params"` and 1 for a forward evaluation. `mode="auto"` times the leading
candidates (as `mode="find"` does) once `expected_uses` reaches 50 and the arguments are
concrete, and trusts the cost model otherwise; a measured choice is kept in a persistent plan
cache (`$OMNIBIAS_PLAN_CACHE`, with `off` keeping it in process, else `plans.json` under
`$OMNIBIAS_CACHE_DIR` or `~/.cache/omnibias`). `explain()` reports the objective, the
resolved `expected_uses` and where it came from (`uses_source`: `explicit`, `auto` or
`hint`).

The training preset is the exhaustive find. `planned_evaluator` (JAX and PyTorch `training_step`) asks for
`mode="find"` and `top_k=None` unless you override them: every feasible program is timed once, on the value-and-gradient the
step runs, and the measured best is kept in the persistent plan cache. A run of days pays that once and does not depend on
where the cost model ranked the program that turns out fastest. The default of `compile_operator` stays `top_k=2` and
`mode="heuristic"`; `top_k=None` is the same switch for any call, and `mode="auto"` keeps its cheaper rule.

How `find` measures, and what it never changes:

- `top_k` is how many candidates it times, in the cost model's order: `2` by default, `None` for every feasible candidate
  (the refused ones stay refused). Exhaustive find costs one compile and a few runs per candidate, once per cache key.
- It times the predicted leader first, then each other candidate under a deadline of `FIND_CUT_FACTOR` (4) times the best
  time so far, so a candidate that cannot win stops after one run. A cut candidate keeps its measured time and a `find`
  note in `explain()`; the chosen program is the minimum of the measured times, as before.
- A follower whose predicted peak memory exceeds `host_memory_limit()` (half of the smaller of the physical and the container
  memory; a scheduler's per-job limit is not visible) is not timed and carries a `find` note saying so. The leader is always
  timed. A candidate that cannot run is a failed measurement ranked last, never an error of the plan. A failure that repeats
  (a JAX program whose compiled size exceeds the memory budget; with `safe_find=True`, a candidate whose process crashed) is
  recorded in the plan's `refused` list with a `find:` reason and the plan is cached, so the next call does not pay for it
  again. A failure that may not repeat (a timeout, an error at run time) is an infinite time and keeps the plan out of the
  cache. The residual compilers treat a program that fails when timed the same way; before, that exception reached the caller
  of the JAX `compile_residual`.
- The time is that of the callable the operator serves: the JAX program is lowered and compiled ahead of time and refused on
  its memory analysis before it runs, under the same precision policy; the PyTorch program is timed under the operator's
  matmul precision and `compile=` mode, with the CUDA cache emptied between candidates.
- Successful timings are appended to a journal beside the persistent plan cache (`plans.json.journal`), so a `find` that was
  killed resumes with the candidates it had measured; a finished `find` closes its entries.
- The cache key of a `find` carries the library versions (JAX or PyTorch, `omnibias-core`, the backend package), a
  non-default `top_k`, the matmul precision, `compute_dtype` and compile mode of the measurement (empty under the defaults),
  and `expected_uses` rounded to one significant digit, not the exact count. A plan measured under other library versions is
  measured again instead of reused; plans cached before this key existed are not found. `PLAN_FORMAT` is unchanged.

```python
from omnibias.jax.operators import PlanCache, compile_operator

train_op = compile_operator(
    mlp, (params, x), operator="laplacian", differentiate="params", expected_uses=1000
)
request = train_op.explain()["request"]
print(train_op.algorithm, request["differentiate"], train_op.explain()["uses_source"])

grads = jax.grad(lambda p: jnp.sum(train_op(p, x) ** 2))(params)


def dense_loss(p):
    def trace(z):
        return jnp.trace(jax.hessian(lambda q: mlp(p, q[None])[0])(z))

    return jnp.sum(jax.vmap(trace)(x) ** 2)


for ours, theirs in zip(
    jax.tree_util.tree_leaves(grads),
    jax.tree_util.tree_leaves(jax.grad(dense_loss)(params)),
    strict=True,
):
    np.testing.assert_allclose(np.asarray(ours), np.asarray(theirs), rtol=1e-8, atol=1e-10)

# `auto` resolves the number of uses and whether to time the candidates
auto_op = compile_operator(
    mlp,
    (params, x),
    operator="laplacian",
    differentiate="params",
    expected_uses="auto",
    mode="auto",
    cache=PlanCache(),
)
auto_plan = auto_op.explain()
assert (auto_plan["mode"], auto_plan["expected_uses"], auto_plan["uses_source"]) == (
    "find",
    1000,
    "auto",
)

# `top_k=None` times every feasible program, not the two the cost model ranks first
exhaustive = compile_operator(
    mlp,
    (params, x),
    operator="laplacian",
    differentiate="params",
    mode="find",
    top_k=None,
    cache=PlanCache(),
)
feasible = [c["name"] for c in exhaustive.explain()["candidates"]]
measured = exhaustive.explain()["measured_seconds"]
assert exhaustive.algorithm in measured
assert set(measured) <= set(feasible)
```

With `exact=False`, `accuracy` is the standard error a stochastic estimator may have as a fraction
of the Hessian's Frobenius norm, and the planner sizes the probe count from it
(`ceil(2 / accuracy^2)` Gaussian or Rademacher probes, `ceil(m / (1 + accuracy^2))` coordinates
for SDGD); without `exact=False` an `accuracy` is refused. These are worst-case bounds on the
standard error, not a guarantee on a single draw.

The training loop itself is a separate cost. `omnibias.jax.experimental.training_step`
compiles the sampler, the planned operators, the loss, the gradient and the optimizer update
into one executable that runs `steps_per_call` steps in a `lax.scan`, draws the next block's
points inside the current one and donates the state on accelerators. Blocks that share an
`evaluate` callable are one launch. `omnibias.torch.experimental.training_step` does the same
for PyTorch with a CUDA graph. `omnibias.jax.optim_pinn` and `omnibias.torch.optim_pinn` hold
the second-order steps (row-budgeted Gauss-Newton with a dense or conjugate-gradient solve,
Levenberg-Marquardt with a held-out trust region, ENGD's grid line search, SPRING momentum,
KFAC-expand and L-BFGS with line-search counts). They state their stopping rules and claim no
convergence for any PDE.

```python
from omnibias.jax.experimental.training_step import (
    StepBlock,
    adam,
    compile_training_step,
    planned_evaluator,
)

dim = 3
net = init([dim, 16, 16, 1])
laplace = planned_evaluator(mlp, net, jnp.zeros((64, dim)), steps=400)


def source(pts):
    return -jnp.sum(jnp.pi**2 * jnp.sin(jnp.pi * pts), axis=-1)


interior = StepBlock(
    "interior",
    sample=lambda key: jax.random.uniform(key, (64, dim), minval=-1.0, maxval=1.0, dtype=jnp.float64),
    evaluate=laplace,
    residual=lambda p, pts, lap, aux: lap - source(pts),
    order=2,
)
boundary = StepBlock(
    "boundary",
    sample=lambda key: jnp.sign(jax.random.normal(key, (32, dim), dtype=jnp.float64)),
    residual=lambda p, pts, ev, aux: mlp(p, pts),
    weight=10.0,
)
step = compile_training_step([interior, boundary], adam(1e-2), steps_per_call=8)
state = step.init(net, key=0)
state, first, _ = step.run(state, 1)
state, last, _ = step.run(state, 20)
assert float(last["loss"][-1]) < float(first["loss"][-1])
```

The assertion is a smoke check that the loss falls on this toy problem. It is not a convergence
statement, and the time a step takes is a measured property of a device, reported in
[performance](performance.md), not a property of this recipe.

## The low-order programs

For the Laplacian of a dense chain the registry holds several exact programs, and which one is
fastest depends on the regime. At PINN sizes (hidden width up to about 128, a few layers) a
program is bound by launch latency, not arithmetic, so the number of kernels per layer matters
more than the flop count.

| program | what it is | served | chosen by default |
|---|---|---|---|
| `projected` | first-layer projection of the operator. For `Delta` it is the stacked stream engine: value, tangent and second-order streams in one array, so each layer is one `dot_general` and one elementwise update, with no factorization and no conditional; `Delta^2` and `Delta^3` use the projected bi- and tri-Laplacian programs | Laplace powers 1 to 3, JAX and PyTorch | yes, where feasible |
| `taylor_closed` | closed-form directional Taylor streams (orders 1 to 4) | `Delta` and `Delta^2` only; `Delta^3` needs derivative order six and is refused with the pointer to `projected` and `forward_polylaplacian` | ranked with the others |
| `fused_streams` | one forward and one backward kernel (Pallas on JAX, Triton on PyTorch): a tile of points stays in registers and shared memory through every layer, and the backward kernel recomputes the forward tile | `Delta` with the activations `tanh`, `sigmoid`, `sin`, `cos`, `softplus`; width at most 128, at most 24 stream channels; one derivative into the parameters, so not `params2` | ranked on PyTorch in float32 inside the region it was measured in (width at most 64, `D` at most 16); forced only on JAX, in float64 and outside that region (below) |
| `forward_laplacian`, `forward_polylaplacian` | forward Laplacian with a collapsed second-order stream; powers 2 and 3 | | ranked with the others |
| `taylor_jet`, `nested_autodiff` | the generic engines | any differentiable model | the fallback |

Three rules in the registry come from measurements, and each is visible in `explain()`.

- **`fused_streams` is ranked on PyTorch in float32 where it was measured, and runs only when named with
  `force` elsewhere**: in float64 and on JAX in any dtype, and on PyTorch outside width at most 64 and `D`
  at most 16. Those cases are listed among the refusals with the reason. In float64 the kernel measured 3 to
  8 times slower than the stream engine on an A100 (the Triton dot has no tensor-core path for it). At width
  128 it ran 3 to 26 times slower in the forward pass. On JAX the Pallas kernel was slower than the XLA stream
  engine in every cell of the exploration that decided this: 25 to 42 us against 12 to 14 us forward, and 100
  to 250 us against 66 to 160 us for value-and-gradient. The [per-call grid](performance.md#per-call-grid)
  repeated it on JAX against `folx`: in the latency-bound cells the Pallas kernel's median device-time speed
  is 0.023 (float64) and 0.34 (float32, highest) of the best `folx` forward, and it meets neither registered
  threshold in any cell. On PyTorch in float32 the candidate-pool run of the grid found the one place where a
  kernel beat the alternatives by the registered factor: at `D = 2`, width 64 and two hidden layers its
  device time (CUDA-graph replay) is 0.033 to 0.043 ms forward against 0.10 to 0.13 ms for the planned
  default (2.6 to 3.1 times), and 0.12 to 0.15 ms against 0.29 to 0.45 ms for value-and-gradient (2.2 to
  3.1 times); at `D = 16` the forward kernel is as fast as the default (0.9 to 1.2 times) and the backward
  kernel is refused by the shared-memory bound. The cost model now prices that kernel from counted kernel
  launches (one forward and one backward kernel against dozens of launches for the stream program) and
  charges its cold compilation, 4.3 to 7.0 s forward and 3.2 to 3.9 s for the gradient (cached on disk
  afterwards), over `expected_uses`. So the planned default takes the kernel on PyTorch in float32 within the
  region when the program is used about ten thousand times or more (long training runs, repeated
  evaluation), and keeps the stream program for a single call and for the default `expected_uses` of 1,000,
  where the compilation does not amortize; `mode="find"` times the two leading candidates (`top_k=None` all of them), which then include
  the kernel. `force="fused_streams"` selects it anywhere it applies.
- **`hardware_requires`**: the fused kernel is refused where its smallest tile would need more
  shared memory than the lowering allows (166,912 bytes, the per-block figure of an sm80-class GPU
  as the Triton lowering reports it). The bound is calibrated against compile outcomes on an A100
  and carries a safety factor (5 % for JAX, 15 % for PyTorch, which has only two failure points).
  A larger GPU gains headroom the check does not use; an untested GPU is not covered by the
  calibration.
- **`taylor_closed` serves powers up to 2.** The streams carry derivative order four at most, and
  `Delta^3` needs order six, so the request is refused with that reason and the pointer to
  `projected` and `forward_polylaplacian`. For `Delta^2` the streams stack `4 m^2 + 1` channels for
  `m` selected inputs; 16,385 channels (`m = 64`) compile and run, while 262,145 (`m = 256`) held
  more than 98 GiB of host memory for 40 minutes without finishing a compile on either back end.
  Above 16,385 channels, the largest size measured to compile, the program is refused with the
  count and the pointer to `projected`. Nothing between 64 and 256 inputs was measured, so the
  cap is a measured lower bound and not a fitted limit.

`explain()["refused"]` lists every program that was not eligible with its reason, including the
programs that are forced-only, so a choice that looks surprising can be read off the plan. On
accelerators `compile_policy="auto"` retries a compile that fails with an XLA autotune or
Triton-GEMM error (`Failed to get configs`, `misaligned address`, `illegal memory access`) with
autotuning disabled and then with Triton GEMM disabled; `explain()` records the rung used.
`safe_find=True` times each `mode="find"` candidate in a fresh process, so a compiler crash or a
hang ends that candidate instead of the caller; it takes layer lists with named activations, and
each candidate pays its own interpreter start-up and compile.

The measured per-call behaviour of these programs against the Taylor-mode and `folx` baselines
is in [performance](performance.md#per-call-grid), with the cells where they lose.

## Coverage

[Operator coverage](operator-coverage.md) is generated from a dense-autodiff oracle: for
each backend it lists which operator, architecture, activation, vector-field and residual
combinations the planner serves with a chain program (`structured`), with a generic
exact engine along reduced directions (`reduced`), with a generic engine without
reduction (`fallback`), by composition, or not at all (`refused` with the reason,
`not_provided` with the route that exists). A cell that disagrees with the oracle fails
the generator, and `tests/test_coverage_matrix.py` fails if a committed cell does.

## Measured cost of the subspace programs

Float64, `H = 64` hidden width, two residual blocks, batch 64, medians of 9 repeats of
blocks of at least 30 ms, compile excluded; each program was forced by name and `planned`
is the unforced default. The model is a residual MLP, so no chain program applies.
Times are milliseconds per call; the last column is the planned default over the faster of
the two generic engines (below 1 means the default was faster). Raw results:
[`benchmarks/planner/`](benchmarks/planner/subspace_jax_cpu.json).

**JAX, CPU** (`subspace_jax_cpu.json`)

| `D` | default program | default | `subspace_jet` | `subspace_nested` | `taylor_jet` | `nested_autodiff` | default / best generic |
|---:|---|---:|---:|---:|---:|---:|---:|
| 64 | `taylor_jet` | 3.35 | – | – | 3.36 | 2.87 | 1.17 |
| 256 | `subspace_jet` | 3.94 | 3.92 | 2.44 | 23.91 | 22.90 | 0.17 |
| 1,024 | `subspace_jet` | 7.96 | 5.25 | 3.72 | 74.52 | 172.85 | 0.11 |
| 4,096 | `subspace_jet` | 14.73 | 14.60 | 11.79 | 327.49 | 3,870.45 | 0.04 |

**JAX, A100 80 GB** (`subspace_jax_a100.json`)

| `D` | default program | default | `subspace_jet` | `subspace_nested` | `taylor_jet` | `nested_autodiff` | default / best generic |
|---:|---|---:|---:|---:|---:|---:|---:|
| 64 | `taylor_jet` | 0.46 | – | – | 0.39 | 0.38 | 1.18 |
| 256 | `subspace_jet` | 1.00 | 0.94 | 0.75 | 0.88 | 0.66 | 1.53 |
| 1,024 | `subspace_jet` | 1.14 | 1.27 | 1.15 | 1.03 | 2.19 | 1.11 |
| 4,096 | `subspace_jet` | 1.33 | 1.61 | 1.42 | 3.27 | 21.00 | 0.41 |

**PyTorch, CPU** (`subspace_torch_cpu.json`)

| `D` | default program | default | `subspace_jet` | `subspace_nested` | `taylor_jet` | `nested_autodiff` | default / best generic |
|---:|---|---:|---:|---:|---:|---:|---:|
| 64 | `taylor_jet` | 29.09 | – | – | 28.40 | 12.05 | 2.42 |
| 256 | `subspace_jet` | 32.66 | 32.87 | 35.30 | 162.76 | 83.55 | 0.39 |
| 1,024 | `subspace_jet` | 69.70 | 69.41 | 70.52 | 2,715.08 | 1,032.95 | 0.07 |
| 4,096 | `subspace_jet` | 272.85 | 266.55 | 271.91 | 45,398.69 | 14,812.37 | 0.02 |

**PyTorch, A100 80 GB** (`subspace_torch_a100.json`)

| `D` | default program | default | `subspace_jet` | `subspace_nested` | `taylor_jet` | `nested_autodiff` | default / best generic |
|---:|---|---:|---:|---:|---:|---:|---:|
| 64 | `taylor_jet` | 10.37 | – | – | 12.93 | 6.36 | 1.63 |
| 256 | `subspace_jet` | 11.00 | 11.10 | 10.93 | 11.58 | 8.62 | 1.28 |
| 1,024 | `subspace_jet` | 11.97 | 10.46 | 8.74 | 20.61 | 52.30 | 0.58 |
| 4,096 | `subspace_jet` | 11.87 | 11.31 | 11.65 | 214.02 | 2,054.29 | 0.06 |

**Basis factorization, singular value decomposition against QR** (default program, ms per call; `subspace_svd_*.json` against `subspace_*.json`)

| `D` | JAX CPU, SVD → QR | JAX A100, SVD → QR | PyTorch A100, SVD → QR |
|---:|---:|---:|---:|
| 64 | 7.57 → 3.35 | 0.46 → 0.46 | 5.18 → 10.37 |
| 256 | 9.10 → 3.94 | 4.80 → 1.00 | 10.92 → 11.00 |
| 1,024 | 18.82 → 7.96 | 3.09 → 1.14 | 14.45 → 11.97 |
| 4,096 | 141.44 → 14.73 | 9.46 → 1.33 | 47.62 → 11.87 |

How to read it. The reduction pays from `D` of about 256 on a CPU and from 1,024
(PyTorch) or 4,096 (JAX) on an A100. At `D = 4,096` the default is 2.5 times faster than the
better generic engine on JAX/A100, 18 times on PyTorch/A100, 22 times on JAX/CPU and 54 times
on PyTorch/CPU. Below that it does not pay, and these are measured losses, not exceptions:
at `D = 64` the subspace programs are refused (the model has 64 weight rows, not fewer
than `D`) and the default is `taylor_jet`, which was 1.2 times (JAX, CPU and A100), 1.6
times (PyTorch/A100) and 2.4 times (PyTorch/CPU) slower than the better generic engine;
at `D = 256` and `1,024` on an A100 the default was 1.5 and 1.1 times (JAX) and 1.3 times at
`D = 256` (PyTorch) slower than it. Every program takes about a millisecond (JAX) or ten
milliseconds (PyTorch eager) in that range because the work is launch-bound, so the choice
between them is worth a few tenths of a millisecond. The planner has no calibrated cost
model for these devices yet and ranks by operation counts; `mode="find"` measures the
candidates and removes the loss.

The basis of the retained directions is held constant under differentiation (it is
computed from `stop_gradient` weights). First-order parameter gradients of the operator are
exact, because the dropped directions are annihilated by the first-layer matrices; second-order
parameter derivatives of the operator are not. The programs are refused, with the reading
operation named, when the input is read anywhere else, and when `rows >= q` (no reduction).

The basis is a thin QR factorization. An earlier version used a
singular value decomposition, which on an A100 cost 7 to 25 ms at `D = 4,096` against 0.9
to 2.7 ms for QR ([profile](benchmarks/planner/basis_profile_a100.json); the two
spans agree to 1e-14), and made the JAX default slower than the generic engines on an
A100 at every size: the last table above is the effect on the whole program. The PyTorch
A100 times at `D = 64` (a program unchanged between the two versions) differ by 2x between
the runs, which is the run-to-run variation of a launch-bound eager program on a shared
GPU node, and sets the resolution of every ratio near 1 in these tables.

## Dispatch regret

How close is the program the planner ranks first to the best one that was measured? A cost model was fitted on 152
calibration problems (seven operator sets, tanh MLPs, float64, one exclusive A100 80 GB), frozen by hash, and then scored once
on 110 held-out problems measured afterwards. The criteria were registered before any measurement
(`omnibias_experiments/dispatch_gate_20261022/PREREGISTRATION.md`); the committed record is
[`dispatch_gate_a100.json`](benchmarks/planner/dispatch_gate_a100.json).

**The gate failed on one of its four criteria, so no A100 calibration file is shipped** and the planner runs on its generic cost
model everywhere. A second round (below) failed on the same criterion.

| criterion | registered | measured | holds |
|---|---|---:|:-:|
| median regret | at most 1.10 | 1.00 | yes |
| worst regret | at most 1.5 | **64.0** | **no** |
| never slower than the generic fallback (lower95 of fallback / chosen time) | at least 0.95 | 2.74 | yes |
| cold start paid back within 100 calls, cached, for the cases that beat the fallback by 5 % | at least 90 % | 100 % | yes |
| 50-digit `mpmath` oracle, every algorithm of every set, CPU and A100 | 1e-10 (operators), 1e-9 (Gram) | 3.2e-15 | yes |

What the table does not show:

- Seven of 109 problems have regret above 1.5. One is `div/9` (divergence, D = 301, H = 256, B = 1,503): the model ranked
  `jacobian_trace` first, which took 5.4 ms against 84 us for `projected`. The calibration set had no problem in which
  `jacobian_trace` was more than 3.4 times slower, so the failure is an extrapolation, not noise. The others are 1.5 to 2.0 times
  and are mostly `projected` against `forward_laplacian` or `nested_autodiff` at small sizes.
- The generic model, the one the planner uses where no calibration is loaded, has median regret 1.00, 90th percentile 1.36, hit
  rate 62 % and worst regret 12.9; the frozen model has 90th percentile 1.23, hit rate 69 % and worst regret 64.0. The numbers
  are the same at the registered `expected_uses = 1000` and at the planner default of 1. Neither has a worst-case guarantee.
- Cached cold start: 100 % within 100 calls. Uncached: 82 % (under the 90 % the cached form had to reach), maximum 203,628 calls.
- The fallback criterion holds on average and not for every problem: ten problems chose a program more than 5 % slower than the
  fallback, the worst 0.51 times its speed (`lap1/8`).
- 110 of 112 held-out cells were scored; two were killed at the 25-minute limit with nothing measured, and one scored cell had
  no finite measurement and is left out of the regret statistics. Sixteen held-out cells ended at the 600 s per-program cap
  (usually in `forward_polylaplacian`), with later programs unmeasured; an unmeasured program that the model ranked first is
  scored as a failure, so the cap can hide a good choice but not a bad one.
- The library changed while the calibration was being measured, the harness gained an unregistered per-algorithm cap, and a first
  fit on incomplete data was discarded. `omnibias_experiments/dispatch_gate_20261022/DEVIATIONS.md` lists each one, with the times.

### Round two

The calibration of round one was timed on an older library state than its held-out set (the per-call dispatch overhead of the compiled
callables was cut in the middle of the calibration; `DEVIATIONS.md` D3), so the registered protocol (Addendum B) re-measured the
same 164 calibration problems on the final tree, refitted the model unchanged, froze it (hash `3befccff`) and scored it once on a
second held-out set of 112 problems that had never been drawn. The four criteria were not changed.

| criterion | registered | round two | holds |
|---|---|---:|:-:|
| median regret | at most 1.10 | 1.00 | yes |
| worst regret | at most 1.5 | **infinite** (first choice never measured); **23.9** without that cell | **no** |
| never slower than the generic fallback (lower95) | at least 0.95 | 2.86 | yes |
| cold start paid back within 100 calls, cached | at least 90 % | 100 % | yes |
| 50-digit `mpmath` oracle | 1e-10 / 1e-9 | 3.2e-15 (round one, algorithms unchanged) | yes |

- **The failure is the same one.** The 23.9 is `div/8` (divergence, D = 90, H = 256, B = 1578): `jacobian_trace` ranked first took
  2.68 ms and `projected` 0.112 ms. In round one it was `div/9` at 64. The largest `jacobian_trace` / `projected` ratio in the
  calibration was 3.4 in round one and 7.0 in round two, so the fit has nothing to extrapolate from. The infinite regret is `lap2/4`,
  where the program ranked first was never measured because the first program timed out at the 600 s cap; the registered rule scores that
  as a failure, and the verdict is the same without it.
- **The frozen model was better on most problems and worse in the worst case.** On the same 110 scored held-out problems its
  90th-percentile regret is 1.09 and its hit rate 68 %, against 1.23 and 56 % for the generic model, whose worst regret is 2.47.
- Six of 110 problems are above 1.5. The calibration cross-validation had already shown a worst regret of 7.0 when the model was frozen.
- Placement was not controlled: the calibration and held-out jobs ran on different A100 hosts, and dispatch overhead depends on the host CPU
  (`DEVIATIONS.md`, "Round two"). A third round was not run; the protocol registered two. No calibration file ships.

What to do with it. Use the default for the choice the model can make from operation counts, use `mode="find"` where a wrong
choice costs more than measuring (it picks the best of the programs it times and does not use the model), and read
`explain()["cost_model"]`, which says `"defaults"` when no calibration is loaded (`"defaults+cpu class prior"` on a CPU with JAX and
`"defaults+counted torch launches"` for PyTorch on an accelerator: constants set from the hardware or counted from the launched
kernels, not fitted to a calibration). The measured failure to look for is divergence
at large `H` and `B`, where the ranking chose `jacobian_trace` and it was 24 to 64 times slower than `projected` in the two held-out sets;
where that matters, use `mode="find"`. A replay of the current ranking over the recorded cells
([below](#replay-of-the-current-planner-over-the-recorded-cells)) picks `projected` in those two cells, which is not a held-out result.

### CPU class prior and counted PyTorch launches

Two device-class priors replace the A100-scale generic constants where no calibration file ships. Neither is fitted.

- **A CPU with JAX** ranks with constants set from the hardware of one core (`CPU_CLASS_PRIOR`: 2e10 float64 flop/s, 4e10 float32,
  1e-6 s per XLA thunk, 2e-5 s per LAPACK call, no device synchronization). The generic constants are A100-scale, so on a CPU the launch and
  solver terms decided a ranking in which the flops decide: the model's predictions for the `D > H` Laplacian cells were 0.17 to 2.6 ms
  where the measured times were 7 to 330 ms. The constants were fixed before the evaluation and evaluated once on two recorded sets
  of 76 Laplacian cells each (JAX, float64, one thread; `planner_probe` records, not re-measured). Dispatch regret, the measured time of the
  chosen program over the best measured time of the cell, recorded default against the prior: `D > H` cells (20 per set) median 1.31 and 1.61
  (forward, value-and-gradient) against 1.00 and 1.00 on the first set, with the 90th percentile 1.75 and 2.27 against 1.00 and 1.00; all 76 cells
  of the first set median 1.02 and 1.09 against 1.00 and 1.00, 90th percentile 1.52 and 1.92 against 1.20 and 1.18; the second set (a
  `differentiate="params"` ranking) median 1.03 and 1.10 against 1.00 and 1.03, 90th percentile 1.35 and 1.68 against 1.15 and 1.23. The
  program that wins nearly every `D > H` cell on a CPU is `projected_cholesky` (1.10 to 2.61 times faster than `projected` forward and 1.58 to
  3.36 times in value-and-gradient, 20 of 20 cells). **A trade ships with it**: the prior picks `nested_autodiff` at `D = 2`, width 32 and two
  layers, where the earlier model picked `projected` (regret 1.07 and 1.23 before, 2.97 and 1.90 after in value-and-gradient at `B = 1,024`, on the two sets), and several
  `D <= 8` ties between `projected` and `forward_laplacian` move by a few percent either way; `force=` and `mode="find"` select the other
  program, and the accelerator model is untouched. PyTorch on a CPU keeps the generic constants: its eager per-operation cost was not measured
  through the planner.
  **A second trade, the log-determinant ranking.** With the corrected work model of the log-determinant programs, the eight recorded CPU
  cells (JAX, float64, one thread; recorded, not re-measured) choose a different program in every cell. Dispatch regret, recorded default
  then corrected ranking: forward median 1.00 and 1.29, 90th percentile 3.77 and 6.20; value-and-gradient median 1.05 and 1.02, 90th
  percentile 5.26 and 2.18. The forward pass pays and the training pass gains, so a forward-only log-determinant call on a CPU can take
  the slower program; `force=` and `mode="find"` select the other one. The eight cells are not a held-out set.
- **PyTorch on an accelerator** counts the CUDA kernels one call launches (`TORCH_LAUNCH_SCALE`, the median of one `torch.profiler` window on an
  A100) where the registry counts the fused XLA programs, and charges the fused Triton kernel its measured cold compilation
  (`TORCH_COMPILE_SECONDS`, 5.5 s forward and 9 s with the gradient), over `expected_uses`. With those the planned default takes the kernel in float32
  inside the region it was measured in (width at most 64, `D` at most 16) once `expected_uses` is about ten thousand, and keeps the stream program
  below that.

### Round three: registered, not run

The planner changed after round two: it now ranks a training objective, carries per-call latency terms (a floor for
sequential work, a solver-call and a synchronization latency) and holds the stacked stream and fused-kernel programs.
Round two's statement that the algorithms were unchanged does not carry over, and neither does its frozen model.
A third round is therefore registered (`omnibias_experiments/dispatch_gate_20261022/PREREGISTRATION.md`, Addendum C, before any
measurement) with the same four criteria. It re-measures all 388 specifications drawn in rounds one and two on one wheel
snapshot, fits one model on forward and on value-and-gradient timings, freezes it, and scores it once on 112 held-out
specifications that have never been drawn, with the first three criteria required to hold **separately** in the forward and in
the value-and-gradient mode.

**It has not been run.** The harness was exercised on a CPU with shrunk problems only, and that is not evidence. Until it is run,
the statements above about rounds one and two describe an older planner, no calibration ships, and the dispatch of the current
planner has no registered held-out regret. A pass would ship an A100 calibration for JAX and float64 tanh chains only. The
`params2` objective is not measured by it, and its three-times-forward prediction stays an unvalidated assumption.

### Replay of the current planner over the recorded cells

The rounds above describe an older planner. Ranking the **current** tree's default (generic cost model, no calibration, `expected_uses`
1,000) over the programs that rounds one and two timed, with no compilation and no new timing, gives the regret below on the 382
distinct specifications (a specification timed in both rounds is scored on its round-two timing; the pooled count over 534 cells is
in the record). The protocol and its amendments were written before the replay and are in
`omnibias_experiments/dispatch_fix_20261101/` (`PROTOCOL.md`, `results/REPLAY.md`).

| current tree | median regret | 90th percentile | worst | cells above 1.5 |
|---|---:|---:|---:|---:|
| before the correction below | 1.000 | 1.293 | 4.25 | 17 |
| after it | 1.000 | 1.196 | 2.71 | 7 |

- **The divergence failure of rounds one and two is not reproduced.** The two cells in which the frozen model chose `jacobian_trace`
  and `projected` was 24 and 64 times faster are ranked correctly (regret 1.00); the divergence worst on the distinct basis is 1.40,
  with no cell above 1.5. No pick failed to run.
- **The worst case was the order-2 residual Gram, and one correction was adopted.** The planner ranked the projection program first in
  all 56 distinct cells; the program that forms explicit first-layer rows was the best measured one in 45. The projection program
  takes a QR of its first-layer metric matrix when it reduces the columns (`D` above the first hidden width): one `geqrf` and one `orgqr`
  library call, present in the compiled CPU program and in the compiled A100 program, absent from the other program. The work model did
  not charge it. It now charges the QR flops and two solver calls, at the cost model's generic solver price; no constant was changed
  and no other program's work model changed. The replay moves 19 picks, all from the projection to the explicit-rows program, and
  all of them improve. A second candidate that also priced the first-layer Gram per program ranked better on every summary
  statistic but raised two near-tie cells above 1.10, which the registered rule forbids, so it was not adopted.
- **Seven cells are still above 1.5**: three order-2 Gram cells (worst 1.66), three Laplacian cells (worst 2.71, the new overall worst)
  and one log-determinant cell (1.61). No count identified a mechanism for them. The current planner therefore still has a dispatch
  worst case above the 1.5 that the gate registered.
- **64 cells cannot be scored.** The Laplacian program the records call `projected` is `projected_cholesky` in the current tree; the
  current `projected` (stacked streams) was never timed in these rounds, so the replay says nothing about when it should be chosen.
- **This is an in-sample replay of older records and not a held-out gate.** The cells that exposed the defect are the cells that score
  the correction, and it was chosen after a first variant failed the rule. Round three is the held-out test and has not been run.

## End-to-end training

Per-call speed is not training time. Two pre-registered suites trained the same 13 problem families, and the second repairs what made the
first uninformative at the dimensions the planner was built for. Registrations, raw per-seed traces and scoring are in
`omnibias_experiments/` (`e2e_20261022/` for suite v1, `e2e_v2_20261024/` for suite v2); the committed record of suite v2 is
`docs/benchmarks/planner/e2e_suite_v2_a100.json`, and its rows are in the [full-scope matrix](performance.md#full-scope-matrix).

### Suite v2

Suite v2 was registered before any measurement (`SUITE_V2.md`, with file hashes). It trains 265 pairs on 13 JAX problems from `D = 2` to
`D = 10,000` (networks 128 to 256 wide and four layers deep, batches of 1,024 to 2,048, 20,000 to 30,000 Adam steps, 900 s per cell, five
seeds, one A100 80 GB, float64 as the primary run and float32 at `highest` precision as a secondary one): Adam, L-BFGS, Gauss-Newton, ENGD, KFAC, SPRING and trust-region Levenberg-Marquardt on the structured or
closed-form omnibias program (the candidates), against nested autodiff, JAX jet, `folx`, sparse `folx`, a dense Hessian, Hutchinson,
SDGD and STDE (the baselines; every second-order optimizer also runs on an explicit Jacobian through each exact engine, and those cells are
baselines too). Each optimizer has three hyperparameter grid points on one tuning seed, and a stronger setting of a baseline may exist. A stage
run first marks a target **not eligible** when no Adam grid point reaches it within the budget; such a row is listed and never scored.

Two amendments are part of the registration, and both change what the numbers mean. Addendum A, written before any v2 cell but after suite
v1's outcome was known, made the primary comparison the time to target **with compile time excluded**, against every baseline including the
stochastic estimators (a compile is paid once and a training run lasts much longer than these cells), kept the cold-compile comparison of the
original registration, and added the SPRING and trust-region pairs and an exhaustive-find engine, `planned_find` (`mode="find"`,
`top_k=None`, with Adam and L-BFGS). Because it was written knowing v1's outcome, the choice of the primary comparison is not independent
of it, and every claim below from it says "compile excluded". Amendment B restarted the campaign from empty results after the first was
lost to a disk-quota fault and the removal of the experiment tree; that first campaign is withdrawn. The headline is the registered one: a
row meets it when the one-sided 95 % bound of the per-seed ratio (baseline time / candidate time) is at least 2, the strongest pair of each
side being chosen on the seeds that are compared.

Rows by comparison (39 problem-target rows per comparison, 32 of them eligible):

| comparison | meets the bound | undecided | baseline faster or equal | loss (bound below 0.5) | no ratio (censored or no pair reached it) |
|---|---:|---:|---:|---:|---:|
| compile excluded, every baseline (primary) | 17 | 7 | 3 | 4 | 1 |
| compile excluded, exact baselines only | 19 | 5 | 3 | 2 | 3 |
| cold compile included, every baseline | 11 | 13 | 3 | 4 | 1 |
| warm compile included, every baseline | 15 | 9 | 3 | 4 | 1 |
| cold compile included, exact baselines only | 14 | 10 | 3 | 2 | 3 |
| the 175 registered pairs, cold compile included (the original rule) | 11 | 11 | 4 | 4 | 2 |
| the 175 registered pairs, compile excluded | 16 | 6 | 4 | 4 | 2 |

**With compile excluded, 17 of 32 eligible rows meet the headline, on 7 of the 13 problems.** In every one of them the candidate is a
second-order optimizer (Gauss-Newton, SPRING, trust-region Levenberg-Marquardt or ENGD) on the structured Gram. No first-order candidate (Adam
or L-BFGS on a closed-form, projected or planned program) meets the bound in any row. Time to target in seconds, medians over five seeds:

| problem | target | candidate | s | strongest baseline | s | ratio | lower 95 % bound |
|---|---:|---|---:|---|---:|---:|---:|
| `allen_cahn` | 0.03 | `lmtr:structured` | 0.389 | `adam:jet` | 2.87 | 7.30 | 5.96 |
| `allen_cahn` | 0.01 | `lmtr:structured` | 0.567 | `adam:jet` | 5.69 | 8.93 | 7.42 |
| `allen_cahn` | 0.003 | `lmtr:structured` | 0.973 | `adam:jet` | 15.5 | 14.99 | 10.81 |
| `cahn_hilliard` | 0.01 | `spring:structured` | 3.49 | `adam:jet` | 49.2 | 15.36 | 11.75 |
| `heat1d` | 0.03 | `lmtr:structured` | 0.179 | `lbfgs:jet` | 1.03 | 5.35 | 4.46 |
| `heat1d` | 0.01 | `gn:structured` | 0.318 | `adam:jet` | 2.04 | 6.61 | 6.28 |
| `heat1d` | 0.003 | `engd:structured` | 0.455 | `adam:jet` | 4.01 | 7.16 | 4.96 |
| `heat_hd_d100` | 0.1 | `spring:structured` | 2.75 | `adam:hutch` | 6.87 | 2.56 | 2.30 |
| `heat_hd_d100` | 0.03 | `spring:structured` | 6.88 | `adam:sdgd` | 45.4 | 6.18 | 5.62 |
| `heat_hd_d100` | 0.01 | `spring:structured` | 14.9 | `adam:folx_sparse` | 154 | 10.24 | 9.27 |
| `helmholtz` | 0.03 | `gn:structured` | 0.981 | `adam:folx` | 9.02 | 8.85 | 7.50 |
| `helmholtz` | 0.01 | `gn:structured` | 1.01 | `adam:folx` | 12.5 | 11.91 | 10.20 |
| `helmholtz` | 0.003 | `engd:structured` | 1.09 | `adam:folx` | 18.6 | 10.96 | 4.74 |
| `hjb_d100` | 0.03 | `gn:structured` | 2.75 | `lbfgs:folx` | 8.28 | 2.84 | 2.35 |
| `hjb_d100` | 0.01 | `gn:structured` | 21.2 | `adam:hutch` | 147 | 7.51 | 6.49 |
| `poisson_d100` | 0.03 | `spring:structured` | 7.04 | `adam:sdgd` | 68 | 9.75 | 9.16 |
| `poisson_d100` | 0.01 | `spring:structured` | 11.2 | `spring:folx` | 529 | 46.81 | 43.92 |

These ratios are 2.56 to 46.81 (geometric means) with compile excluded. In 13 of the 17 rows the candidate compiles slower and is not
faster per step (a second-order step is heavier), so no per-step break-even exists and the saving is in the number of steps; in the other 4
it compiles no slower (break-even 0). Its extra cold compile is at most 3.0 s, and after paying it the saving at the target is 0.2 s to
518.4 s, positive in every row. Where compile time is included the same pairs reach the bound in fewer rows (11 cold, 15 warm), because the
winning cells of the low-dimensional problems train for 0.2 to 1.1 s, which is shorter than the compile that precedes them.

**Where it does not hold.** Seven rows are losses or parity for the candidate, with compile excluded:

| problem | target | verdict | candidate | strongest baseline | ratio | lower 95 % bound |
|---|---:|---|---|---|---:|---:|
| `heat_hd_d1000` | 0.1 | loss | `spring:structured` | `adam:stde` | 0.25 | 0.23 |
| `heat_hd_d1000` | 0.03 | loss | `adam:planned_find` | `adam:sdgd` | 0.46 | 0.43 |
| `logdet_flow_d16` | 0.1 | baseline faster or equal | `adam:omnibias` | `adam:dense` | 0.99 | 0.99 |
| `logdet_flow_d16` | 0.03 | baseline faster or equal | `adam:omnibias` | `adam:dense` | 0.98 | 0.98 |
| `logdet_flow_d16` | 0.01 | baseline faster or equal | `adam:omnibias` | `adam:dense` | 0.98 | 0.98 |
| `triharmonic` | 0.1 | loss | `adam:projected` | `adam:jet` | 0.16 | 0.15 |
| `triharmonic` | 0.03 | loss | `adam:projected` | `adam:jet` | 0.17 | 0.16 |

- At `D = 1,000` the stochastic estimators reach the looser targets sooner than any exact pair (STDE at 0.1, SDGD at 0.03). Against exact
  baselines only, a row there meets the bound.
- The composed sixth-order residual of `triharmonic` (a 2-D problem) trains 6 times slower through the projected program than through JAX
  jet under Adam, and no pair reached 0.01. The suite does not establish why.
- On `logdet_flow_d16` the closed-form program is level with the dense Hessian (0.98 on time to target), so training time is level too; that
  is not a win. On `divflow_d64` it is 1.37 times as fast to 0.1 and 0.03 (the bound is also 1.37, below 2), which is not a headline.
- Seven further rows are undecided, with a bound below 2 (`cahn_hilliard` at 0.1 and 0.03, `divflow_d64` at 0.1 and 0.03, `hjb_d100`,
  `poisson_d100` and `poisson_d10000` at 0.1), and seven are not eligible: `divflow_d64` at 0.01, `heat_hd_d1000` at 0.01, `poisson_d10000`
  at 0.03 and 0.01 and `hjb_d1000` at all three targets. At `hjb_d1000` no Adam pair reached 0.1 within the cap and no L-BFGS pair did in more than
  two of five seeds, while the structured Gauss-Newton, SPRING and ENGD pairs did in every seed, in 8 to 15 s of training; that row has no
  ratio because the target is not eligible, and it is not a headline.
- Resource refusals are recorded as errored pairs, not as slow ones: `folx` and sparse `folx` at `poisson_d10000` (98 to 372 GiB asked of
  80 GiB) and the explicit-Jacobian nested pairs at `triharmonic` (125 GiB).
- The `planned_find` engine (the exhaustive measured find) planned in a median of 4.0 s (at most 9.9 s) from an empty plan cache and in 0.25 s
  from a filled one; its median step was within 1.15 of the best exact Adam engine on 6 of 9 problems, and the registered prediction that it
  would be on 7 of 9 missed. It is the candidate in no row that meets the bound.

**Float32 at `highest` precision (secondary, incomplete).** The float32 campaign is complete for 10 of the 13 problems and is not for
`heat1d`, `allen_cahn` and `cahn_hilliard`: their jobs stopped part-way, after 4.8 to 4.9 hours, the interval at which the workload administrator had
terminated the float64 jobs of the same problems (the float32 job logs were in a working directory that was later removed, so the cause is
inferred). The experiment environment and wheel snapshot were removed before the jobs could be resubmitted. They hold 6, 9 and 49 fewer
records than their float64 runs, so later seeds of several pairs are missing, and no float32 row is reported for them or counted anywhere. Over the 10 complete problems (30 rows, 22 eligible), 8 eligible rows meet the headline with compile
excluded and 8 with the cold compile included, all with a structured second-order candidate (SPRING, trust-region Levenberg-Marquardt or
Gauss-Newton): `heat_hd_d100` at 0.03 and 0.01, `helmholtz` at all three targets, `hjb_d100` at 0.03 and 0.01 and `poisson_d100` at 0.03, with
ratios 3.09 to 26.52 (geometric means) and lower bounds 2.60 to 23.19 with compile excluded. These are 8 of the 10 float64 headline rows of the
same ten problems. The other two do not repeat: `heat_hd_d100` at 0.1 has a ratio of 1.67 (bound 1.52) against `adam:sdgd`, and the baseline did
not reach `poisson_d100` at 0.01 in every seed, so that row has no ratio. The losses and parity rows are where float64 had them: `heat_hd_d1000`
at 0.1 and 0.03 against STDE and SDGD (0.45 and 0.37), `logdet_flow_d16` at all three targets (0.76 to 0.95) and `triharmonic` at 0.1 (0.75).
Float32 meets the bound on no problem on which float64 does not. The per-row ratios are in `docs/benchmarks/farm_time_to_target_v2.json`.

Of 19 registered predictions (10 of the original registration, 9 of Addendum A), 9 hit, 7 missed and 3 were not scored: the registered
scorer expects 12 problems with a structured Gauss-Newton pair and an eligible tightest target and 11 have both, so the three predictions
of SPRING and trust-region steps against Gauss-Newton record no verdict (the counts it computed on the 11 are: SPRING on the structured Gram
ahead of Gauss-Newton on 4, ahead of the best explicit-Jacobian SPRING on 7, and the trust-region step ahead of Gauss-Newton on 1). The float32
prediction (every float32 cell at least as fast per step as its float64 counterpart, and no more eligible targets) missed on the 10 complete
problems: 113 of 192 pairs were slower per step in float32, by up to 1.88 times (`lmtr:omnibias` at `divflow_d64`), although float32 had no
more eligible targets (22 against 23). The other misses are in the counts of rows that meet the bound on the low-dimensional
problems and at `D = 100`, which exceeded the prediction; at `D >= 1,000`, where fewer problems had an eligible row than predicted; the
step ratio of the divergence-free flow; the break-even in steps (prediction A2: it does not exist when the candidate's step is slower, so no
row was below the baseline's step count); and the planned-find step time.

The statement this supports is narrow. **With compile excluded, float64, on one A100, structured second-order optimizers reach the targets
of the table above sooner than every baseline pair, on 7 of 13 problems (Allen-Cahn, Cahn-Hilliard at its tightest target, 1-D heat,
Helmholtz, and the `D = 100` heat, HJB and Poisson problems).** They do not on the log-determinant and divergence-free flows, on
`triharmonic`, or against the stochastic estimators at `D = 1,000`, and the suite cannot rank anything at `poisson_d10000` or `hjb_d1000`.
What makes the difference is the cost of the second-order step: the strongest baseline is an Adam or L-BFGS pair in 16 of the 17 rows and an
explicit-Jacobian SPRING pair in one, so the same optimizers on an explicit Jacobian are slower, and on the structured Gram they are faster.
A faster first-order step is not what wins. The incomplete float32 run on ten problems agrees with this and does not extend it.

### Suite v1

A suite of 13 problems with a registered headline rule was trained with Adam, L-BFGS,
Gauss-Newton, ENGD and KFAC pairs against nested, JAX-jet, `folx`, sparse `folx`, Hutchinson, SDGD and STDE baselines (three seeds,
three accuracy targets, one A100 80 GB, JAX). The headline is time to target, compile time included, against the strongest baseline, with
the lower 95 % bound of the per-seed ratio at least 2. The registration, the raw per-seed traces and the scoring are in
`omnibias_experiments/e2e_20261022/` (`SUITE.md`, `SUITE_RESULTS.md`, `PREDICTIONS.md`).

**No row of the registered comparison meets the headline** (0 of 39 problem-target rows).

- 6 rows are losses (lower bound below 0.5), 13 are rows in which no pair reached the target in every seed, and 11 rows reach the
  bound when compile time is excluded. The compile-excluded comparison is not the registered one.
- At the registered budgets no pair reached even the loosest target on `poisson_d10000` and `hjb_d1000`, the high
  dimensions the planner was built for, so the suite says nothing about them.
- Of eight registered predictions, two hit, five missed and one was not scored.
- On PyTorch, a subset of two Poisson problems (`D = 100`, `D = 1000`; Adam; eager; untuned) also has no headline: the planner's choice is
  indistinguishable from the collapsed Taylor-mode library (`jet`) at `D = 100`, and no engine reached the loosest target at `D = 1000`.
  Per step it takes 5.8 ms and 0.20 GiB against 22.9 ms and 3.90 GiB for the dense Hessian trace at `D = 1000`, which is a per-call
  result and not a training one.

Treat the planner as a derivative primitive that is exact and differentiable in the parameters, with per-call wins and losses listed in
[performance](performance.md), and not as a demonstrated end-to-end training speedup.

Suite v2 was built to answer what suite v1 could not at the dimensions the planner was built for (see above). Read together, the two suites
support the second-order result of suite v2 for the structured step, and do not support a training speedup from the per-call programs on a
first-order optimizer: in neither suite does an Adam or L-BFGS candidate meet the registered bound against the strongest baseline.

## Limits

- The planner chooses among programs; it does not make an unsupported model supported.
  A model that reads its input outside weight products runs on the generic engines, which
  cost what they cost.
- A heuristic plan can be slower than another feasible program. The cost model is
  calibrated per device only where a calibration file ships, and none does yet (the A100 gate failed twice and a third round
  is registered but not run, see [dispatch regret](#dispatch-regret)); a replay of the current ranking over the recorded cells
  still has a worst regret of 2.71 and seven cells above 1.5, and that replay is in-sample. It counts operations with launch, sequential-work,
  solver-call and synchronization terms whose base values come from the defaults, not from a measurement of your device.
  `mode="find"` measures instead, but by default only for the two leading candidates (`top_k=2`), so a program the cost model
  ranks third is never timed unless you pass `top_k=None` (the training preset of `planned_evaluator`): on PyTorch in float32 at `D = 2` and the default `expected_uses` of 1,000 that is the fused kernel, 2.2 to 3.1
  times faster than the default on device time once compiled (its cold compilation of 4 to 7 s does not amortize over 1,000
  uses; pass a larger `expected_uses` or `force="fused_streams"`, see the fused bullet above).
- An unlimited `memory_budget` lets `mode="find"` and `mode="auto"` time candidates whose buffers no card holds, and XLA can
  end the process while compiling one (a segmentation fault on a candidate it sized at 2.2 TB, JAX 0.10.2, order-four
  residual at `D = 1024`, `B = 2048`); a finite budget (the default is 75 % of device memory on an accelerator) excludes the
  candidates the memory model sizes above it, and `safe_find=True` times each candidate in its own process.
- A PyTorch call made eagerly costs about a millisecond of host time whatever the program, so eager timings of programs that
  run for 0.1 ms on the device are mostly that cost; the device time shows only under CUDA-graph replay
  (`compile_training_step`, or a captured evaluation).
- `differentiate="params2"` ranks by three times the forward work and was never compared with a measured second derivative.
  The fused kernels differentiate once into the parameters and are refused for it.
- The stream programs cover dense chains. A gated or PirateNet-style model is not one, so it runs on the reduced or generic
  engines and has no stream program; the PyTorch projected residual streams are not provided; the PyTorch projected `Delta^2`
  keeps dense Hessian storage.
- On a model written as a plain function, the `taylor_jet` program uses `jax.experimental.jet`, which
  raises `UnexpectedTracerError` on `jnp.logaddexp` (so on `jax.nn.softplus`) in JAX 0.10.0 and 0.10.2. The
  planner then refuses that program at trace time and runs another; `force="taylor_jet"` raises a `ValueError`
  that names the refusal. `jax.nn.sigmoid` and `jax.nn.silu` are not affected.
- The subspace programs hold their retained directions constant for differentiation
  (see above); only first-order parameter gradients are exact.
- Compile time is separate from steady-state time. `explain()` reports predicted compile
  cost; `bucket` and a persistent `cache` reduce it, and neither is on by default.
- The measured records are re-checkable with a reproduction kit kept outside this repository (`omnibias_experiments/repro_kit_20261022/`:
  lockfiles, the wheels that were measured, one command per claim, the raw records and a verifier with pre-registered bands). One
  internal replication has been run, by the same organisation on one shared H100, once: every audit reproduced, three of five
  replication bands were met, and two were not (three precision records over a band set from one A100 run, and two step-time ratios
  measured while the card was shared). Exclusive-device repeats on an A30 met them; the same repeats on an H100 were cancelled unrun in the queue, so the
  planner's speed on an H100 is unmeasured. It is not an independent replication.
- Exact means exact in the arithmetic used. Direct formulas can lose digits to cancellation,
  and the Gauss-Newton step has error about `kappa * eps` with `kappa <= (lambda_max + d) / d`;
  the conditioning report and the `"floor"` policy make that visible and bounded.
