# Interop: folx, DeepXDE, PINA and third-party engines

The planner is only useful to code that already exists if that code can call it, and it should not be a closed list of programs.
This page covers four connections: a `folx`-compatible `forward_laplacian`, any third-party engine registered as a planner
executor, and adapters for DeepXDE and PINA. Each states what is the same, what is not, and how it was checked. None of them
changes what is computed; they change which program computes it.

## A `folx`-compatible `forward_laplacian`

`omnibias.jax.folx_compat.forward_laplacian` accepts the call and returns the fields of `folx.forward_laplacian`: for every
output a `FwdLaplArray` with `.x`, `.jacobian` (dense, `(n_inputs, *output_shape)`) and `.laplacian`, `weights` as a prefix
tree of the arguments, pytree arguments and outputs.

```python
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from omnibias.jax.folx_compat import forward_laplacian

rng = np.random.default_rng(0)
W1, W2 = jnp.asarray(rng.standard_normal((20, 12)) / 4), jnp.asarray(rng.standard_normal((12, 12)) / 4)
w3 = jnp.asarray(rng.standard_normal(12) / 4)


def f(x):
    return jnp.tanh(jnp.tanh(x @ W1) @ W2) @ w3


x = jnp.asarray(rng.standard_normal(20))
wrapped = forward_laplacian(f)
out = wrapped(x)  # value, Jacobian and Laplacian of f at x, as folx returns them
print(out.x, out.dense_jacobian.shape, out.laplacian)
np.testing.assert_allclose(out.laplacian, jnp.trace(jax.hessian(f)(x)), rtol=1e-10)
print(wrapped.explain(x)["selected_strategy"])  # the planner's choice for this model

batch = jnp.asarray(rng.standard_normal((6, 20)))
lap = jax.jit(jax.vmap(wrapped))(batch).laplacian  # the usual folx pattern: vmap over points
np.testing.assert_allclose(lap, jax.vmap(lambda z: jnp.trace(jax.hessian(f)(z)))(batch), rtol=1e-10)
```

What differs from `folx`:

- The Laplacian of a scalar `fn` of one vector argument is planned (a recognised dense chain gets the program the cost model
  picks, any other model the planner's generic executor) whenever `fn`'s weights are concrete arrays, also under `vmap` and
  `jit`. Under `vmap` one batched plan is made for the whole batch (a `custom_vmap` rule). If the weights are tracers of an outer
  transformation (a gradient through a closed-over parameter) that rule is skipped and a per-point plan is used, or nested
  autodiff if no plan can be built. `wrapped.explain(x)` shows the plan.
- Every other call (several arguments, pytree or array outputs, weights that are tracers of an outer `grad`, keyword arguments,
  non-unit `weights`) uses nested autodiff. It is exact; it has none of `folx`'s propagation rules and is not faster than
  `folx` there.
- The Jacobian is always dense; `sparsity_threshold` is accepted and ignored. `folx`'s custom-rule registration has no
  counterpart, and the returned types are not `folx`'s own.

`tests/test_folx_compat.py` compares value, Jacobian and Laplacian with the dense Hessian on single-argument chains, several
arguments with `weights`, pytree outputs, `vmap`, `jit` and a gradient through a closed-over parameter, and with `folx` itself
(`folx 0.2.30`) when it is installed: agreement is at roundoff in float64. Whether it is *faster* is a separate question: see
[measured](#measured) below.

## Any engine as a planner executor

An engine that is not part of omnibias can compete in a plan. `register_engine(name, operator, engine, work=...)` adds it to the
planner's registry and to the back end's executors. The engine contract is narrow: it receives the pointwise model `f` and a
batch `x` and returns the operator at every point. The planner then refuses it, with a reason, for requests it cannot serve (a
metric, a coordinate subset, drift directions, a higher Laplace power than declared), prices it with a work model you declare
(`work` is a function or the name of a registered algorithm to be priced like), can time it with `mode="find"`, and names it in
`explain()`. Calibrate it per device through `CostModel.efficiency` like a built-in.

```python
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from omnibias.jax.operators import PlanCache, compile_operator, register_engine, unregister_engine

rng = np.random.default_rng(1)
params = [
    (jnp.asarray(rng.standard_normal((9, 6)) / 3), jnp.zeros(6)),
    (jnp.asarray(rng.standard_normal((6, 1)) / 3), jnp.zeros(1)),
]


def model(p, x):
    return (jnp.tanh(x @ p[0][0] + p[0][1]) @ p[1][0] + p[1][1])[:, 0]


X = jnp.asarray(rng.standard_normal((5, 9)))


def hessian_trace(f, x, power=1):  # any callable with this contract
    return jax.vmap(lambda z: jnp.trace(jax.hessian(f)(z)))(x)


register_engine("hessian_trace", "laplacian", hessian_trace, work="nested_autodiff")
try:
    own = compile_operator(model, (params, X), operator="laplacian", cache=PlanCache())
    mine = compile_operator(model, (params, X), operator="laplacian", force="hessian_trace", cache=PlanCache())
    print(own.explain()["selected_strategy"], "vs", mine.explain()["selected_strategy"])
    np.testing.assert_allclose(mine(params, X), own(params, X), rtol=1e-10)
finally:
    unregister_engine("hessian_trace", "laplacian")
```

`generic=True` also makes the engine available for models that are not a recognised dense chain (after the built-in generic
executors, or by `force`). PyTorch has the same function in `omnibias.torch.operators`. `omnibias.jax.folx_compat.register_folx_engine()`
registers the installed `folx` this way; `tests/test_engines.py` forces it and checks it against the planner's own program.
Registration is process-global, and a registered name that clashes with a built-in program is refused. Only `laplacian` and
`divergence` can be served through this contract; the residual operators, `logdet` and `solve` have programs that need more than
a pointwise function.

**`folx` is a candidate whenever it is installed.** On JAX the first `compile_operator` call registers the installed `folx`
(`register_folx_engine()`) with the planner, so a Laplacian request is priced against it, in the cost model like
`forward_laplacian`, and `mode="find"` times it when it is among the leading candidates and keeps it if it is fastest. It appears in
`explain()["candidates"]` under the name `folx`. In heuristic mode the choice is only as good as the cost model's ranking of `folx`,
which no calibration has measured per device. Set `OMNIBIAS_FOLX_ENGINE=off` to keep it out of
the pool; an engine of your own registered under the name `folx` is kept in preference to it. Nothing is registered when `folx` is not
installed, and omnibias does not import it otherwise.

## DeepXDE

`DeepXDEOperators(net)` takes the DeepXDE `FNN` (PyTorch backend) and gives `laplacian(x, component=0)` and `divergence(x)` with
DeepXDE's shapes. DeepXDE forms its derivatives from the graph between `y` and `x`, which a `pde(x, y)` function cannot replace
from the inside, so the adapter takes the network itself.

<!-- docs-test: skip reason="needs DeepXDE, which is not a dependency of omnibias" -->
```python
import deepxde as dde
from omnibias.torch.deepxde_adapter import DeepXDEOperators

net = dde.nn.FNN([2, 64, 64, 64, 1], "tanh", "Glorot normal")
ops = DeepXDEOperators(net)


def pde(x, y):
    return ops.laplacian(x) - forcing(x)  # replaces dde.grad.hessian(y, x, i=0, j=0) + dde.grad.hessian(y, x, i=1, j=1)


data = dde.data.PDE(geometry, pde, bcs, num_domain=2000, num_boundary=200)
model = dde.Model(data, net)
```

A plain `FNN` with a known activation is read as a dense chain; the plan reads the live weights, so optimizer steps are seen. A
network with an input or output transform (a hard boundary condition, for instance) or an unknown activation runs on the
generic autodiff executors and `explain()["adapter_fallback"]` says why; there it is no faster than DeepXDE. Mixed second
derivatives stay with `dde.grad`. `tests/test_interop_adapters.py` checks the Laplacian and divergence against `dde.grad`, the
parameter gradients of a loss built on them, the generic route, and that a small Poisson problem trains to the same loss history
(`rtol 1e-7`) with and without the adapter (DeepXDE 1.15.0, PyTorch backend, float64). Other DeepXDE backends are not supported.

## PINA

`omnibias.torch.pina_adapter.laplacian(model, input_, components, d, output_labels=...)` has the interface of
`pina.operator.laplacian` with the model where PINA takes the computed output, and returns the same labelled tensor (`dd<name>`).

<!-- docs-test: skip reason="needs PINA, which is not a dependency of omnibias" -->
```python
from omnibias.torch.pina_adapter import laplacian


def poisson(input_, output_):
    lap_u = laplacian(model, input_, components=["u"], d=["x", "y"], output_labels=["u"])
    return lap_u - forcing(input_)
```

`model` is the network (a `FeedForward` or an `nn.Sequential` of `Linear` and activation modules); another module runs on the
generic executors. `d` restricts the Laplacian to a subset of inputs as in PINA. The test compares with PINA's operator on all
and some inputs and multiple components, the labels, parameter gradients, and a module that is not a chain (PINA
`0.3.2.post2610`). Only the Laplacian is provided.

## Measured

Timings are from `omnibias_experiments/interop_20261022` (`bench_jax.py`, `bench_adapters.py`; tables rendered by `make_tables.py` from
the raw records). Every row is one fresh process; nothing is tuned per method.

<!-- INTEROP RESULTS BEGIN -->

**`folx.forward_laplacian` against the drop-in and the planner, CPU** (`cpu:0`, JAX 0.10.2, folx 0.2.30, float64, tanh MLP `D -> 64 -> 64 -> 64 -> 1`, `jit(vmap(...))`, median of 9 calls; first call includes compilation).

| D | batch | method | plan | first call s | steady ms | steady vs folx | max deviation from folx |
|---:|---:|---|---|---:|---:|---:|---:|
| 16 | 256 | folx | - | 0.34 | 8.53 | 1 (reference) | 0.0e+00 |
| 16 | 256 | folx_compat | - | 3.68 | 6.89 | 1.24x faster | 5.6e-16 |
| 16 | 256 | planner | forward_laplacian | 0.21 | 5.21 | 1.64x faster | 1.1e-15 |
| 128 | 256 | folx | - | 0.19 | 42.80 | 1 (reference) | 0.0e+00 |
| 128 | 256 | folx_compat | - | 4.57 | 21.31 | 2.01x faster | 8.9e-16 |
| 128 | 256 | planner | projected | 0.56 | 23.36 | 1.83x faster | 8.9e-16 |
| 1024 | 256 | folx | - | 0.40 | 174.99 | 1 (reference) | 0.0e+00 |
| 1024 | 256 | folx_compat | - | 3.45 | 24.22 | 7.22x faster | 1.3e-15 |
| 1024 | 256 | planner | projected | 0.71 | 24.83 | 7.05x faster | 1.3e-15 |
| 4096 | 64 | folx | - | 0.74 | 267.59 | 1 (reference) | 0.0e+00 |
| 4096 | 64 | folx_compat | - | 3.83 | 6.54 | 40.94x faster | 1.6e-15 |
| 4096 | 64 | planner | projected | 0.82 | 7.26 | 36.87x faster | 1.7e-15 |

**`folx.forward_laplacian` against the drop-in and the planner, A100-80GB** (`cuda:0`, JAX 0.10.2, folx 0.2.30, float64, tanh MLP `D -> 64 -> 64 -> 64 -> 1`, `jit(vmap(...))`, median of 9 calls; first call includes compilation).

| D | batch | method | plan | first call s | steady ms | steady vs folx | max deviation from folx |
|---:|---:|---|---|---:|---:|---:|---:|
| 16 | 256 | folx | - | 1.91 | 0.20 | 1 (reference) | 0.0e+00 |
| 16 | 256 | folx_compat | - | 8.41 | 0.18 | 1.11x faster | 6.7e-16 |
| 16 | 256 | planner | forward_laplacian | 0.60 | 0.22 | 1.12x slower | 6.7e-16 |
| 128 | 256 | folx | - | 2.53 | 0.32 | 1 (reference) | 0.0e+00 |
| 128 | 256 | folx_compat | - | 10.65 | 0.36 | 1.12x slower | 7.8e-16 |
| 128 | 256 | planner | projected | 1.07 | 0.50 | 1.54x slower | 7.8e-16 |
| 1024 | 256 | folx | - | 2.59 | 1.23 | 1 (reference) | 0.0e+00 |
| 1024 | 256 | folx_compat | - | 8.73 | 0.41 | 2.99x faster | 2.7e-15 |
| 1024 | 256 | planner | projected | 1.11 | 0.45 | 2.75x faster | 2.7e-15 |
| 4096 | 64 | folx | - | 2.38 | 1.59 | 1 (reference) | 0.0e+00 |
| 4096 | 64 | folx_compat | - | 10.22 | 0.38 | 4.16x faster | 3.6e-15 |
| 4096 | 64 | planner | projected | 1.21 | 0.42 | 3.81x faster | 3.8e-15 |

**DeepXDE and PINA operators against the adapters, CPU** (`cpu`, PyTorch 2.9.0+cu128, DeepXDE 1.15.0, float64, tanh MLP width 64 depth 3 (the same module for both), loss `mean(laplacian^2)`, timed call = loss and its gradient with respect to the network parameters, median of 9 calls; PINA 0.3.2.post2610, version not stored in the record).

| library | D | batch | library ms | adapter ms | adapter vs library | plan | max deviation |
|---|---:|---:|---:|---:|---:|---|---:|
| deepxde | 2 | 1024 | 27.78 | 36.77 | 1.32x slower | projected | 3.9e-16 |
| pina | 2 | 1024 | 32.60 | 59.32 | 1.82x slower | - | 1.0e-16 |
| deepxde | 10 | 1024 | 101.83 | 76.66 | 1.33x faster | projected | 5.8e-16 |
| pina | 10 | 1024 | 113.28 | 103.51 | 1.09x faster | - | 6.2e-17 |
| deepxde | 50 | 512 | 403.00 | 92.04 | 4.38x faster | forward_laplacian | 1.8e-15 |
| pina | 50 | 512 | 488.99 | 116.83 | 4.19x faster | - | 6.9e-17 |

**DeepXDE and PINA operators against the adapters, A100-80GB** (`cuda`, PyTorch 2.9.0+cu128, DeepXDE 1.15.0, float64, tanh MLP width 64 depth 3 (the same module for both), loss `mean(laplacian^2)`, timed call = loss and its gradient with respect to the network parameters, median of 9 calls; PINA 0.3.2.post2610, version not stored in the record).

| library | D | batch | library ms | adapter ms | adapter vs library | plan | max deviation |
|---|---:|---:|---:|---:|---:|---|---:|
| deepxde | 2 | 1024 | 5.38 | 5.24 | 1.03x faster | projected | 2.8e-16 |
| pina | 2 | 1024 | 7.83 | 13.56 | 1.73x slower | - | 9.2e-17 |
| deepxde | 10 | 1024 | 18.97 | 6.39 | 2.97x faster | projected | 8.9e-16 |
| pina | 10 | 1024 | 37.33 | 16.78 | 2.23x faster | - | 8.3e-17 |
| deepxde | 50 | 512 | 88.79 | 4.87 | 18.22x faster | forward_laplacian | 2.2e-15 |
| pina | 50 | 512 | 119.55 | 9.09 | 13.16x faster | - | 6.9e-17 |

<!-- INTEROP RESULTS END -->

**How to read the folx table.** `folx` here is `folx.forward_laplacian` with its default arguments, which seeds a dense `D`-row
Jacobian. The planner row is the planner's own choice (`plan` column: `forward_laplacian` at `D = 16`, `projected` from `D = 128`).
`projected` carries `H` rows instead of `D` through the network, which is where the gains at large `D` come from: on CPU 7 times at
`D = 1024` and 37 to 41 times at `D = 4096`; on the A100 3 times and 4 times. It is the same reduction as in the benchmarks of
[performance](performance.md) and **not a general speed-up of `folx`**: `folx`'s sparsity option (off by default,
`sparsity_threshold=0`; it targets local operators and sparse inputs) was not used. **At small `D` `folx` is as fast or faster**:
on the A100 the planner is 1.12 times slower at `D = 16` and 1.54 times slower at `D = 128`, and the drop-in is 1.11 times faster
at `D = 16` and 1.12 times slower at `D = 128`; on CPU both are faster at `D = 16` (1.2 to 1.6 times) and at `D = 128`
(1.8 to 2.0 times). The drop-in's first call (compile included) is where it still trails `folx`: the `folx_compat` rows of the tables above
carry the first-call column of the first measurement, taken before the plan was built in one jitted program. The paired
fresh-process A/B below replaces that column: the same four cells, one fresh process per (round, method), the order alternated,
the intervals the 95 % hierarchical bootstrap over rounds of the ratio (7 CPU rounds, 5 A100 rounds, float64, same tanh MLP).

| device | D | batch | first call before, s | first call now, s | `folx` first call, s | before / now | `folx` / now | steady state now vs `folx` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CPU | 16 | 256 | 3.47 | 1.15 | 0.19 | 3.01x [2.65, 3.02] | 0.164 [0.146, 0.170] | 0.99 |
| CPU | 128 | 256 | 3.35 | 1.21 | 0.19 | 2.77x [2.71, 2.85] | 0.156 [0.146, 0.170] | 0.97 |
| CPU | 1024 | 256 | 4.05 | 1.74 | 0.33 | 2.37x [2.21, 2.38] | 0.187 [0.177, 0.196] | 0.14 |
| CPU | 4096 | 64 | 5.18 | 3.07 | 0.37 | 1.72x [1.66, 1.79] | 0.118 [0.112, 0.130] | 0.03 |
| A100 | 16 | 256 | 6.13 | 2.53 | 1.47 | 2.40x [2.21, 2.62] | 0.568 [0.554, 0.596] | 1.04 |
| A100 | 128 | 256 | 6.34 | 2.84 | 1.75 | 2.25x [2.22, 2.28] | 0.621 [0.584, 0.777] | 0.91 |
| A100 | 1024 | 256 | 8.09 | 3.94 | 1.85 | 2.08x [1.99, 2.61] | 0.475 [0.442, 0.494] | 0.25 |
| A100 | 4096 | 64 | 9.48 | 6.19 | 1.77 | 1.54x [1.49, 1.58] | 0.284 [0.259, 0.288] | 0.20 |

The drop-in's first call is now 1.5 to 3.0 times shorter with the same steady state and the same values, and it is still 1.8 to
8.5 times `folx`'s: the shipped drop-in confirms the matched chain against the model in one jitted program (it used to
dispatch each primitive as its own compile), draws its example on the host, and builds the plan of one point lazily, only when a
function that cannot be planned as a batch needs it. What remains is the planner's own compile of the streamed program; those
first-call cells stay in the full-scope matrix as losses. It is a one-off per shape.

**How to read the adapter table.** The adapters replace the autograd graph of the library's own operator. At `D = 2` there is no
gain: the DeepXDE adapter is 1.3 times slower on CPU and level on the A100 (1.03 times faster), the PINA adapter is 1.8 and 1.7
times slower. From `D = 10` the adapters are faster (CPU 1.1 to 1.3 times; A100 2.2 to 3.0 times) and at `D = 50` much faster
(CPU about 4 times; A100 13 to 18 times). Why the adapters lose at `D = 2` was not profiled. Training-loop wall clock depends on much
more than the Laplacian; a faster operator is not a faster solver.

**Withdrawn.** A first CPU run of the folx table (`jax_cpu_withdrawn_planner_unmatched.json`) was withdrawn: its planner row gave the
planner a closed-over model without declaring that there were no parameters, so the plan was refused as not-a-chain and the
generic executor ran, and the drop-in under `vmap` built one plan per point. Both are bugs in the benchmark or the drop-in and were
fixed (the drop-in now builds one batched plan under `vmap`); that file's slower-than-`folx` numbers describe those bugs, not the
planner.

## Limits

- The adapters and the drop-in are tested against the library releases named above; a newer release is outside the promise until
  it is tested.
- Nothing here claims convergence or accuracy of a trained model; it is the same derivative computed by another program, and the
  float64 agreement above is for benign random weights.
- An engine registered by name is trusted: the planner can verify applicability and time it, not that the engine is correct.
  Run `omnibias.core.conformance.run_correctness` on `conformance_engine(force=name)` before relying on one (see
  [stability](stability.md#conformance)).
