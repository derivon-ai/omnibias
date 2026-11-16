---
name: omnibias-jax
description: Maintain JAX tracing, functional derivative operators and compiled kernels in omnibias-jax.
---

# Maintaining omnibias-jax

Owned implementation: [source](../../../packages/omnibias-jax/src/omnibias/jax);
public surface: [API](../../../docs/api/jax.md).
Load [shared numerical contracts](../../../docs/development/numerical-contracts.md)
when changing mathematics or tensor behavior.

`_fastpath.py` and `activations.py` supply activation kernels; `jet.py`,
`jet_mv.py` and `laplacian.py` own the differentiated field operators. Preserve
functional parameter flow and pytree structure. A value that changes between
calls must remain a runtime argument, including network weights in benchmarks.

When tracing fails, distinguish shape or order metadata from tensor values before
introducing static arguments. Avoid converting a tracer to a Python scalar or
NumPy array. Inspect `jit`, `vmap` and parameter `grad` together: a function that
works eagerly can still fail under batching or compilation. Enable x64 before
constructing arrays for float64 parity checks, rather than changing global JAX
configuration from a library import. Do not keep an array made inside a trace on an object that a later trace, or a
`custom_jvp` rule, captures: hold numpy constants and turn them into arrays where they are used (the block layout of
`experimental/projected_bilaplacian.py` failed to lower under `jit(grad)` once `min(D, H) >= 32` until it did).
`tests/test_experimental_planner_v2.py` differentiates the planned bi-Laplacian at that width.

`operators.py` is the stable front end of the planner (`compile_operator`, `compile_residual`, the policies, the
distributed functions, `conformance_engine`); `experimental/compiler.py` and the programs behind it stay experimental.
Promoted names remain aliases through the module `__getattr__` of `experimental/__init__.py`: add a name to `PROMOTED`
in `operators.py` rather than importing it eagerly there. Run `tests/test_conformance.py`,
`tests/test_plan_format.py` and `tests/test_deprecation_schedule.py` when touching the facade.
`folx_compat.py` mirrors `folx.forward_laplacian` field by field; `tests/test_folx_compat.py` compares it with the installed
folx. The planner path needs concrete closed-over weights (a model whose weights are tracers falls back to the generic path)
and a failed plan must not be cached. See [interop](../../../docs/interop.md).

Low-order programs and the training objective. `experimental/streams.py` is the stacked XLA stream engine (one `dot_general`
and one elementwise update per layer; `Delta` through order-four directional jets); `fused_streams.py` is the Pallas kernel
pair, forced-only on JAX and in float64 and bounded by the shared-memory check in `plan_algorithms.py`; `_pallas_compat.py`
isolates the Pallas API differences (run the fused tests with `OMNIBIAS_PALLAS_INTERPRET=1` on a CPU). `training_step.py`
(`compile_training_step`, `planned_evaluator`) and `optim_pinn.py` are new and outside the stable surface. A ranking change
needs `tests/test_experimental_planner_v2.py` and a regenerated coverage matrix: a work model that undercounts a program
(the polylaplacian once did) puts a program that cannot be traced at the top of the ranking. `compile_policy="auto"`
retries XLA autotune and Triton-GEMM failures (`coldcost.RETRY_LADDER`), `safe_find` times candidates in a child process
(`_safe_find.py`), and an installed `folx` is a planner candidate unless `OMNIBIAS_FOLX_ENGINE=off`. `compile_operator` times every candidate under `memory_budget=inf`, and XLA can crash the process compiling one it sizes at terabytes
(seen at order four, `D = 1024`, `B = 2048`): benchmark and CI code that times candidates passes a finite budget or `safe_find=True`. Heavy measurements run
on a snapshot (`omnibias_experiments/percall_*/snapshot.py`), never on the live tree.

Audit rules (2026-10-26). `deep_field_polylaplacian` routes `k` in `{2, 3}` to the projected program from `STRUCTURED_MIN_DIM`
(`STRUCTURED_MIN_DIM_CUDA` on an accelerator) and past the support budget wherever it applies; `mode="structured"` selects it. The streams
take `remat=` (a `jax.checkpoint` per layer, off by default) and `identity_block=` (the Gram form's first-layer identity channels as
`B k H` products: faster forward on medium and large cells, up to 4.6 % more temporary memory in a gradient, so off by default); a new
option defaults to the current behavior. Distributed Gram and Gauss-Newton programs are built once per signature, not per call, and the
sparse polarization plans of `omnibias.core.polarization` are contracted by gather and segment-sum (`_contract_plan`). A refactor claimed
to be identical must be bit-identical on the float64 parity cells of the A/B harness: a bias folded into the activation input rather than
added to the stack changed the last bit under XLA (`jax.jets.o3.s`) and was reverted; XLA temporary bytes plus the constants baked into the
module are the memory gate. `jit` programs that capture a table as a constant count it as memory.
The support enumeration (`mode="support"`) is unrolled, one traced jet per support, up to `SUPPORT_UNROLL_MAX_COEFFS` total jet
coefficients (the program of every earlier release, bit for bit), and a vmapped, `lax.map`-chunked program per support size above it: the
batched program compiles 8 to 12 times faster but its XLA temporaries are 1.7 to 1.8 times larger at the sizes where the unrolled one still
compiles, so it is not the default there. PyTorch has no such crossover (the batched program is faster and its training peak lower at every
size measured), and `deep_field_polylaplacian` there keeps the readout bias connected to the result (an exact zero) in every mode, because
`torch.autograd.grad` without `allow_unused` raises for a parameter the program does not read.
A device-dependent route needs a measurement on that device: a traced `hess_form` takes the `m` coordinate tangents with the metric in-graph up to `m <= 2H` on a CPU and only `m <= H/4` on an accelerator (`HESS_FORM_COORDINATES_MAX_RATIO[_ACCELERATOR]`; at `m / H = 0.67` the forward pass was 1.36 times slower on an A100 while the gradient was faster), and a rewrite that is identical on a CPU can still add XLA temporaries on a GPU (`projected_gram.omega` by direct column assembly: +0.7 %, reverted). Gate a JAX GPU change with A, `A2` and the candidate in one job: two jobs land on different cards, and the autotuned product algorithm moves the last bit between processes of the same code.

Compilation and steady execution are separate measurements; synchronize returned
arrays before stopping a timer. Check the compiled computation when a speedup
looks implausibly flat, since closed-over constants can erase the workload.
Operator changes should exercise both direct evaluation and parameter gradients;
Born–Oppenheimer derivatives have additional invariance tests in
`test_bo_derivatives.py`. Consult the Torch twin for algorithm changes, not for
framework-specific control flow.

From the omnibias repository, start with:

```bash
uv run pytest packages/omnibias-jax/tests/test_jet.py packages/omnibias-jax/tests/test_deep_laplacian.py packages/omnibias-jax/tests/test_bo_derivatives.py -q
```
