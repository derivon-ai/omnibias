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
configuration from a library import.

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
