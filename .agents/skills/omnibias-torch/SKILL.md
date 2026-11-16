---
name: omnibias-torch
description: Maintain PyTorch tensor kernels, trainable modules and optimizer integration in omnibias-torch.
---

# Maintaining omnibias-torch

Owned implementation: [source](../../../packages/omnibias-torch/src/omnibias/torch);
public surface: [API](../../../docs/api/torch.md).
Load [shared numerical contracts](../../../docs/development/numerical-contracts.md)
when changing mathematics or tensor behavior.

`fastpath/` and `activations/` adapt activation formulas; `jet.py`, `jet_mv.py`
and `laplacian.py` own spatial operators. Read the corresponding JAX implementation
when changing a paired algorithm, but keep PyTorch module and optimizer behavior
here. The shared numerical reference owns cross-backend invariants.

For `nn.Module` changes, inspect registration of parameters and buffers, device
moves, state dictionaries and framework-default dtype behavior. For optimizer
changes, distinguish functional parameter-vector drivers from optimizer subclasses
whose closures recompute a loss or residual. `JetLBFGSOptimizer` and `JetLBFGS`
are different interfaces. Reusing an old autograd graph across closure calls can
break otherwise correct formulas.

`operators.py` is the stable front end of the planner (`compile_operator` is `experimental.planned.compile_planned_operator`);
`experimental.compile_operator` is a different function that compiles an `OperatorSpec` and plans only for a string, so do
not alias one to the other. Promoted names stay reachable from `experimental` through its `__getattr__`; add a name to
`PROMOTED` in `operators.py` instead of importing it eagerly. Planner changes need the JAX twin's explain keys and refusal
reasons to stay the same (`tests/test_cross_backend_fixtures.py`, `tests/test_conformance.py`).
`deepxde_adapter.py` and `pina_adapter.py` are tested against one release each (named in [interop](../../../docs/interop.md));
they import their libraries lazily and `tests/test_interop_adapters.py` skips when the library is absent.

Low-order programs and the training objective. `experimental/streams.py` and `fused_streams.py` mirror the JAX stream engine
and the one-kernel programs (Triton, wrapped in `torch.library` custom ops so `torch.compile` and CUDA graphs accept them;
run the fused tests with `TRITON_INTERPRET=1` on a CPU). `training_step.py` captures a CUDA-graph training step and
`optim_pinn.py` holds the second-order steps; both are new and outside the stable surface. `projected_polylaplacian.py`
ports `Delta^2` and `Delta^3` with dense Hessian storage; the projected residual streams are not ported and
`compile_residual` refuses `projected_streams` with that reason. Check a ranking change against the JAX twin with
`tests/test_planned_v2.py`, and do not run Torch planner probes on a login node (a planner `find` compiles).
A CUDA-graph capture that fails part way leaves the default generator flagged as capturing and, if it ran out of memory,
its private pool allocated; both break every later call of the process. `training_step.py` resets the graph and renews the
generator state on that path (`test_a_failed_capture_falls_back_to_a_working_eager_step_and_leaves_the_device_usable`), and a
benchmark that captures many methods in one process should skip a capture that the free memory cannot hold. Torch must be run
with CUDA to see either failure.

Hot-path rules (audit of 2026-10-26). A cache keyed by a tensor holds a weak reference and validates `_version` and `data_ptr` on a
hit (`fused_streams._PADDED`; the fields operator cache keys the identity of leaf tensors); it never holds a tracer and is bypassed under
`torch.compiler.is_compiling()` and CUDA-graph capture (`capture_safe_branches()`); a zero-copy numpy view of a JAX array is copied
before it is kept. `compile_planned_operator(graph=...)` replays a no-gradient CUDA call as a CUDA graph and is off by default
(`"auto"` falls back and `True` raises). `deep_field_polylaplacian` routes `k` in `{2, 3}` to the projected program from
`STRUCTURED_MIN_DIM` (`STRUCTURED_MIN_DIM_CUDA` on an accelerator) and takes `mode="structured"`; its result stays connected to the readout
bias, so a gradient into a parameter the operator does not depend on is an exact zero, never "unused" (a routing change must not make
`torch.autograd.grad` without `allow_unused` raise: `tests/test_deep_laplacian.py` pins it, and the audit harness's `none_mask.py`
compares which leaves are reached against the pre-audit tree). Past the support budget `auto` still samples. The per-template launch
table of the planner was fitted to XLA lowering; eager PyTorch launches differ and applying it worsened replayed decisions, so the torch
cost model keeps its generic launch cost. A `find` timer must time the callable the operator will run (precision policy, compile
mode, `torch.cuda.empty_cache()` between candidates on CUDA).

Spatial derivatives must remain connected to trainable weights. Test the final
residual loss backward pass, not just a derivative tensor's numeric value.
`test_deep_laplacian.py` covers wide coordinate dimensions; `test_jet_mv.py` covers
mixed coefficients. Module and closure changes also need `test_nn_optim.py` or
`test_unit.py` as appropriate. Benchmark in the execution mode the documented
user actually runs, with thread policy recorded.

From the omnibias repository, start with:

```bash
uv run pytest packages/omnibias-torch/tests/test_jet.py packages/omnibias-torch/tests/test_deep_laplacian.py packages/omnibias-torch/tests/test_nn_optim.py -q
```
