# omnibias-binary

Differentiable quantization primitives.

- `omnibias.binary.torch.ops` and `omnibias.binary.jax.ops`: hard quantizers
  with smooth surrogate derivatives.
- `omnibias.binary.BetaAnnealScheduler`: temperature scheduling.

The surrogate backward path is a modeling choice for optimization; it is not
the classical derivative of a discontinuous hard quantizer. Temperature changes
the smoothing and can change optimization conditioning. This primitive is
also consumed by the differentiable Boolean backend.

Install this distribution with `pip install omnibias-binary`; select its
backend extras when needed. See [guarantees](../guarantees.md).

<!-- BEGIN GENERATED API INVENTORY -->

Version **0.1.0a2** · Python **>=3.10** · **3 - Alpha** · Apache-2.0

<details markdown="1">
<summary>Public modules and top-level exports</summary>

[Source](https://github.com/derivon-ai/omnibias/tree/main/packages/omnibias-binary/src/omnibias/binary). Modules below are relative to `omnibias.binary`; underscored modules are internal.

`jax`, `jax.ops`, `jax.ops.quantize`, `jax.ops.surrogate`, `schedule`, `torch`, `torch.ops`, `torch.ops.quantize`, `torch.ops.surrogate`.

Exports from `omnibias.binary`:

`BetaAnnealScheduler`.

</details>

<!-- END GENERATED API INVENTORY -->
