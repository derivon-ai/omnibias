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
