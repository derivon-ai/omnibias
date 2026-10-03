# omnibias-core

Shared algebra and verified numerics.

Pure Python; independent of tensor frameworks.

- `omnibias.core.polynomials`: shared activation derivative coefficients.
- `omnibias.core.bell`: composition combinatorics.
- `omnibias.core.multi_index`: ordering and products for mixed Taylor jets.
- `omnibias.core.spec.ActivationSpec`: activation metadata and derivative providers.
- `omnibias.core.verified`: interval and Taylor-model enclosures.
- `omnibias.core.proof.certificate`: canonical certificates and digest checks.

Tensor implementations consume these coefficients. Extend a recurrence here
once, then update and test each backend. An integrity digest is separate from
a successful mathematical checker run.

Install this distribution with `pip install omnibias-core`; select its
backend extras when needed. See [guarantees](../guarantees.md).
