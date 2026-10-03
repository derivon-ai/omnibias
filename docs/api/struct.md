# omnibias-struct

Differentiable structured computation.

`omnibias.struct` provides soft dynamic programming on chains, acyclic graphs
and shared semiring/hypergraph representations.

- `ChainTrellis`, `DAG`, `Hypergraph`: structural inputs.
- `MaxPlusSemiring`, `LogSemiring`, `CountingSemiring`: evaluation algebras.
- `omnibias.struct.torch` and `.jax`: differentiable implementations.
- `DPGapCertificate`, `certify_soft_dp`: soft-versus-hard value bounds.

Log-sum-exp temperature controls smoothing. The bound depends on the counted
finite alternatives and the inverse temperature; it does not assert that an
arbitrary learned decoder is correct. Consult the operation's signature for
tensor layout and its associated small-instance oracle.

Install this distribution with `pip install omnibias-struct`; select its
backend extras when needed. See [guarantees](../guarantees.md).
