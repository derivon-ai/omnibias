---
name: omnibias-core
description: Maintain pure-Python activation coefficients, Taylor combinatorics, verified numerics and certificate machinery in omnibias-core.
---

# Maintaining omnibias-core

Owned implementation: [source](../../../packages/omnibias-core/src/omnibias/core);
public surface: [API](../../../docs/api/core.md).
Load [shared numerical contracts](../../../docs/development/numerical-contracts.md)
when changing mathematics or tensor behavior.

Start with the smallest mathematical definition that owns the change: `polynomials.py`
for activation coefficients, `bell.py` for composition, `multi_index.py` for mixed
indices, `spec.py` for activation metadata. Consumers and tensor backends must
reuse those definitions. Keep exact integer or rational calculations exact until
the documented numerical boundary; do not replace a recurrence with fitted data.

`verified/` owns enclosure arithmetic and its assumptions. `proof/` serializes
and checks finite obligations; the Lean kernel is a separate executable checker.
Trace a proposed verification change through all three layers before altering a
verdict. Preserve refusal behavior for invalid domains, inconclusive bounds and
missing checkers. Core's dependency tests must continue to work without installed
Torch, JAX or Keras.

For coefficient work, exercise algebraic identities and independent values across
orders, including zero and negative input orders. For enclosure work, include
endpoint, degeneracy and cancellation cases that can expose unsound rounding.
Use the existing certificate audit and package-boundary tests when touching
serialization, imports or public exports.

From the omnibias repository, start with:

```bash
uv run pytest packages/omnibias-core/tests/test_polynomials_invariants.py packages/omnibias-core/tests/test_bell.py packages/omnibias-core/tests/test_multi_index.py -q
```
