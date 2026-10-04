---
name: omnibias-boolean
description: Maintain exact Boolean representations and their optional differentiable gate realizations.
---

# Maintaining omnibias-boolean

Owned implementation: [source](../../../packages/omnibias-boolean/src/omnibias/boolean);
public surface: [API](../../../docs/api/boolean.md).
Load [shared numerical contracts](../../../docs/development/numerical-contracts.md)
when changing mathematics or tensor behavior.

`_core/` owns exact Boolean objects, transforms and algebra; `torch/` and `jax/`
implement relaxed gates and solver steps. `inequality.py` bridges supported
constraints. Distinguish the truth-table, ANF and Walsh representations, including
index order, variable order and transform normalization. Exact arithmetic over
two elements is not ordinary floating-point polynomial arithmetic.

Boolean derivative and reproductive-equation changes need exhaustive small truth
tables or exact algebraic witnesses. Relaxed solvers need separate checks of
feasibility after decoding and their continuous training objective. Reaching a
small relaxation loss does not certify the decoded assignment.

The binary package supplies optional quantization behavior; import it only through
the declared backend extras. Keep optional tensor dependencies out of exact
algebra imports. `test_anf.py`, `test_walsh.py` and `test_equations.py` cover
representation changes; gate and solver tests cover differentiable behavior.
Verified spectrum or inequality changes additionally need their enclosure tests.
Avoid moving MaxSAT problem front-ends here: that integration belongs to the
discrete/logic family, even when it reuses Boolean algebra.

From the omnibias repository, start with:

```bash
uv run pytest packages/omnibias-boolean/tests/test_anf.py packages/omnibias-boolean/tests/test_walsh.py packages/omnibias-boolean/tests/test_equations.py -q
```
