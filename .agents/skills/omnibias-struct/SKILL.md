---
name: omnibias-struct
description: Maintain structured dynamic programs, semirings, differentiated values and certified decoding.
---

# Maintaining omnibias-struct

Owned implementation: [source](../../../packages/omnibias-struct/src/omnibias/struct);
public surface: [API](../../../docs/api/struct.md).
Load [shared numerical contracts](../../../docs/development/numerical-contracts.md)
when changing mathematics or tensor behavior.

`_core/` owns structural definitions and semiring operations; backend directories
implement tensor programs. `select.py`, `decode.py`, `verified.py` and
`decision/` provide distinct selection, decoding, enclosure and regional bridges.
Check state ordering, impossible transitions and empty structures before changing
a recurrence shared by multiple algorithms.

Soft and hard dynamic programs must agree in the documented limit without losing
tie conventions or path multiplicities. A log-sum-exp value, partition function,
normalized marginal and best-path score are different outputs. Derivative jets
apply to the specified scalar value and normalization, not automatically to a
decoded structure.

Use exhaustive tiny chains, graphs or grammars as oracles for recurrence changes.
Gradient tests must include all trainable transition and emission parameters.
For certified decoding, retain the link between the feasible structure, its
objective and the bound; interval verification is separate from floating-point
path selection. Semiring, family-specific and decode tests catch different
classes of errors. Soft-tree training and tabular boosting remain consumer
responsibilities, even when they reuse the decision bridge.

From the omnibias repository, start with:

```bash
uv run pytest packages/omnibias-struct/tests/test_semiring.py packages/omnibias-struct/tests/test_gradients.py packages/omnibias-struct/tests/test_decode.py -q
```
