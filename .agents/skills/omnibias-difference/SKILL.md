---
name: omnibias-difference
description: Maintain finite-difference extraction, exact recurrence fitting and umbral calculus.
---

# Maintaining omnibias-difference

Owned implementation: [source](../../../packages/omnibias-difference/src/omnibias/difference);
public surface: [API](../../../docs/api/difference.md).
Load [shared numerical contracts](../../../docs/development/numerical-contracts.md)
when changing mathematics or tensor behavior.

`_core/`, `umbral.py` and the backend stencil implementations own discrete-step
calculus. `validation/` exercises convergence and certificate behavior;
`singularity.py` consumes series and recurrence information. Keep exact sequence
algebra distinct from floating-point stencil evaluation: the former can prove an
identity, while the latter needs conditioning and truncation analysis.

The public `recurrence` module owns rational constant-coefficient fitting used by
symbolic discovery and holonomic consumers. Preserve `Fraction` arithmetic and
the validation horizon rather than accepting a recurrence merely because it fits
its construction samples. Compatibility entry points in consumers should delegate
here, not fork the fitter.

For stencil work, distinguish repeated nodes, irregular coordinates and changing
step size. Test moment identities and the intended limiting operator before
measuring convergence. Certified extraction must retain the assumptions needed
for its remainder bound. Umbral and special-number routines remain in this
package; q-calculus consumes their exact algebra and defines a separate deformation.
Use irregular-stencil, extraction and parity tests for tensor work, and sequence
identity tests for algebraic changes.

From the omnibias repository, start with:

```bash
uv run pytest packages/omnibias-difference/tests/test_irregular.py packages/omnibias-difference/tests/test_extraction.py packages/omnibias-difference/tests/test_identities.py -q
```
