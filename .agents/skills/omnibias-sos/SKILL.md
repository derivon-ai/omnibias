---
name: omnibias-sos
description: Maintain polynomial positivity proposals, rigorous Gram certificates and formal obligations.
---

# Maintaining omnibias-sos

Owned implementation: [source](../../../packages/omnibias-sos/src/omnibias/sos);
public surface: [API](../../../docs/api/sos.md).
Load [shared numerical contracts](../../../docs/development/numerical-contracts.md)
when changing mathematics or tensor behavior.

`problem.py`, `monomials.py` and `families.py` define polynomial and basis
conventions. `solve.py` proposes numerical Gram matrices; `certify.py`,
`positivstellensatz.py` and `formal.py` own different verification stages. Preserve
monomial ordering and coefficient factors when translating a polynomial into a
symmetric Gram representation.

An SDP solver's status is not a positivity proof. Reconstruct the polynomial
identity and verify the required matrix property with the rigorous checker.
Singular positive-semidefinite cases cannot be accepted by a positive-definite
LDL check without an additional justified reduction. Rounding a candidate must
retain an explicit residual or exact identity.

Inspect certificate assumptions for domains and constraints: a positivity claim
on a constrained set does not establish global positivity. Formal flags depend
on a real checker pass and should remain false when the toolchain is absent.
Use the PSD soundness regression for matrix changes, Positivstellensatz tests for
constraint handling, and formal-honesty tests for checker integration. Consumer
optimization packages request certificates through this interface; they should
not duplicate the numerical-to-rigorous transition.

From the omnibias repository, start with:

```bash
uv run pytest packages/omnibias-sos/tests/test_psd_soundness_regression.py packages/omnibias-sos/tests/test_positivstellensatz.py packages/omnibias-sos/tests/test_honesty_formal.py -q
```
