# Jacobian n=2 finite box

The Jacobian conjecture in dimension 2 is still open. Moh showed there
is no counterexample of degree at most 100. omnibias searches a
**finite universal** `C_box(d, h, G)`: every integer-coefficient map
`Q^2 -> Q^2` of total degree `<= d` and coefficient height `<= h`
either has `det JF` not identically a nonzero constant, or has no two
distinct points of a finite rational grid `G` sharing an image.

A grid collision plus an **identical** (not probed) nonzero constant
Jacobian would be a genuine `n=2` counterexample. So would a Keller map
that fails Gabber's inverse-degree test (`deg(F^{-1}) <= deg F` for
`n=2`). A miss proves only `C_box`. It is not injectivity on all of
`Q^2`, and it is not the parent. Probe samples of `det JF` can agree on
an axis while the polynomial is non-constant (`1+y`); the gate uses
`identical_jacobian_constant`.

```python
from omnibias.core.proof import run_discovery
from omnibias.holonomic import JacobianN2DegreeFamily, escalate_n2_result

family = JacobianN2DegreeFamily(max_degree=0, coeff_height=1)
assert family.complete is True
assert family.statement.existential is False
assert family.statement.parent_status == "open"
result = run_discovery(family.statement, family, "score_guided", budget=16)
assert result.status == "PROVED"
assert result.detail == "no counterexample in complete family"
sealed = escalate_n2_result(result)
assert sealed["escalate_parent"] is False
assert sealed["honesty"]["jacobian_n2_claim"] is False
assert sealed["honesty"]["jacobian_conjecture_proof_claim"] is False
```

A shear automorphism is in the degree-1 box and is not a violator:

```python
from omnibias.holonomic.jacobian_n2 import JacobianN2DegreeFamily

degree_one = JacobianN2DegreeFamily(max_degree=1, coeff_height=1)
identity = degree_one.check((0, 1, 0, 0, 0, 1))
assert identity is not None
assert identity.ok is False
assert identity.payload["jacobian_identity"] == "identical"
assert identity.payload["jacobian_constant"] == "1"
assert identity.payload["honesty"]["jacobian_n2_claim"] is False
```

A shear automorphism passes Gabber and is not a violator:

```python
from omnibias.holonomic import gabber_n2_test
from omnibias.holonomic.jacobian_n2 import shear_map

shear = shear_map(0, (0, 0, 1))
result = gabber_n2_test(shear)
assert result.keller is True
assert result.inverse_ok is True
assert result.fails is False
```

`PROVED` on the machine is the finite obligation. Catalog kinds
`jacobian_n2_degree_box` and `jacobian_n2_homogeneous`. Degree `>= 2`
walks a structured incomplete slice; a miss there is `search_incomplete`.

## O1 leftover charts (not a parent proof)

O1 scales each component so the leading homogeneous part is monic.
Case C leftover is `0`. Case D leftover is a nonzero constant (the
leading content removed by O1). Both are finite rationals. Soft CSP
misses stay refutations, not proofs.
`JACOBIAN_CONJECTURE_PROOF_CLAIM_ALLOWED` stays false until O1, O2,
and A/B/C/D are all sealed. O2 is not sealed.

```python
from omnibias.holonomic import (
    JACOBIAN_CONJECTURE_PROOF_CLAIM_ALLOWED,
    classify_leftover_chart,
    o1_normalize,
)
from omnibias.holonomic.jacobian_n2_normalize import (
    named_case_c_shear,
    named_case_d_content,
)

assert JACOBIAN_CONJECTURE_PROOF_CLAIM_ALLOWED is False
monic = named_case_c_shear()
scaled = named_case_d_content()
norm = o1_normalize(scaled)
assert norm[0] == monic[0]
assert norm[1].homogeneous_part(2) == monic[1].homogeneous_part(2)
case_c = classify_leftover_chart(monic)
case_d = classify_leftover_chart(scaled)
assert case_c.case == "C" and case_c.leftover == 0
assert case_d.case == "D" and case_d.leftover != 0
assert case_c.honesty()["jacobian_conjecture_proof_claim"] is False
assert case_d.honesty()["jacobian_conjecture_proof_claim"] is False
```

## Case A `(b11, b21, b31)` leftover (not the parent)

After `b20=b30=b40=0` on the I+Q+C+Qu leftover, the three generators in
`Q[b11, b21, b31]` vanish over `Q` only at the origin. That is a local
chart seal. The remaining Case A leftover in `(b02, b03, b04)` is a
later slice. A complete miss is `BLOCKED`, not parent-false.

```python
from omnibias.core.collapse import reset_collapse_registry
from omnibias.core.proof import prove
from omnibias.holonomic import (
    JACOBIAN_CONJECTURE_PROOF_CLAIM_ALLOWED,
    case_a_b31_statement,
    seal_case_a_b31,
)
from omnibias.holonomic.jacobian_n2_case_a import case_a_b31_identity_payloads

reset_collapse_registry()
assert JACOBIAN_CONJECTURE_PROOF_CLAIM_ALLOWED is False
statement = case_a_b31_statement()
assert statement.parent_status == "open"
assert statement.existential is False
for payload in case_a_b31_identity_payloads():
    assert prove("identity", payload).proved
sealed = seal_case_a_b31()
assert sealed.proved
assert sealed.certificate["honesty"]["jacobian_conjecture_proof_claim"] is False
assert sealed.certificate["honesty"]["continuum_parent_inferred"] is False
assert sealed.certificate["payload"]["parent_status"] == "open"
refused = prove("residual", {"value": 0.0})
assert refused.blocked
```
