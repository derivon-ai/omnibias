# Jacobian n=2 finite box

The Jacobian conjecture in dimension 2 is still open. Moh showed there
is no counterexample of degree at most 100. omnibias searches a
**finite universal** `C_box(d, h, G)`: every integer-coefficient map
`Q^2 -> Q^2` of total degree `<= d` and coefficient height `<= h`
either has `det JF` not identically a nonzero constant, or has no two
distinct points of a finite rational grid `G` sharing an image.

A grid collision plus an **identical** (not probed) nonzero constant
Jacobian would be a genuine `n=2` counterexample. A miss proves only
`C_box`. It is not injectivity on all of `Q^2`, and it is not the
parent. Probe samples of `det JF` can agree on an axis while the
polynomial is non-constant (`1+y`); the gate uses
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

`PROVED` on the machine is the finite obligation. Catalog kind
`jacobian_n2_degree_box`. Degree `>= 2` walks a structured incomplete
slice; a miss there is `search_incomplete`.
