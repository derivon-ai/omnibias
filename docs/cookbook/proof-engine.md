# Proof engine cookbook

The engine routes a named finite kind through a shipped collapse or a
catalog family, seals a v1 certificate, and returns a reason tree.
Natural language is never a premise. A float residual is not a proof.
`BLOCKED` is not falsity. Continuum parents are not inferred.
`theorem_prover_verified` is earned only by a genuine `lake build`.
Pairing is not a strong solution. Rank / holonomic syzygy is not a
special-function theorem.

## Residual `{0}` proves; a float is refused

```python
from omnibias.core.collapse import reset_collapse_registry
from omnibias.core.proof import prove

reset_collapse_registry()
proved = prove("residual", {"lo": 0.0, "hi": 0.0})
assert proved.proved
assert "A float residual is not a proof." in proved.explain()
refused = prove("residual", {"value": 0.0})
assert refused.blocked
```

## External parents stay out of scope

```python
from omnibias.core.proof import prove

blocked = prove("external", {"parent": "rh"})
assert blocked.blocked
assert "not in scope" in blocked.verdict.detail
assert blocked.reason[0].honesty["continuum_parent_inferred"] is False
```

## Identity and an OPT sandwich

```python
from omnibias.core import collapse as collapse_package
from omnibias.core.proof import prove

assert not hasattr(collapse_package, "gap_collapse")
identity = prove(
    "identity",
    {"left": [1, 2, 1], "right": [1, 2, 1], "domain": [-2.0, 2.0]},
)
assert identity.proved
tight = prove("gap", {"L": 3.0, "U": 3.0})
assert tight.proved
assert tight.reason[0].honesty["enclosure_collapse"] is True
```

Jacobian `n=2` Case A leftover identities (`(b11,b21,b31)` and the
`(b02,b03,b04)` axis) are the same `identity` kind. The parent stays
open; see [Jacobian n=2 finite box](jacobian-n2-box.md).
