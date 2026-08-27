# Inequality engine

One payload, four sorts, a `ProofMachine` verdict. Propose is a
search; the check is exact or a sound enclosure. This is not founding
bias collapse (`delta -> 0`) and not a complexity-class claim.
Temperature collapse may appear in a linear or CSP proposer.

## A linear box is feasible

```python
from omnibias.core.proof import Conjecture
from omnibias.core.proof.inequality import (
    INEQUALITY_KIND,
    InequalitySystem,
    build_inequality_machine,
)

machine = build_inequality_machine()
system = InequalitySystem(
    sort="linear",
    existential=True,
    data={"A": [["1"], ["-1"]], "b": ["1", "0"]},
    name="unit_box",
)
verdict = machine.evaluate(Conjecture("box", INEQUALITY_KIND, system.as_dict()))
assert verdict.status == "PROVED"
assert verdict.certificate["payload"]["pipeline"] == [
    "propose",
    "rationalize",
    "check",
]
```

## A forged complexity claim is blocked

```python
from omnibias.core.proof import Conjecture
from omnibias.core.proof.inequality import INEQUALITY_KIND, build_inequality_machine

machine = build_inequality_machine()
verdict = machine.evaluate(
    Conjecture(
        "forged",
        INEQUALITY_KIND,
        {
            "sort": "linear",
            "existential": True,
            "data": {"A": [["1"], ["-1"]], "b": ["1", "0"]},
        },
        claims={"p_equals_np_claim": True},
    )
)
assert verdict.status == "BLOCKED"
assert verdict.honesty_ok is False
```
