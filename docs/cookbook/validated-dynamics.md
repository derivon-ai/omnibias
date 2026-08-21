# Validated jet Lohner steps

A tanh scalar field has a closed-form Jacobian. Every
validated run names the dominant width term before any
improvement claim. This is one trajectory, not an
attractor.

```python
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.jet_flow import (
    DISCLAIMER,
    TowerFieldSpec,
    honesty_payload,
    lohner_flow_jet,
    tower_field,
    tower_jacobian,
)

spec = TowerFieldSpec("tanh")
run = lohner_flow_jet(
    tower_field(spec),
    tower_jacobian(spec),
    [Interval(0.2, 0.3)],
    0.05,
    8,
    order=8,
)
assert run.budget.dominant in {
    "truncation",
    "jacobian",
    "wrapping",
    "rounding",
}
assert run.horizon == 0.4
assert honesty_payload()["continuum_existence_claim"] is False
assert honesty_payload()["theorem_prover_verified"] is False
assert "not a continuum existence theorem" in DISCLAIMER
```

A step suggested from the solution jet must stay inside a
known radius of convergence. The a-priori enclosure still
has to accept it.

```python
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.jet_flow import (
    adaptive_step_from_singularity,
    riccati_field,
)
from omnibias.core.verified.lohner import LohnerSet

state = LohnerSet.from_box([Interval.point(1.0)])
step = adaptive_step_from_singularity(
    riccati_field(), state, order=16, safety=0.5
)
assert 0.0 < step <= 1.0
```
