# Force-free BKM slabs (07-18, 07-19, 07-21)

One periodic box, one horizon, `f = 0`. Plants are the exact 2-D
Taylor–Green vortex and the exact 3-D ABC flow. The BKM time integral
of a sound `||ω||_∞` bound is an outward-rounded `Interval`. Weak
residuals on TG reuse 07-02. Continuation accepts a decaying slab
strictly below a named rational budget, or returns `Halt`.

Status is **shipped**. G1–G5 are CI-gated
(`benchmarks/unforced_bkm_slab.py`,
`benchmarks/unforced_slab_continuation.py`,
`benchmarks/unforced_abc_slab.py`). This is A/B
*architecture*, not Clay (A)/(B). See theory specs
[07-18](https://github.com/derivon-ai/omnibias/blob/main/theory/07-frontier/18-unforced-bkm-slab.md),
[07-19](https://github.com/derivon-ai/omnibias/blob/main/theory/07-frontier/19-unforced-slab-continuation.md),
and
[07-21](https://github.com/derivon-ai/omnibias/blob/main/theory/07-frontier/21-unforced-abc-slab.md).

Home: `omnibias.pinn.certified.unforced`. Do not grow
`omnibias.pinn.certified.navier_stokes`.

```python
from omnibias.core.proof.obligations.convergence_ledger import (
    NS_AB_EXTERNAL_PREMISES,
    check_ledger,
    navier_stokes_ab_architecture_ledger,
)
from omnibias.pinn.certified.unforced import (
    Continue,
    Halt,
    UNFORCED_CONTINUATION_LEFTOVER,
    force_free_bkm_slab,
    honesty_payload,
    locked_two_slab_continuation,
    try_continue_slab,
    locked_force_free_slab,
    LOCKED_HORIZON,
)

report = force_free_bkm_slab()
assert report["force_zero"] is True
assert report["weak_covers"] is True
assert report["bkm_contains_exact"] is True
assert report["integrand_grid_and_sample"] is True
flags = honesty_payload()
assert flags["navier_stokes_proof_claim"] is False
assert flags["three_d_claim"] is False
assert flags["infinite_time_leftover"] is True
assert flags["bridge_theorem_leftover"] is True
ledger = navier_stokes_ab_architecture_ledger()
assert check_ledger(ledger).strength == "CONDITIONAL"
assert NS_AB_EXTERNAL_PREMISES

first = locked_force_free_slab()
accepted = try_continue_slab(
    first, remaining_budget=1, next_horizon=LOCKED_HORIZON
)
assert isinstance(accepted, Continue)
growing = try_continue_slab(
    locked_force_free_slab(growing=True),
    remaining_budget=1,
    next_horizon=LOCKED_HORIZON,
)
assert isinstance(growing, Halt)
assert growing.reason == "BLOCKED"
chain = locked_two_slab_continuation()
assert chain["n_slabs"] == 2
assert chain["leftover_id"] == 57
assert UNFORCED_CONTINUATION_LEFTOVER["leftover_id"] == 57
assert chain["covers_infinite_time"] is False
```

## 3-D ABC plant (07-21)

The same BKM / continuation budget on the exact decaying 3-D ABC
flow. The fixture is force-free and Beltrami. A sound Euclidean hull
of the component bounds encloses `‖u_0‖_∞`. `three_d_claim` stays
false: one named plant is not the Clay 3-D quantifier. Leftover **#57**
is reused. The A/B premise `"three-dimensional unforced NS, not 2-D
Taylor-Green"` stays.

```python
from omnibias.pinn.certified.unforced import (
    THREE_D_AB_PREMISE,
    abc_honesty_payload,
    force_free_abc_bkm_slab,
    locked_two_abc_slab_continuation,
    try_continue_abc_slab,
    locked_force_free_abc_slab,
    Continue,
    Halt,
    LOCKED_HORIZON,
)

abc = force_free_abc_bkm_slab()
assert abc["dimension"] == 3
assert abc["force_zero"] is True
assert abc["bkm_contains_exact"] is True
assert abc["integrand_grid_and_sample"] is True
assert abc_honesty_payload()["three_d_claim"] is False
assert abc_honesty_payload()["exact_3d_abc_plant"] is True
assert THREE_D_AB_PREMISE in NS_AB_EXTERNAL_PREMISES
accepted = try_continue_abc_slab(
    locked_force_free_abc_slab(),
    remaining_budget=1,
    next_horizon=LOCKED_HORIZON,
)
assert isinstance(accepted, Continue)
growing = try_continue_abc_slab(
    locked_force_free_abc_slab(growing=True),
    remaining_budget=1,
    next_horizon=LOCKED_HORIZON,
)
assert isinstance(growing, Halt)
abc_chain = locked_two_abc_slab_continuation()
assert abc_chain["n_slabs"] == 2
assert abc_chain["leftover_id"] == 57
```
