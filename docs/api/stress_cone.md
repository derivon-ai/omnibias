# Admissible stress cone (07-10)

`T` lies in the interior of `cone(v1, v2)` iff `det(v1, v2) ≠ 0` and
both Cramer weights are strictly positive. Parallel generators and the
opposite cone are `BLOCKED` and named. No LP solver.

The companion scale ledger (`h < 1/100`, `h < 1/6`, `κ_s ≤ 10^{-5}`)
is `navier_stokes_scale_ledger()` on the 07-08 spine.

Status is **shipped**. G1–G5 are CI-gated
(`benchmarks/stress_cone.py`). Parent flags stay false. See theory spec
[07-10](https://github.com/derivon-ai/omnibias/blob/main/theory/07-frontier/10-admissible-stress-cone.md).

Home: `omnibias.core.proof.obligations.stress_cone`.

```python
from fractions import Fraction

from omnibias.core.proof.obligations.convergence_ledger import (
    check_ledger,
    navier_stokes_scale_ledger,
)
from omnibias.core.proof.obligations.stress_cone import (
    check_cone,
    locked_interior_cone,
    parallel_generators_cone,
    seal_cone_certificate,
)

report = check_cone(locked_interior_cone())
assert report.holds
assert report.lambda1 == Fraction(1)
assert report.lambda2 == Fraction(1)

blocked = check_cone(parallel_generators_cone())
assert blocked.holds is False
assert blocked.reason == "degenerate_generators"

sealed = seal_cone_certificate(locked_interior_cone(), run_lean=False)
assert sealed.mathlib_verified is False
assert sealed.certificate["honesty"]["navier_stokes_proof_claim"] is False

scale = check_ledger(navier_stokes_scale_ledger())
assert scale.holds
assert scale.binding_threshold is not None
assert scale.binding_threshold[0] == "h"
assert scale.binding_threshold[2] == Fraction(1, 100)
```
