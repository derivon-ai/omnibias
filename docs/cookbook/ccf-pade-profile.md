# CCF jet-Padé profile diagnostic

Padé poles of a fixed profile jet are a **diagnostic**. They are not a
residual substitute and they do not clear
`CCF_STRETCH_RESIDUAL_GATE` (`1e-13`). The module never consumes a
zeta / L-function series.

```python
from omnibias.difference._core.ccf_profile import (
    diagnose_ccf_profile,
    geometric_profile_jet,
)

coeffs = geometric_profile_jet(radius=2.0, order=12)
report = diagnose_ccf_profile(coeffs, snapshot="geometric_radius_2")
assert report.honesty["navier_stokes_proof_claim"] is False
assert report.honesty["residual_substitute"] is False
assert report.honesty["dirichlet_consumed"] is False
assert report.estimate.failed is False
assert report.estimate.location is not None
assert abs(report.estimate.location.real - 2.0) < 0.25
```

Optional MultiPack (01-01) and irregular-stencil (01-04) arms are named
flags on `benchmarks/ccf_pade_profile.py`, not default CI. A better
floor versus `hardy_corrected_pv` is reported as a floor, not stretch.

## See also

- Smoke: [`ccf_pade_profile_smoke.json`](../benchmarks/ccf_pade_profile_smoke.json)
- [CCF self-similar singularities](ccf-singularity.md)
