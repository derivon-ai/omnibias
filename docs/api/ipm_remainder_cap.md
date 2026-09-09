# IPM remainder CAP (07-15)

CubicGN discovery on Gaussian-poly IPM jets plus a named interval hull
of the smoke-grid residual. The hull contains a truth sample. The full
streamfunction-Poisson remainder stays external (leftover #53). The
named closing object remains `ipm_banded_toy_radii`.
`full_ipm_proved` stays false.

Status is **shipped**. G1–G5 are CI-gated
(`benchmarks/ipm_remainder_cap.py`). Not Navier-Stokes. See theory spec
[07-15](https://github.com/derivon-ai/omnibias/blob/main/theory/07-frontier/15-ipm-remainder-cap.md).

Home: `omnibias.pinn.certified.ipm` plus `omnibias.pinn.jax.discovery.ipm`.

```python
from omnibias.pinn.certified.ipm import (
    IPM_REMAINDER_LEFTOVER,
    enclose_ipm_grid_residual,
    ipm_banded_toy_radii,
)
from omnibias.pinn.jax.discovery.ipm import IPMDiscoveryConfig, run_ipm_discovery

out = run_ipm_discovery(IPMDiscoveryConfig(n=6, steps=2))
assert out["optimizer"] == "cubic_gn"
assert out["honesty"]["navier_stokes_proof_claim"] is False
report = enclose_ipm_grid_residual(out)
assert report["contains_truth_sample"]
assert report["full_ipm_proved"] is False
assert IPM_REMAINDER_LEFTOVER["leftover_id"] == 53
toy = ipm_banded_toy_radii()
assert toy["full_ipm_proved"] is False
```
