# Boussinesq remainder CAP (07-16)

CubicGN discovery on Boussinesq jets plus a named interval hull of the
smoke-grid residual. The hull contains a truth sample. The full
streamfunction-Poisson remainder stays external (leftover #54).
`lambda_n` stays an empirical hypothesis. `full_boussinesq_proved`
stays false.

Status is **shipped**. G1–G5 are CI-gated
(`benchmarks/boussinesq_remainder_cap.py`). Not Navier-Stokes. See
theory spec
[07-16](https://github.com/derivon-ai/omnibias/blob/main/theory/07-frontier/16-boussinesq-remainder-cap.md).

Home: `omnibias.pinn.certified.boussinesq` plus
`omnibias.pinn.jax.discovery.boussinesq`.

```python
from omnibias.pinn.certified.boussinesq import (
    BOUSSINESQ_REMAINDER_LEFTOVER,
    enclose_boussinesq_grid_residual,
)
from omnibias.pinn.jax.discovery.boussinesq import (
    BoussinesqDiscoveryConfig,
    run_boussinesq_discovery,
)

out = run_boussinesq_discovery(BoussinesqDiscoveryConfig(n=6, steps=2))
assert out["optimizer"] == "cubic_gn"
assert out["honesty"]["navier_stokes_proof_claim"] is False
assert out["lambda_n_hypothesis"]["status"] == "empirical_hypothesis_not_theorem"
report = enclose_boussinesq_grid_residual(out)
assert report["contains_truth_sample"]
assert report["full_boussinesq_proved"] is False
assert BOUSSINESQ_REMAINDER_LEFTOVER["leftover_id"] == 54
```
