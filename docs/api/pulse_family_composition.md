# Pulse-family composition (07-17)

A locked occupancy pulse `P = s t` from 07-12 composes into the 07-13
forced field. `P'` matches the sigmoid tower on a finite rational grid.
The corrected axis jet times `P` is `{0}`. `P` times the uncorrected
jet is enclosed (grid + random sample). That is one named pulse family,
not cutoff summation and not joining.

Status is **shipped**. G1–G5 are CI-gated
(`benchmarks/pulse_family_composition.py`). This is a fragment of a
constructed forced blowup. It does not re-prove Clay (C)/(D) and does
not touch unforced (A)/(B). See theory spec
[07-17](https://github.com/derivon-ai/omnibias/blob/main/theory/07-frontier/17-pulse-family-composition.md).

Home: `omnibias.core.pulse_envelope` plus
`omnibias.pinn.certified.forced_flat`.

```python
from omnibias.core.proof.obligations.convergence_ledger import (
    NS_SCALE_EXTERNAL_PREMISES,
)
from omnibias.core.pulse_envelope import locked_pulse_grid_matches_tower
from omnibias.pinn.certified.forced_flat import (
    compose_locked_pulse_family,
    honesty_payload,
)

assert locked_pulse_grid_matches_tower()
report = compose_locked_pulse_family()
assert report["identity_holds"] is True
assert report["leftover_id"] is None
assert report["honesty"]["navier_stokes_proof_claim"] is False
assert report["honesty"]["forced_blowup_reproof_claim"] is False
assert report["honesty"]["pulses_leftover"] is False
assert honesty_payload()["pulses_leftover"] is True
assert "construction of each pulse family and the cutoff summation" in (
    NS_SCALE_EXTERNAL_PREMISES
)
```
