# Domain-program tools

These are better derivatives and bases, not discoveries.
The 1-D oscillator ground energy is `1/2`. The Harris sheet
is force-balanced. A 5-pair quarter-wave mirror already has
a closed-form reflectance.

```python
from omnibias.ferminet.hermite import (
    DISCLAIMER,
    gaussian_envelope_energy,
    honesty_payload,
    qho_ground_energy,
)

assert qho_ground_energy() == 0.5
assert abs(gaussian_envelope_energy(1.0) - 0.5) <= 1e-15
assert honesty_payload()["many_body_solution_claim"] is False
assert "not a many-body solution" in DISCLAIMER
```

```python
from omnibias.pinn.plasma import harris_force_errors

assert max(harris_force_errors(0.1)) <= 1e-12
```

```python
from omnibias.pinn.stack import g0_quarter_wave_report

report = g0_quarter_wave_report()
assert report["closed_matches"] is True
assert report["transfer_matches"] is True
```
