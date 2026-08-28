# IPM radii construction

`build_ipm_cap_bundle` still packs discovery residuals as JSON. The residual +
tail + radii-polynomial path is `build_ipm_radii_construction`. The **named
simplified sub-case** that can close is `ipm_banded_toy_radii` (banded
quadratic Fourier manufactured solution). `full_ipm_proved` stays `False`.
`navier_stokes_proof_claim` stays `False`.

```python
from omnibias.pinn.certified.ipm import build_ipm_cap_bundle

bundle = build_ipm_cap_bundle(
    {
        "lam": 1.0,
        "residual_theta": [0.01],
        "residual_psi": [0.02],
        "validation_inputs": {"grid": "smoke"},
    }
)
assert bundle["honesty"]["navier_stokes_proof_claim"] is False
```

<!-- docs-test: skip reason="banded toy radii is a multi-second manufactured-solution certificate, not a 30-second docs smoke" -->
```python
from omnibias.pinn.certified.ipm import export_ipm_toy_cap_replay, ipm_banded_toy_radii

toy = ipm_banded_toy_radii()
assert toy["proved"]
assert toy["full_ipm_proved"] is False
trace = export_ipm_toy_cap_replay(toy)
assert len(trace.steps) >= 2
```

## API

::: omnibias.pinn.certified.ipm
    options:
      show_root_heading: false
      heading_level: 3
