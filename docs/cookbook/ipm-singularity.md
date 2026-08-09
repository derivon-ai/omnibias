# IPM self-similar singularities (scaffold)

Incompressible porous media (IPM) self-similar residual + smoke discovery for
the DeepMind unstable-singularity ladder. Honesty-first: local Darcy velocity
proxy, not Biot–Savart; **not** Navier–Stokes.

```python
import jax; jax.config.update("jax_enable_x64", True)
from omnibias.pinn.jax.discovery import ipm
from omnibias.pinn.certified.ipm import build_ipm_cap_bundle
from omnibias.symbolic.ipm import verify_ipm_bundle

out = ipm.run_ipm_discovery(ipm.IPMDiscoveryConfig(n=12, steps=40))
bundle = build_ipm_cap_bundle(out)
assert verify_ipm_bundle(bundle)["residual_samples_match"]
assert bundle["honesty"]["navier_stokes_proof_claim"] is False
```
