# Boussinesq self-similar singularities (scaffold)

2-D Boussinesq-with-boundary self-similar residual scaffold — DeepMind's proxy
toward axisymmetric Euler-with-boundary. The empirical
\(\lambda_n \sim 1/(1.4187n+1.0863)+1\) relation is stored as a **hypothesis
artifact**, never a theorem.

```python
import jax; jax.config.update("jax_enable_x64", True)
from omnibias.pinn.jax.discovery import boussinesq
from omnibias.pinn.certified.boussinesq import build_boussinesq_cap_bundle
from omnibias.symbolic.boussinesq import verify_boussinesq_bundle

out = boussinesq.run_boussinesq_discovery(boussinesq.BoussinesqDiscoveryConfig(n=12, steps=40))
bundle = build_boussinesq_cap_bundle(out)
assert verify_boussinesq_bundle(bundle)["residual_samples_match"]
assert bundle["honesty"]["lambda_n_hypothesis_is_theorem"] is False
```
