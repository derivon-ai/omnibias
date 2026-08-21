# Boussinesq self-similar singularities (scaffold)

2-D Boussinesq-with-boundary self-similar residual scaffold — DeepMind's proxy
toward axisymmetric Euler-with-boundary. The empirical
\(\lambda_n \sim 1/(1.4187n+1.0863)+1\) relation is stored as a **hypothesis
artifact**, never a theorem.

Pipeline adapter: `omnibias.pinn.jax.discovery.pipeline.BoussinesqAdapter`
(Adam smoke until CubicGN earn path lands). CPU smoke:
`python benchmarks/ipm_boussinesq_scaffold_smoke.py --family boussinesq`
→ [`docs/benchmarks/boussinesq_scaffold_smoke.json`](../benchmarks/boussinesq_scaffold_smoke.json).

```python
import jax; jax.config.update("jax_enable_x64", True)
from omnibias.pinn.jax.discovery import boussinesq
from omnibias.pinn.jax.discovery.pipeline import BoussinesqAdapter, run_singularity_pipeline
from omnibias.pinn.certified.boussinesq import build_boussinesq_cap_bundle
from omnibias.symbolic.boussinesq import verify_boussinesq_bundle

out = boussinesq.run_boussinesq_discovery(boussinesq.BoussinesqDiscoveryConfig(n=12, steps=40))
bundle = build_boussinesq_cap_bundle(out)
assert verify_boussinesq_bundle(bundle)["residual_samples_match"]
assert bundle["honesty"]["lambda_n_hypothesis_is_theorem"] is False
pipe = run_singularity_pipeline(BoussinesqAdapter(n=8, steps=5), None)
assert pipe.honesty["navier_stokes_proof_claim"] is False
```

Optional paper chart (eqs. 5–6): hats on `(q, β)` with closed-form envelope
jets. This is still a scaffold, not a DeepMind residual claim.

```python
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from omnibias.pinn.jax.equations.boussinesq_compactified import (
    affine_hat_jet,
    compactify_yb_lambda,
    compose_boussinesq_envelope_fields,
)
from omnibias.pinn.jax.equations.boussinesq_selfsimilar import (
    infer_lambda_from_streamfunction_u1_y1,
)
from omnibias.pinn.jax.discovery import boussinesq

y1 = jnp.array([0.0, 0.2, -0.2])
y2 = jnp.array([0.0, 0.1, 0.1])
lam = 1.5
q, beta, _r2 = compactify_yb_lambda(y1, y2, lam)
assert float(q[0]) == 1.0
hat = affine_hat_jet(q, beta, 0.4, 0.0, 0.0)
fields = compose_boussinesq_envelope_fields(
    y1, y2, lam, hat_omega=hat, hat_theta=hat, hat_psi=hat
)
assert fields["omega"].shape == y1.shape
assert float(infer_lambda_from_streamfunction_u1_y1(0.0)) == -3.0
out = boussinesq.run_boussinesq_discovery(
    boussinesq.BoussinesqDiscoveryConfig(n=6, steps=2, compactified=True)
)
assert out["honesty"]["deepmind_residual_claim"] is False
assert out["honesty"]["navier_stokes_proof_claim"] is False
```
