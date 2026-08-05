# Proof-carrying PDE certificates

This cookbook is the first end-to-end proof-carrying neural PDE path:

```python
from omnibias.core.proof import Conjecture
from omnibias.core.proof.certificate import verify_certificate_digest
from omnibias.core.verified.pde_certificate import (
    aposteriori_error_certificate,
    laplace,
    structural_invariant,
    user_stability_estimate,
)
from omnibias.pinn.certified import build_default_machine

layers = [([[2.0, -3.0]], [1.0], None)]  # u(x, y) = 2x - 3y + 1
domain = [(-1.0, 1.0), (-1.0, 1.0)]
stability = user_stability_estimate(
    0.5,
    1.0,
    source="manufactured harmonic fixture",
    pde_family="laplace",
    domain="unit square",
)
invariant = structural_invariant(
    "harmonic_affine",
    "Delta(2x - 3y + 1) = 0",
)

cert = aposteriori_error_certificate(
    layers,
    domain,
    laplace(2),
    stability=stability,
    invariants=[invariant],
    max_error=1e-6,
    splits=2,
)
assert verify_certificate_digest(cert.certificate)

machine = build_default_machine()
verdict = machine.evaluate(
    Conjecture(
        "affine harmonic PDE certificate",
        "pinn_aposteriori_error",
        {"certificate": cert.certificate, "max_error": 1e-6},
    )
)
assert verdict.status == "PROVED"
```

The certificate proves a **model statement**:

```text
||u_NN - u_true||_inf <= C_Omega * R_interior + C_boundary * R_boundary
```

where the residual terms are computed by the certified multivariate jet over the
whole domain. The stability constants are recorded with provenance; they are not
invented by the library.

## Certifying a Fourier mode (`sin` / `cos`)

The certified jet's function class includes the trigonometric pair, so a
closed-form **plane-wave** field is certifiable directly. A single `cos` layer
`u(x, y) = cos(w·x)` is an exact **Helmholtz eigenfunction** `Δu + |w|²u = 0`, and
the verified jet encloses that residual over the *whole* cell — tightening toward
zero as the box is subdivided (the interval-analysis signature):

```python
import math
from omnibias.core.verified.pde_certificate import certified_interior_residual, helmholtz

w1, w2 = 1.3, -0.7
k = math.hypot(w1, w2)                     # u = cos(w.x) solves Helmholtz with k = |w|
layers = [([[w1, w2]], [0.0], "cos")]      # one cos layer -> a single Fourier mode
domain = [(0.0, 1.0), (0.0, 1.0)]

for splits in (1, 4, 16, 64):
    res = certified_interior_residual(layers, domain, helmholtz(2, k), splits=splits)
    print(splits, res.mag)
# 1 -> 1.60e+00, 4 -> 9.36e-01, 16 -> 2.57e-01, 64 -> 6.54e-02  (-> 0)
```

The residual is *exactly* zero; the finite enclosures are interval dependency
overestimation that subdivision drives down soundly. This is a rigorous,
whole-domain enclosure for a constructed closed-form field — **not** a claim that
the network is the unique PDE solution, and **not** a continuum/global-regularity statement.
The same `cos`/`sin` layers feed `aposteriori_error_certificate` (above) when a
stability estimate is supplied.

## What the proof machine checks

The `pinn_aposteriori_error` prover applies the same gates as the other certified-evidence
stacks:

- schema validation, including stability provenance and invariant descriptors;
- digest validation through the v1 certificate format;
- independent replay of the a-posteriori arithmetic;
- honesty claims, so `unproven_claim=True` is blocked unless externally supported;
- optional Lean checking of the finite numerical margin `threshold - error_bound`.

That Lean check is intentionally narrow. It can confirm a finite inequality
carried by the sealed payload; it does **not** formalize the analytic PDE
stability theorem or any continuum global-regularity-grade statement.

## Runnable demo

The CI-sized demo lives in
[`examples/proof_carrying_pde/`](../../examples/proof_carrying_pde/):

```bash
python -m examples.proof_carrying_pde.run_demo
```

For trained `JetMLP` models, use `omnibias.verify.certify_pinn_aposteriori(...)`
to extract verified layers, record model provenance, adaptively evaluate the
residual, and seal the same certificate format.
