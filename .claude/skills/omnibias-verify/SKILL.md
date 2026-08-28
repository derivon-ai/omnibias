---
name: omnibias-verify
description: Produce rigorous, sound certificates — Taylor-model output bounds, robustness / Lipschitz / monotonicity, certified minima, proof-carrying training, and Fokker-Planck / Ito residual enclosures — plus validated dynamics. Use when inventing a new enclosure, sealing a verdict, or when the user mentions certificates, interval / Taylor-model bounds, or validated dynamics.
---

# Sound certificates on the same tower

`omnibias.verify` propagates interval / Taylor-model enclosures through a
trained net and seals tamper-evident certificates. Differentiable kernels
return a backprop-able scalar; certified kernels return `lo <= true <= hi`.

## Why nested AD fails

Autodiff returns a point. Interval bound propagation without a closed-form
tower inflates until the box is vacuous. Generic verifiers ingest ReLU graphs
and cannot read omnibias jets, seal a hash-evident JSON certificate, or earn
`theorem_prover_verified` from a genuine `lake build`. High-order remainders
are exactly where nested AD is most expensive and least rigorous.

## What only this tower unlocks

Taylor models whose polynomial part is the exact `sigma^(n)` jet, remainder
enclosed outward-rounded, tighter than IBP by construction. Robustness,
Lipschitz, monotonicity, reachable sets, certified minima, proof-carrying
training, and SDE residual enclosures compose that object. Validated dynamics
(`omnibias-dynamics`) reuses the same QR-Lohner / radii-polynomial substrate.
This is the rigorous register of the one tower.

## Use

| You want | Import from | Key entry points |
| --- | --- | --- |
| Rigorous scalars | `omnibias.core.verified.interval` | `Interval` (outward-rounded) |
| Neutral net + enclosures | `omnibias.verify` | `Network`, `affine_layer`, `TanhLayer`, `taylor_output_bounds` |
| Property certificates | `omnibias.verify` | `certify_robustness`, `lipschitz_bound`, `interval_jacobian`, `monotonicity`, `reachable_box` |
| Certified global optimum | `omnibias.verify` | `certified_minimize`, `certified_network_minimize` |
| Proof-carrying training | `omnibias.verify` | `certify_trained_network`, `certify_trained_min`, `certify_trained_global_min`, `param_jet` |
| SDE operator residuals | `omnibias.verify` | `certify_fokker_planck_residual`, `certify_ito_generator_residual` |
| Validated ODE dynamics | `omnibias.dynamics` | `variational_flow`, `prove_periodic_orbit`, `certified_lyapunov_exponent` |
| Composed encoder + tab head | `omnibias.tab` | `certify_composed` |
| Localization / PCI / uncertainty | `omnibias.verify.localization`, `omnibias.verify._core.pci`, `omnibias.verify.uncertainty` | Krawczyk unique-peak, TM hidden state, conformal slabs |
| Certified train step | `omnibias.verify.train_step` | accept `theta'` only if a Lipschitz / output-box stays in cap |

```python
from omnibias.core.verified.interval import Interval
from omnibias.verify import Network, affine_layer, TanhLayer, taylor_output_bounds

net = Network([
    affine_layer([[1.0, 1.0], [1.0, -1.0]], [0.0, 0.0]),
    TanhLayer(),
    affine_layer([[1.0, 1.0]], [0.0]),
])
box = [Interval(-0.4, 0.4), Interval(-0.4, 0.4)]
(out,) = taylor_output_bounds(net, box, order=3)
```

`certified_*` returns an enclosure. Inconclusive is first-class when
branch-and-bound exhausts its budget. `theorem_prover_verified` is set by a
genuine kernel `lake build` (see `omnibias-certificate-lean`).
`certify_composed` is IBP / `tab+tab` when ingest works; otherwise
`sampled_latent`.

Cookbook: `docs/cookbook/train-then-certify.md`,
`docs/cookbook/proof-carrying-pde.md`.

## Extend

- Source: `packages/omnibias-verify`. Primitives: `omnibias-verified-primitive`.
- Tests: `python -m pytest packages/omnibias-verify/tests -q`.
- Compose with `omnibias-dynamics`, `omnibias-formal`, `omnibias-sos`,
  `omnibias-control`, `omnibias-tab`, `omnibias-frontier`.

## Next invention

A trained tanh net whose Taylor-model output box, Lipschitz enclosure, and
certified global min over a parameter box seal together, then a Lean kernel
pass on the finite rational core of that verdict.

## Further references

- API: `docs/api/verify.md`, `docs/api/dynamics.md`
- Scope: `docs/scope-and-guarantees.md`
