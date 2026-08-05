---
name: omnibias-verify
description: Use omnibias to produce rigorous, sound certificates -- certified output bounds, robustness / Lipschitz / monotonicity, certified global minima, proof-carrying training, and Fokker-Planck / Ito residual enclosures -- plus validated dynamics. Use when using omnibias to certify or verify a trained network, enclose a quantity with guaranteed bounds, or when the user mentions certificates, verification, interval / Taylor-model bounds, or validated dynamics.
---

# Using omnibias: rigorous certification

`omnibias.verify` propagates rigorous (interval / Taylor-model) enclosures
through a trained net and seals tamper-evident certificates. Everything is
**sound by construction**: a certified verdict is a proof.

## Import map

| You want | Import from | Key entry points |
| --- | --- | --- |
| Rigorous scalars | `omnibias.core.verified.interval` | `Interval` (outward-rounded) |
| Neutral net + enclosures | `omnibias.verify` | `Network`, `affine_layer`, `TanhLayer`, `taylor_output_bounds` |
| Property certificates | `omnibias.verify` | `certify_robustness`, `lipschitz_bound`, `interval_jacobian`, `monotonicity`, `reachable_box` |
| Certified global optimum | `omnibias.verify` | `certified_minimize`, `certified_network_minimize` |
| Proof-carrying training | `omnibias.verify` | `certify_trained_network`, `certify_trained_min`, `certify_trained_global_min`, `param_jet` |
| SDE operator residuals | `omnibias.verify` | `certify_fokker_planck_residual`, `certify_ito_generator_residual` |
| Validated ODE dynamics | `omnibias.dynamics` | `variational_flow`, `prove_periodic_orbit`, `certified_lyapunov_exponent` |

## Minimal example (verified)

```python
from omnibias.core.verified.interval import Interval
from omnibias.verify import Network, affine_layer, TanhLayer, taylor_output_bounds

net = Network([
    affine_layer([[1.0, 1.0], [1.0, -1.0]], [0.0, 0.0]),
    TanhLayer(),
    affine_layer([[1.0, 1.0]], [0.0]),
])
box = [Interval(-0.4, 0.4), Interval(-0.4, 0.4)]
(out,) = taylor_output_bounds(net, box, order=3)   # tighter-than-IBP, provably contains the true output
```

## Gotchas that bite

- **`certified_*` returns an interval / enclosure, never a point.** That is the whole point: `lo <= true value <= hi` rigorously. Differentiable kernels return a backprop-able scalar; certified ones return bounds.
- **Soundness, not completeness.** The verifier returns *inconclusive* rather than over-claim when branch-and-bound runs out of budget; enclosures are reported honestly, never silently widened.
- **State the scope.** These are rigorous **local** (small net, low-dimensional box, `tanh`/`sigmoid`/`gaussian`) or **global-over-a-box** proofs -- not continuum or global-regularity-grade claims. `certify_trained_min` is local; `certify_trained_global_min` is global over a parameter box; the stochastic residuals are local (finite-box).
- **`theorem_prover_verified` is earned, not asserted.** It is set only on a genuine Lean `lake build` pass and degrades gracefully (stays `False`) with no toolchain.

## More detail

- API: [verify](https://github.com/derivon-ai/omnibias/blob/main/docs/api/verify.md), [dynamics](https://github.com/derivon-ai/omnibias/blob/main/docs/api/dynamics.md)
- Cookbook: [train-then-certify](https://github.com/derivon-ai/omnibias/blob/main/docs/cookbook/train-then-certify.md), [proof-carrying PDE](https://github.com/derivon-ai/omnibias/blob/main/docs/cookbook/proof-carrying-pde.md)
