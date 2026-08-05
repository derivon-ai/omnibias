# omnibias-partition

A light, **certified soft partition-of-unity** primitive -- the keystone shared by four
downstream bridges (a discontinuity-capturing PINN, a region-wise Riemannian atlas,
per-region symbolic discovery, and a certified decision layer).

`depth` oblique split gates `g(x) = sigmoid(beta·(w·x − t))` route an input into
`2**depth` regions with weights `w_l(x)` that are **non-negative**, **sum to one**, and
**harden** to a crisp `{0,1}` partition as `beta → ∞`. On top of the weights it ships a
sound soft→hard membership-gap certificate and a per-region model registry whose one
`combine(X, beta) = Σ_l w_l · out_l` engine every bridge calls.

\[
\underbrace{\sum_l |w^{\text{soft}}_l - w^{\text{hard}}_l|}_{\text{measured}} \;\le\;
\underbrace{2\,\min\!\Big(1, \sum_j e_j\Big)}_{\text{certified sound bound}},
\qquad e_j = \sigma(-\beta\,|z_j|).
\]

Terminology: the gate's `beta → ∞` hardening is the **feasibility / temperature** sense of
"collapse" (a soft indicator becoming a 0/1 step), distinct from the **founding bias
collapse** -- the multi-bias `delta → 0` limit to the closed-form derivative `sigma^(K-1)`
(see [Theory](../theory.md)). The bridges differentiate products of sigmoids by autodiff,
**not** the closed-form derivative tower.

## Configuration & parameters (numpy)

::: omnibias.partition._core.config
    options:
      show_root_heading: false
      heading_level: 3

::: omnibias.partition._core.params
    options:
      show_root_heading: false
      heading_level: 3

## Partition-of-unity weights (numpy reference)

::: omnibias.partition._core.weights
    options:
      show_root_heading: false
      heading_level: 3

## Bit-identical backend twins

The same weights under torch (autograd-friendly, embeddable) and jax (`jit` / `grad` /
`vmap`), parity `~1e-9` (float64).

::: omnibias.partition.torch.weights
    options:
      show_root_heading: false
      heading_level: 3

::: omnibias.partition.jax.weights
    options:
      show_root_heading: false
      heading_level: 3

## Certificate

::: omnibias.partition.certify
    options:
      show_root_heading: false
      heading_level: 3

### Sound interval / gap primitives (numpy)

The always-available certificate engine: outward-rounded [`Interval`](core.md) enclosures of
each region weight over an input box, plus the closed-form soft→hard L1 gap. Imports only
`omnibias-core`, so it certifies any depth without a backend.

::: omnibias.partition._core.verified
    options:
      show_root_heading: false
      heading_level: 4

## Per-region model registry

::: omnibias.partition.registry
    options:
      show_root_heading: false
      heading_level: 3

## Bridges

`omnibias-partition` is the keystone under four submodules of the existing substrates:

- [`omnibias.pinn.partition`](pinn.md) -- discontinuity-capturing PINN (`u = Σ_l w_l u_l`).
- [`omnibias.geometry.atlas`](geometry.md) -- region-wise Riemannian metric (`g = Σ_l w_l G_l`).
- [`omnibias.symbolic.piecewise`](symbolic.md) -- per-region symbolic law discovery.
- [`omnibias.struct.decision`](struct.md) + [`omnibias.tab.decision`](tab.md) -- a certified
  differentiable decision layer.

Status: Alpha (`0.1.0a1`).
