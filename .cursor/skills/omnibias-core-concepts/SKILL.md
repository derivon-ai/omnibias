---
name: omnibias-core-concepts
description: The conceptual foundations of omnibias — bias collapse as the multi-bias delta->0 limit to sigma^(K-1), the derivative tower, the Riccati family, OMBU / OperatorBlock including the closed-form integral, jets, and the three distinct limits (bias, temperature, Enclosure Collapse). Use when reasoning or writing about any core omnibias idea, when a term feels overloaded, or before inventing a new field on the primitive.
---

# Core concepts — the one idea in three registers

omnibias computes the closed-form n-th derivative `sigma^(n)(z)` for arbitrary
`n` from a **single** `sigma` evaluation, bit-identical across PyTorch / JAX /
Keras because every backend imports the **same** pure-Python coefficients.

## Why nested AD fails

Generic autodiff has no name for the object this library is built on: K
parallel hyperplanes coalescing to a derivative, an antiderivative window on
the same geometry, and a sound enclosure that contracts to a point plus a
proof. Nested AD gives `grad`; it does not give a tower, an OperatorBlock
role, or a vocabulary that keeps the three limits apart. Wrong words here
propagate into packages.

## What only this tower unlocks

| Concept | What it is | Canonical source |
|---|---|---|
| Derivative tower | `sigma, sigma', ..., sigma^(n)` in closed form, one eval | `omnibias.core.polynomials` |
| Riccati family | activations with a closed tower (`sigmoid'=s(1-s)`, `tanh'=1-t^2`, …) | `omnibias.core.spec`; `docs/theory.md` sec 4 |
| OMBU | trainable K-bias unit `f_K(z)=sum_k s_k sigma(z+b_k)` | `omnibias.torch.unit`, `omnibias.torch.stencil` |
| OperatorBlock | `identity / grad / laplacian / derivative / band / integral` | `omnibias.torch.blocks.operator`; `docs/operator-surface.md` |
| Closed-form integral | antiderivative window `S(z+b_hi)-S(z+b_lo)`, `S'=sigma` | `ActivationSpec.integral`; `analytic_integral` |
| Jets | exact truncated Taylor propagation | `omnibias.{torch,jax}.jet`, `.jet_mv`; `omnibias.core.bell` |

### Bias collapse — founding `delta -> 0`

With `K` biases on a difference stencil (spread `delta`) and signs
`s_k = (-1)^(K-k) * C(K-1, k-1) / delta^(K-1)`, the multi-bias unit
`f_K(z) = sum_k s_k * sigma(z + b_k)` converges to `sigma^(K-1)(z + b_mean)`
as `delta -> 0`. The closed-form tower evaluates this limit **exactly**.

With `z = w . x`, each bias `b_k` places a transition on the hyperplane
`w . x + b_k = 0`. A `K`-bias OMBU is **K parallel hyperplanes**. Bias
collapse is those K parallel hyperplanes **coalescing into one**; what
survives is the `(K-1)`-th derivative transverse to the single plane.

| Gap | Roles | Output |
|---|---|---|
| one plane (`K=1`) | `identity` | `sigma(z+b)` |
| gap `-> 0` | `grad` / `laplacian` / `derivative` | transverse tower `sigma^(n)` |
| gap held finite | `band` / `integral` | slab response, or mass via the antiderivative |

The closed-form `integral` is the same geometry read in the antiderivative
direction.

### Three distinct limits

| Sense | What moves | Limit | Output |
|---|---|---|---|
| **Bias collapse** (founding) | K biases coalesce | `delta -> 0` | smooth `sigma^(K-1)` |
| **Temperature collapse** | one gate sharpened | `beta -> inf` | 0/1 feasibility step |
| **Enclosure Collapse** | enclosure width | `width -> 0` of a *sound enclosure* | a point **plus a proof**, or `Inconclusive` |

Write **temperature collapse** for `beta -> inf`. Additional named collapses
in `omnibias.core.collapse` mint a different surviving object (verdict, germ
identity, winding number, weak residual, exact syzygy).

OperatorBlock has **six** roles, including `integral`. Three senses of
"integral": (1) antiderivative window; (2) domain quadrature; (3) `int f dmu`.
Matrix: `docs/operator-surface.md`.

## Use

Ground a concept in `docs/theory.md`, `docs/operator-surface.md`,
`omnibias.torch.blocks.operator`, `omnibias.torch.unit` / `.stencil`,
`omnibias.core.spec` / `omnibias.core.polynomials` before writing it into a
docstring or a new skill.

Two invention axes:

- **Derivative collapse** (`delta -> 0`) → discrete calculus / jets / umbral.
- **Temperature collapse** (`beta -> inf`) → discrete optimization / soft gates.

## Extend

Pair with `omnibias-derivative-tower` for the coefficient contract,
`omnibias-backends` for OperatorBlock, `omnibias-difference` for the founding
register, `omnibias-discrete` for temperature collapse consumers.

## Next invention

A new named collapse in `omnibias.core.collapse` whose surviving object is
genuinely new (not a rebrand of derivative / indicator / point-plus-proof)
and whose CI test pins the word **temperature collapse** wherever `beta -> inf`
appears.

## Further references

- `docs/theory.md` (two senses of collapse; geometric statement)
- `docs/operator-surface.md`
- `tests/test_terminology.py`
