---
name: omnibias-dev-core-concepts
description: The conceptual foundations a maintainer must get right before explaining, documenting, or extending omnibias -- what "bias collapse" actually is (the multi-bias delta->0 limit to sigma^(K-1)), the derivative tower, the Riccati family, OMBU / OperatorBlock, and jets -- plus the two distinct senses of "collapse" that must never be conflated, and the discipline of grounding a concept claim in canonical source before asserting it. Use when reasoning or writing about any core omnibias idea, when a term feels overloaded, or before inventing a new field/package on top of the primitive. For contributors modifying omnibias itself, not for consumers using it.
---

# omnibias core concepts (get these right before you build)

The whole library is one idea in three registers. Get the idea exactly right
*before* you explain it, document it, or build a new field on top of it: a
confident-but-wrong concept claim is the most contagious mistake here, because
it propagates from a chat message into docstrings, skills, and new packages.

## The thesis (one sentence)

omnibias computes the closed-form n-th derivative `sigma^(n)(z)` for arbitrary
`n` from a **single** `sigma` evaluation, with bit-identical results across
PyTorch / JAX / Keras because every backend imports the **same** pure-Python
coefficients.

## The mental-model map (with the one source of truth for each)

| Concept | What it is | Canonical source |
|---|---|---|
| Derivative tower | `sigma, sigma', ..., sigma^(n)` in closed form, one eval | `omnibias.core.polynomials` (`sigmoid_polynomial_coeffs`, `tanh_polynomial_coeffs`, `hermite_coeffs`) |
| Riccati family | activations with a closed tower (`sigmoid'=s(1-s)`, `tanh'=1-t^2`, softplus, gaussian, exp) | `omnibias.core.spec`; `docs/theory.md` sec 4 |
| OMBU | the trainable K-bias operator unit `f_K(z)=sum_k s_k sigma(z+b_k)` | `omnibias.torch.unit`, `omnibias.torch.stencil` |
| OperatorBlock | typed op dispatch: `identity / grad / laplacian / derivative / band / integral` | `omnibias.torch.blocks.operator` (code of record); `docs/operator-surface.md` (capability matrix) |
| Closed-form integral | antiderivative window `S(z+b_hi)-S(z+b_lo)`, `S'=sigma` (`OperatorBlock(op="integral")`) | `omnibias.core.spec` (`ActivationSpec.integral`); `omnibias.torch.unit` (`analytic_integral`) |
| Jets | exact truncated Taylor propagation through compositions | `omnibias.{torch,jax}.jet`, `.jet_mv`; combinatorics in `omnibias.core.bell`, `omnibias.core.multi_index` |
| Multi-pack (gated 01-01) | heterogeneous Birkhoff sample `sum_g c_g sigma^(n_g)(z+mu_g)` | `omnibias.core.multipack`; `omnibias.{torch,jax}.multipack`; `docs/api/multipack.md` |
| Bias scan (gated 01-02) | shared template on a bank of offsets along `w` | `omnibias.core.scan`; `omnibias.{torch,jax}.scan`; `docs/api/scan.md` |
| Irregular stencils (gated 01-04) | exact-`Q` Birkhoff weights; order asymptotic in `h` | `omnibias.difference._core.irregular`; `docs/api/difference.md` |
| Mollifier (gated 01-05) | pack as a test-function generator; certified exponential tails, **not** compact support; higher-order kernels take **negative** values | `omnibias.core.mollifier`; `docs/api/mollifier.md` |
| Band plan (gated 01-07) | pack order is a **band selector** (`BandPlan`, `peak_frequency`); 01-06 wavelet frames stay concept | `omnibias.core.spectral_design`; `docs/api/spectral_design.md` |

## Bias collapse -- the founding definition (canonical)

This is what the library is named for. It is a **`delta -> 0`** limit and it
produces a **derivative**:

> With `K` biases on a difference stencil (spread `delta`) and signs
> `s_k = (-1)^(K-k) * C(K-1, k-1) / delta^(K-1)`, the multi-bias unit
> `f_K(z) = sum_k s_k * sigma(z + b_k)` converges to `sigma^(K-1)(z + b_mean)`
> as `delta -> 0` (the biases collapse onto one value). The closed-form tower
> evaluates this limit **exactly**, with no `1/delta^(K-1)` catastrophic
> cancellation. It yields a smooth derivative, **never** a 0/1 step.

Endpoints of the same unit (see `docs/theory.md` sec 2-3):

- **Lemma identity** -- tied biases, signs summing to 1 => `f_K = sigma(z+b)`
  bit-identically (a fresh OMBU is a drop-in `sigma`).
- **Lemma collapse** -- stencil biases + finite-difference signs =>
  `f_K = sigma^(K-1)(z+b_mean)` (the unit *becomes* a differential operator).

### The geometric statement (say it this way when explaining)

With `z = w . x`, each bias `b_k` places a transition on the hyperplane
`w . x + b_k = 0`. Changing `b_k` slides that plane along `w` without rotating
it, so a `K`-bias OMBU is **`K` parallel hyperplanes**. Bias collapse is those
`K` parallel planes **coalescing into one**; what survives the merge is the
`(K-1)`-th derivative *transverse* to the single plane. One decision boundary,
carrying its derivative tower.

Every `OperatorBlock` role is a choice about the gap between those planes:

| Gap | Roles | Output |
|---|---|---|
| one plane (`K=1`) | `identity` | the boundary, `sigma(z+b)` |
| gap `-> 0` | `grad` / `laplacian` / `derivative` | transverse tower `sigma^(n)` |
| gap held finite | `band` / `integral` | the slab: its response, or its mass `S(z+b_hi)-S(z+b_lo)` |

So the closed-form `integral` is the *same* geometry read in the antiderivative
direction -- derivative and integral are the two directions of one construction,
not two unrelated features. `omnibias.core.probability` already words the `band`
case as "the probability mass of the slab between the two parallel hyperplanes".
Canonical write-up: `docs/theory.md` sec 4a.

Wave-1 gated extensions of this geometry (not shipped): heterogeneous packs
(`MultiPackUnit`), the position knob (`BiasScan` — interior shift along `w`;
`gamma` is not `delta -> 0`), and exact-`Q` irregular stencils. `BiasScan`
templates reuse the six `OperatorBlock` roles; it is not a seventh role.
Wave-3 knobs on the same geometry: a pack as a mollifier (`MollifierSpec` /
`tail_bound`; tails, not compact support) and pack order as a frequency
selector (`BandPlan`; not a Littlewood-Paley frame). Stacked consumers
(`ScanNet`, `JetKAN`, `fields.weak`, `pinn.interface`) stay gated; `alpha -> inf`
at an interface is sharpening, neither collapse.

**Guard the boundary.** "Single hyperplane" alone does **not** identify bias
collapse: `sigma(beta (w . x - t))` also has one hyperplane as its decision
boundary, and sharpening it is *temperature* collapse. The distinction is
**`K` parallel hyperplanes coalescing** (`delta -> 0`, yields a derivative)
versus **one hyperplane sharpened** (`beta -> inf`, yields a 0/1 step).

## Two senses of "collapse" -- DO NOT CONFLATE

"Collapse" is overloaded in this repo. Only the first is *the* bias collapse;
always qualify the others.

| Sense | What moves | Limit | Output | Where |
|---|---|---|---|---|
| **Bias collapse** (founding) | the `K` biases coalesce (spread `delta -> 0`) | finite difference -> derivative | a smooth `sigma^(K-1)` | `unit.py`, `stencil.py`, the tower |
| **Temperature collapse** (downstream) | one gate sharpened (`beta -> inf`) | soft -> hard threshold | a 0/1 feasibility step (indicator) | `omnibias.convex` / `omnibias.control` / `omnibias.routing` |
| **Operator/proximal collapse** | a `K` / config setting | reduces to a named classical operator | e.g. an L1 proximal map | activation dictionary, `docs/theory.md` sec 5 |

Trap: `beta -> inf` on a sigmoid saturates to 0 or 1 (a step) -- that is
**temperature collapse**, not a derivative. If you find yourself describing
"bias collapse" as a hard step or a constraint indicator, you mean temperature
collapse. Bias collapse takes **many** biases to **one** and yields a
derivative; temperature collapse takes **one** soft gate and hardens it into a
constraint. Different limit (`delta` vs `beta`), different output.

**Write "temperature collapse".** The older wordings -- "collapsed-bias penalty",
"bias-collapse penalty" -- are retired precisely because they invite the trap
above; `tests/test_terminology.py` fails the build if they come back.

## The operator surface -- six roles, incl. a closed-form INTEGRAL

`OperatorBlock` has **six** roles, not four. Under-stating this is a known,
contagious mistake (an agent once "corrected" itself into claiming omnibias has
no closed-form integral operator -- it does):

- `identity` (K=1): `sigma(z+b)`.
- `grad` (K=2) / `laplacian` (K=3) / `derivative` (K=n+1): closed-form
  `sigma^(n)(z+b_mean)` -- the bias-collapse fast path; `grad`/`laplacian` are
  the `n=1,2` aliases of `derivative`.
- `band` (K=2): the literal window `sigma(z+b_hi) - sigma(z+b_lo)`.
- `integral` (K=2): the **closed-form antiderivative** window
  `S(z+b_hi) - S(z+b_lo)` with `S'=sigma` (the `ActivationSpec.integral`
  kernel; e.g. `sigmoid`'s antiderivative is `softplus`). This is the
  fundamental-theorem twin of the derivative tower -- same OMBU machinery run in
  the antiderivative direction -- **not** a difference of `sigma` values (that
  is `band`).

"Integral" is overloaded across the repo; qualify which of the three you mean:
(1) the activation antiderivative window above (closed form, `OperatorBlock`);
(2) domain quadrature `sum_q w_q u(x_q)` (`omnibias.fields` / `-variational` /
`-geometry`, numerical); (3) the measure integral `integral f dmu`
(`omnibias.measure`, numerical; certified variant in `omnibias.verify`). Canonical
matrix: `docs/operator-surface.md`.

## Epistemic discipline (the rule that prevents the mistake)

Before asserting what a core concept means -- in chat, a docstring, a doc, or a
new skill -- **ground it in canonical source**: `docs/theory.md`,
`docs/operator-surface.md` (the operator/capability matrix),
`omnibias.torch.blocks.operator`, `omnibias.torch.unit` / `.stencil`,
`omnibias.core.spec` / `omnibias.core.polynomials`, and the
glossary (now in the separate `omnibias_web` project, `omnibias/docs/glossary.md`).
Do **not** generalize from a single
downstream package's docstring. The known failure mode is recency bias: after
working in `convex` / `control` / `routing`, their temperature-collapse penalty
feels like the definition of bias collapse -- it is not. When a term
feels overloaded, read the founding source first, then write.

## Inventing a new field on top of the primitive

Two axes flow from the two senses above; pick the right one and label it
honestly:

- **Derivative collapse** (`delta -> 0`, this document's founding sense) ->
  discrete *calculus* / analytic combinatorics (finite differences,
  Stirling / Bernoulli / Euler numbers read off the towers, umbral calculus,
  higher-order / Taylor-mode AD). The exact, bit-stable tower is the engine.
- **Temperature collapse** (`beta -> inf`, the feasibility sense) -> discrete
  *optimization* (LP/QP, combinatorial relaxations, routing, soft gates). A
  smooth surrogate annealed to a hard object, plus a certified gap.

Either way, keep the repo invariants (`omnibias-dev-derivative-tower` skill):
one shared coefficient source, bit-identical torch/jax twins, honest
closed-form vs autodiff vs numerical labels, and a regression test per change.
