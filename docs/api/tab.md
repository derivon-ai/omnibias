# omnibias-tab

Differentiable, **exactly second-order-trained**, and **certified** soft decision-tree
ensembles for tabular data. A tree split is a hard threshold `1[w.x > t]`; omnibias makes
it a **soft oblique gate** `g(x) = sigmoid(beta·(w.x − t))` and anneals `beta → ∞` toward
a genuine hard split. Learning an *optimal* hard tree is NP-hard, so `tab` is a well-posed
**yes-if** object, not an exactness claim:

\[
\underbrace{\ell}_{\text{certified output / rounding bound}} \;\le\; f(x) \;\le\;
\underbrace{u}_{\text{certified output / rounding bound}} .
\]

1. A **differentiable soft-tree ensemble** -- oblivious soft trees whose `depth == 1` tier
   is a pure sum-of-sigmoids (additive, directly certifiable) and whose `depth >= 2` tier
   multiplies gates into `2**depth` leaf memberships (native interactions). Bit-identical
   torch + jax forwards (parity `~1e-9`, float64).
2. **Exact second-order training** of the whole model (splits included) via
   `omnibias.torch.optim`, plus a stagewise **Newton-boosting** driver (GBM-mirror) using
   the closed-form loss curvature.
3. **Sound certificates**: output bounds, Lipschitz, per-feature monotonicity, optional
   sealed scalar global-min, and a certified **train-soft / deploy-hard** rounding gap as
   `beta → ∞`.

Terminology: the gate's `sigmoid(beta·)`, `beta → ∞` is the **feasibility / temperature**
sense of "collapse" (a soft indicator hardening to a 0/1 step), distinct from the
**founding bias collapse** -- the multi-bias `delta → 0` limit to the closed-form
derivative `sigma^(K-1)` (see [Theory](../theory.md)). Beating LightGBM is a *benchmarked*
claim, earned per dataset (see [Benchmarks](../benchmarks.md)), never asserted.

## Configuration & parameters (numpy)

::: omnibias.tab._core.config
    options:
      show_root_heading: false
      heading_level: 3

::: omnibias.tab._core.params
    options:
      show_root_heading: false
      heading_level: 3

## Reference forward & losses (numpy)

::: omnibias.tab._core.forward
    options:
      show_root_heading: false
      heading_level: 3

::: omnibias.tab._core.loss
    options:
      show_root_heading: false
      heading_level: 3

## Trainable module & second-order trainers (torch)

::: omnibias.tab.torch.model
    options:
      show_root_heading: false
      heading_level: 3

::: omnibias.tab.torch.train
    options:
      show_root_heading: false
      heading_level: 3

## Arrangement classifier (05-02)

Hyperplane-arrangement view of tabular classification: ``H`` soft gates applied
to every input, soft cell memberships over ``2**H`` sign patterns (reusing
``omnibias.partition``), per-cell logits, beta anneal, an optional ``L1`` /
sparse warm-start path, and ``certify_arrangement_gap`` (partition soft->hard
gap). Wave-0 gates G1/G2 are earned on constructed datasets; G3 reports the
full win/loss table on eight public binary sets via
``benchmarks/tabular_arrangement_public.py`` (trees / LightGBM still expected
to win most). Capacity / optimizer ablations (Newton, H=3/4, Newton-boosted
H=2, SoftTree) live in ``benchmarks/tabular_arrangement_capacity.py``; G3b
stays unearned (predeclared ``boost_h2`` not-worse on 4/8; need >=6/8).
The obliqueness diagnostic detects linear oblique structure only
(not XOR parity) and is frozen for G4 reporting.

**Plugin contract.** ``as_head(z, kind)`` returns a :class:`~omnibias.tab.torch.plugin.TabHead`
wrapping SoftTree / Arrangement / Boosted on ``z.device`` / ``z.dtype``.
``forward`` is tensor-in / tensor-out on ``(..., d)`` with logits ``(..., k)``
(``k=1`` for binary; numpy ``predict`` squeezes so G3 stays 1-D). Attribute
access forwards to the inner module (``head.W`` / ``head.members``). ``beta`` / ``lr`` are buffers (optional
``learnable_beta``). Compose with any encoder via ``fit_joint(encoder, head, X, y)``
or optional ``encoder=`` on ``fit_first_order`` / ``fit_second_order`` /
``fit_arrangement`` (G3 path unchanged; stagewise ``fit_boosted`` /
``fit_arrangement_boosted`` raise ``TypeError``). Host optimizer -- see
``docs/examples/tab_as_layer.py`` and
``tab_as_layer_jax.py``. JAX: ``arrangement_forward`` / ``boosted_forward``.
Optional Equinox wrappers (``[equinox]`` extra) live in
``omnibias.tab.jax.equinox_head`` (``ArrangementHead`` / ``SoftTreeHead`` /
``BoostedHead``) so ``import omnibias.tab.jax`` stays Equinox-free.
Equinox tests **fail** when ``CI`` is set and the extra is missing; local
runs still ``importorskip``. Keras 3 layers live in ``omnibias.tab.keras``
(not ``omnibias-keras``). ``torch.compile`` over all three heads is required
on CI (CPU inductor); local runs skip if inductor is missing. CUDA AMP stays
skip-if-no-CUDA. Recipe: [`cookbook/tab-as-layer.md`](../cookbook/tab-as-layer.md).

::: omnibias.tab.torch.plugin
    options:
      show_root_heading: false
      heading_level: 3

::: omnibias.tab.arrangement
    options:
      show_root_heading: false
      heading_level: 3

::: omnibias.tab.torch.arrangement
    options:
      show_root_heading: false
      heading_level: 3

## Newton-boosting driver (torch)

::: omnibias.tab.torch.boosting
    options:
      show_root_heading: false
      heading_level: 3

## Functional forward twin (JAX)

Bit-identical JAX twin of the soft-tree forward (parity `~1e-9`, float64),
plus arrangement / boosted-arrangement kernels for the same plugin contract.

::: omnibias.tab.jax.model
    options:
      show_root_heading: false
      heading_level: 3

::: omnibias.tab.jax.arrangement
    options:
      show_root_heading: false
      heading_level: 3

## Closed-form tree jets

Product-of-sigmoids jets of the **soft** surrogate (``delta -> 0``), not of the
hardened ``beta -> inf`` tree. Depth-1 uses the additive ``Linear -> sigmoid ->
Linear`` weights (same as ``to_additive_sequential``) then ``mlp_jet_mv``
(multivariate) / ``mlp_jet`` (directional); depth ``>= 2`` uses the Leibniz
product of per-gate sigmoid jets.

::: omnibias.tab.torch.jet
    options:
      show_root_heading: false
      heading_level: 3

::: omnibias.tab.jax.jet
    options:
      show_root_heading: false
      heading_level: 3

## Keras 3 layers

``omnibias.tab.keras`` (optional ``[keras]`` extra). ``ArrangementBoosted`` is the
stacked-member twin of the torch ensemble. ``learnable_beta`` makes ``beta`` a
trainable weight on SoftTree / Arrangement; on ``ArrangementBoosted`` it is
forwarded to **members** (ensemble ``learning_rate`` / ``base`` stay frozen).
``ArrangementBoosted.build()`` builds members so Keras 3 does not warn on first
call. TF is ``keras.ops``-identical; CI smokes ``KERAS_BACKEND=jax``,
``KERAS_BACKEND=tensorflow``, and ``KERAS_BACKEND=torch`` on ``test_keras.py``
(TensorFlow is not a required dep; torch is already in the tab job). The full
tab suite is not run under Keras-torch.

::: omnibias.tab.keras.model
    options:
      show_root_heading: false
      heading_level: 3

::: omnibias.tab.keras.arrangement
    options:
      show_root_heading: false
      heading_level: 3

## Certificates

::: omnibias.tab.certify
    options:
      show_root_heading: false
      heading_level: 3

### Sound interval primitives (numpy)

The always-available certificate engine: outward-rounded [`Interval`](core.md) enclosures of
the forward, the by-gate Jacobian (for monotonicity / Lipschitz), and the soft→hard rounding
gap. Imports only `omnibias-core`, so it certifies any depth without a backend.

::: omnibias.tab._core.verified
    options:
      show_root_heading: false
      heading_level: 4

## Certified leaf-routing decision

`omnibias.tab.decision` (a thin adapter; **no change to tab's trained forward or
benchmark path**) turns a fitted soft tree's leaf memberships into a *certified
decision*: it maps them to Gibbs-law leaf logits and reuses
[`omnibias.struct`](struct.md)'s `certify_argmax`, so `certified_leaf_decision`
returns the routed leaf plus a sound `log(n_leaves)/beta` selection certificate
and an `L^inf` routing-stability radius that hardens as `beta -> ∞`. This is the
feasibility / measure-mode collapse axis, never the founding `delta → 0` bias
collapse.

::: omnibias.tab.decision
    options:
      show_root_heading: false
      heading_level: 3

## Benchmark harness (vs LightGBM)

A fair, multi-seed head-to-head against gradient boosting -- the reusable engine behind the
CPU-smoke [`docs/examples/tab_validate.py`](https://github.com/derivon-ai/omnibias/tree/main/docs/examples/tab_validate.py)
and the cluster sweep `packages/omnibias-tab/bench/sweep.py`. "Not worse than LightGBM" is a
*benchmarked* verdict (within the baseline's own across-seed noise), never asserted.

::: omnibias.tab.bench
    options:
      show_root_heading: false
      heading_level: 3

Status: Alpha (`0.1.0a1`).
