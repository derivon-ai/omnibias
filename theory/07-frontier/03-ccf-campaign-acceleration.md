# 07-03 CCF campaign acceleration

## 1. Thesis and status

The running CCF campaign is stalled at a documented **Hilbert-by-dictionary
floor near `1e-1`** against a stretch gate of `1e-13`, and the campaign's own
audit says more optimizer steps will not clear it — so the only path is to
enlarge the function class, which is precisely what the closed-form conjugate
tower does.

- **Status**: gated
- **Depends on**: 01-01, 01-04, 01-07, 01-12, 03-10, 07-01
- **Blocks**: none

## 2. Where it lands

`omnibias.core.verified.hardy_line` for the conjugate-tower dictionary,
`omnibias.pinn.certified.ccf_hardy` for the CCF-specific assembly, and
`benchmarks/reproduce_deepmind_ccf.py` plus `deepmind_campaign_tick.py` for the
campaign drivers. All exist.

## 3. Prior art in omnibias

This is the most concretely grounded spec in the tree, because the campaign is
live and its blocker is written down.

- `benchmarks/reproduce_deepmind_ccf.py` — compactified Omega-PINN,
  `hardy_corrected_pv` Hilbert, Martens-Grosse Gauss-Newton with exact JVP,
  optional multistage. Its module docstring states the blocker verbatim:
  *"spectral/PV Hilbert alone err at O(1e-1); with high `proj_defect_weight` the
  neural Omega is pulled into a Hardy span that itself floors near ~1e-1 under
  MG. Stretch remains unearned until Hilbert/dictionary capacity improves — more
  MG alone does not clear 1e-13."*
- `_next_actions()` in the same file emits
  `hilbert_dictionary_catch22_enrich_dict_or_free_omega_hilbert` when the
  residual exceeds `1e-2` under `hardy_corrected_pv`, and always emits
  `never_weaken_1e-13_stretch` and `never_forge_navier_stokes`.
- `benchmarks/_gates.py` — `CCF_STRETCH_RESIDUAL_GATE = 1e-13`,
  `CCF_RESIDUAL_GATE_1ST_UNSTABLE = 1e-11`,
  `CCF_LAMBDA_1ST_UNSTABLE = 0.6057`, `ccf_absolute_gates` with
  `anti_circularity: "targets are published digits, never empirical-law outputs"`.
- `omnibias.core.verified.hardy_line` — the generalized Cauchy-Hardy pair with
  `H[P] = Q`, `H[Q] = -P`, and verified derivative rules
  `P' = -alpha Q_{a, alpha+1}`, `Q' = alpha P_{a, alpha+1}`.
- `omnibias.pinn.certified.ccf_hardy`, `benchmarks/ccf_hardy_rung_acceptance.py`,
  `ccf_rung1_residual_push.py`, `ccf_line_discovery.py`,
  `deepmind_campaign_tick.py`.
- `.cursor/rules/deepmind-campaign.mdc` — the claim rules for this campaign.

**Confirmed gap.** The dictionary is finite and its Hilbert transform is applied
either spectrally (error `O(1e-1)`) or by projection onto a Hardy span whose own
approximation floor is `O(1e-1)`. There is no dictionary that is **closed under
both differentiation and the Hilbert transform**, which is exactly the object
spec 01-12 constructs.

## 4. Mathematics

### The catch-22, stated precisely

The CCF residual needs `H[omega]` where `omega` is the learned profile. Two
routes, both floored:

1. **Spectral / principal-value Hilbert on the compactified grid.** The Hilbert
   transform is non-local, so a truncated or discretized evaluation carries an
   error set by the grid and the tail treatment. Measured at `O(1e-1)`.
2. **Project `omega` onto a Hardy dictionary** whose Hilbert transform is known
   exactly, then transform exactly. Now the error is the *projection* error, and
   the dictionary's approximation floor is also `O(1e-1)`.

Raising `proj_defect_weight` moves error from route 1 to route 2 without
reducing it. That is the catch-22 the campaign names, and no amount of
Martens-Grosse iteration touches it, because both floors are **approximation-
theoretic, not optimization-theoretic**.

### Why the conjugate tower breaks it

Spec 01-12's observation is that the Hilbert transform commutes with
differentiation:

```
H[ f^(n) ] = ( H[f] )^(n)
```

So if a dictionary element `f` has closed-form `H[f]`, then **every derivative
of `f` is also in a dictionary with closed-form Hilbert transform**, at no extra
derivation cost. For the Cauchy-Hardy pair the tower closes exactly:

```
P' = -alpha Q_{a, alpha+1}          H[P] = Q
Q' =  alpha P_{a, alpha+1}          H[Q] = -P
```

so differentiating stays inside the pair and only shifts `alpha`. A dictionary
built from `{P_{a, alpha}, Q_{a, alpha}}` over a grid of scales `a` and orders
`alpha`, **closed under the derivative tower**, is therefore closed under `H`
as well.

The consequence for the campaign: the projection error and the Hilbert error are
no longer separate floors to trade against each other. Once `omega` is in the
span, `H[omega]` is exact, so route 2's error collapses to pure approximation
error of a dictionary that is now much larger for the same number of *derived*
elements — because every derivative of every element is free.

### Quantifying the dictionary enlargement

A dictionary of `M` base elements at `S` scales, closed to derivative order `N`,
has `M * S * (N + 1)` members, all with exact `H`. The campaign's current Hardy
span is the `N = 0` slice. Going to `N = 4` multiplies the span by `5` at the
cost of **zero additional `sigma` evaluations**, since the tower computes all
orders from one evaluation — which is the founding property of the library
applied to the campaign's actual bottleneck.

Whether a `5x` larger span moves an `O(1e-1)` floor by twelve orders of
magnitude is an entirely open empirical question, and the honest expectation is
that it does not. What it does do is remove the *structural* obstruction the
audit identifies, converting the question from "this cannot work" to "how far
does this get".

### Three secondary contributions

- **Multi-pack profiles.** Heterogeneous packs (spec 01-01) placed at the
  self-similar profile's features give matched jet coordinates there.
- **Irregular stencils.** The compactified grid is non-uniform; spec 01-04's
  confluent-Vandermonde weights are exact over the rationals for arbitrary node
  sets, replacing whatever uniform-grid approximation is currently used at the
  boundary.
- **Jet-Padé as an independent diagnostic.** Spec 03-10 locates the nearest
  complex singularity from a high-order jet. Used as a *diagnostic* it provides
  an independent check on the self-similar profile's analytic structure — and it
  must never be used to claim continuation of anything (spec 06-02, spec 07-01).

## 5. Worked example

**Counting the span, and being honest about the odds.**

Current configuration: a Hardy dictionary of `M = 32` base elements over
`S = 8` scales, so `256` members, `N = 0`. Measured residual floor: `~1e-1`.

Conjugate-tower configuration: same `M = 32`, same `S = 8`, closed to `N = 4`.
Span size `32 * 8 * 5 = 1280` members. Cost per member evaluation: unchanged,
because the tower yields orders `0` through `4` from one evaluation of the base
function.

**What a `5x` span buys, estimated honestly.** For a smooth target in a
well-chosen basis, approximation error typically falls at a rate like
`C n^{-p}` in the number of basis functions, with `p` set by the target's
smoothness relative to the basis. To go from `1e-1` to `1e-13` is `12` orders.
At `p = 2`, that would need a span ratio of `10^6`; at `p = 4`, `10^3`. A `5x`
increase buys, at `p = 4`, a factor of `5^4 = 625` — about `2.8` orders.

So the arithmetic says: **the conjugate tower alone plausibly moves the floor
from `1e-1` to somewhere near `1e-3` or `1e-4`, and does not reach `1e-13`.**
That estimate should be stated before running, because the alternative is
discovering it afterwards and describing a `2`-order improvement as progress
toward a gate that remains `9` orders away.

Two things make the estimate worth acting on anyway:

1. It is the difference between a floor that is **structural** (the audit's
   current reading: dictionary capacity is the binding constraint and cannot be
   raised for free) and one that is merely **distant**. Removing a structural
   obstruction is the prerequisite for everything else, and the campaign cannot
   proceed while the catch-22 stands.
2. The estimate assumes the error is approximation-limited with a fixed `p`. If
   the true obstruction is that the current dictionary **cannot represent the
   profile's local structure at all** — a qualitative, not quantitative, gap —
   then the multi-pack profile elements could produce a much larger jump. The
   benchmark distinguishes these by measuring the convergence exponent `p`
   empirically as `N` grows from `0` to `4`.

Measuring `p` is therefore the single most informative experiment in this spec,
and it is cheap: five runs at `N = 0, 1, 2, 3, 4` with everything else fixed.

**The reporting discipline.** `_next_actions()` already emits
`orders_to_stretch=log10(residual / 1e-13)`. At `1e-1` that reads `12.00`. If
the conjugate tower reaches `1e-4` it reads `9.00`. Reporting `9.00` is the
result. Writing "substantial progress toward the stretch gate" is not, and
`never_weaken_1e-13_stretch` is already in the action list precisely because
this is the moment the temptation appears.

## 6. Proposed API

```python
# omnibias/core/verified/hardy_line.py  -- additions
def hardy_conjugate_dictionary(
    scales: Sequence[float], alphas: Sequence[float], *, max_order: int,
) -> ConjugateDictionary:
    """Cauchy-Hardy pairs closed under the derivative tower, hence closed under
    the Hilbert transform. Every member carries its exact H."""

@dataclass(frozen=True)
class ConjugateDictionary:
    members: tuple[HardyMember, ...]
    max_order: int
    def hilbert(self, coeffs) -> FloatArray:
        """Exact, by the conjugate relation. No quadrature, no projection."""

# benchmarks/reproduce_deepmind_ccf.py  -- new arm
#   --dictionary conjugate --max-order N
# Reuses the existing gate constants unchanged.
```

The dictionary is pure-Python in `omnibias.core.verified`, consistent with the
pure-core rule; the torch and jax evaluation paths consume the same
coefficients.

## 7. Practical use cases

1. **Unblocking the campaign's named catch-22**, which is the only reason this
   spec exists.
2. **Measuring the convergence exponent `p`**, which tells the campaign whether
   the remaining distance is reachable by any dictionary enlargement or requires
   a different idea entirely. This is valuable even if the floor barely moves.
3. **Exact Hilbert transforms elsewhere** — any signal-processing or
   analytic-signal use in the library inherits the closed conjugate tower.
4. **An independent singularity diagnostic** for the self-similar profile, via
   jet-Padé, cross-checking the fitted `lambda`.

## 8. Acceptance gates

The gate constants are **not touched**. `CCF_STRETCH_RESIDUAL_GATE` stays
`1e-13` and Rung-1 stays `1e-11`.

- **G1 exact Hilbert.** For every dictionary member, the closed-form `H` agrees
  with a high-precision numerical Hilbert transform to `1e-14` relative. This
  validates the construction independently of the campaign.
- **G2 floor improvement.** The conjugate-tower arm's dense residual is at least
  `10x` below the current `hardy_corrected_pv` floor at matched compute, over
  three seeds. **A `10x` improvement leaves the stretch gate unearned and must
  be reported as such.**
- **G3 exponent measured.** The convergence exponent `p` is fitted from runs at
  `N = 0 ... 4` and reported with its uncertainty. This gate is about producing
  the number, not about its value.
- **G4 lambda unmoved.** The recovered `lambda` still matches
  `CCF_LAMBDA_1ST_UNSTABLE = 0.6057` within `abs_tol = 5e-5`, confirming the new
  dictionary does not distort the eigenvalue.
- **G5 anti-circularity preserved.** Targets remain published digits; the
  existing `ccf_absolute_gates` honesty block is emitted unchanged.
- **G6 no forged flags.** `navier_stokes_proof_claim`, `whole_line_certified`
  and `theorem_prover_verified` stay `False` unless genuinely earned. Asserted
  by test.

## 9. Benchmark plan

- New arm in `benchmarks/reproduce_deepmind_ccf.py` selected by
  `--dictionary conjugate --max-order N`; the `N` sweep for G3 in
  `benchmarks/ccf_conjugate_sweep.py`.
- Smoke JSON in `docs/benchmarks/`; heavy runs under
  `$OMNIBIAS_SCRATCH/deepmind_campaign/` as the existing `_scratch()` helper
  already does, with only the small gates summary committed.
- `deepmind_campaign_tick.py` extended so the tick reports the new arm's
  `orders_to_stretch`.

## 10. Honesty and scope

- **The stretch gate is never weakened.** `1e-13` is a property of the problem.
  If the method reaches `1e-4`, the artifact says `1e-4` and
  `orders_to_stretch=9.00`.
- The pre-registered expectation in section 5 is that the conjugate tower
  improves the floor by roughly `2` to `3` orders and does **not** earn the
  stretch gate. Recording that prediction before the run is what makes the
  result interpretable either way.
- Jet-Padé is used **only** as a diagnostic on the CCF profile. It is not
  applied to any Dirichlet series and claims no analytic continuation; spec
  07-01 records this boundary and spec 06-02 requires a test enforcing it.
- The founding `delta -> 0` bias collapse supplies the derivative tower that
  closes the dictionary. No temperature collapse appears.
- Certificate tier: the campaign's existing tiers, unchanged. CCF residual
  results are **empirical gates**; nothing here earns a Lean flag.
- CCF is a model problem studied in the context of fluid singularity formation.
  It is not Navier-Stokes, and `navier_stokes_proof_claim` stays `False`.

## 11. Open questions and risks

- **The estimate may be optimistic.** `p = 4` is a guess. If the true exponent
  is `1`, a `5x` span buys `0.7` orders and the approach is nearly worthless;
  G3 exists to find out.
- **Conditioning.** A dictionary closed to order `4` contains near-parallel
  elements (a high-order derivative at a large scale resembles a lower-order one
  at a smaller scale). The Gram matrix condition number must be reported, and it
  may cap the usable `N` well below where span size would.
- **The floor may not be the dictionary at all.** The audit's reading could be
  wrong, and the true limit could be the Omega-PINN's own representation or the
  Gauss-Newton solver's attainable residual. Then G2 fails flat, which is itself
  the most valuable possible outcome because it redirects the campaign.
- **`alpha`-grid choice.** The conjugate tower shifts `alpha` on
  differentiation, so a dictionary closed to order `N` needs `alpha` values
  spanning `N` shifts; a poorly chosen grid wastes most of the span.
- **Falsifier.** If G1 fails — the closed-form `H` does not match a
  high-precision numerical transform — the conjugate relation is being applied
  outside its validity conditions and the whole approach is void. G1 runs first
  and costs nothing.

## 12. Implementation checklist

- [ ] `hardy_conjugate_dictionary` in
      `packages/omnibias-core/src/omnibias/core/verified/hardy_line.py`
- [ ] G1 validation against a high-precision numerical Hilbert transform, run
      before anything else
- [ ] Gram-matrix condition number reported for each `N`
- [ ] `--dictionary conjugate --max-order N` arm in
      `benchmarks/reproduce_deepmind_ccf.py`
- [ ] `benchmarks/ccf_conjugate_sweep.py` fitting the convergence exponent `p`
- [ ] Multi-pack profile elements (spec 01-01) as a separate, ablatable arm
- [ ] Irregular-stencil boundary weights (spec 01-04) on the compactified grid
- [ ] Jet-Padé diagnostic, with a test asserting it never touches
      `omnibias.core.verified.dirichlet`
- [ ] `deepmind_campaign_tick.py` reporting the new arm's `orders_to_stretch`
- [ ] Gate constants untouched; test asserting `CCF_STRETCH_RESIDUAL_GATE`
      is still `1e-13`
- [ ] Heavy artifacts under `$OMNIBIAS_SCRATCH`, only gates JSON committed
- [ ] Index row in `theory/README.md`

## 13. Parent problem and the exact reason it stays an external obligation

**Parent: finite-time singularity formation for the 3D incompressible Euler and
Navier-Stokes equations** — the Clay Millennium Problem's negative side, and the
broader question the CCF model problem is studied as a proxy for.

CCF is a **one-dimensional model equation**. It shares a nonlinearity-plus-
Hilbert-transform structure with certain fluid models, which is why its
self-similar profiles are studied, but it is not the Euler or Navier-Stokes
system and a result about it is not a result about them. Beyond that, everything
this spec produces is a **numerically minimized residual on a fixed compactified
grid with a fixed dictionary** — an empirical quantity, at the lowest rung of the
claim ladder, not even a sound enclosure.

Two independent gaps therefore separate this work from the parent: model versus
system, and empirical residual versus proof. The campaign's own drivers encode
this — `never_forge_navier_stokes` is in every action list and
`navier_stokes_proof_claim` is pinned `False` in `ccf_absolute_gates`.

This spec does not claim, imply, or provide evidence for finite-time blowup in
Euler or Navier-Stokes.
