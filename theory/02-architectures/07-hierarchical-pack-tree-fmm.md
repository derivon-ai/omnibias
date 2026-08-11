# 02-07 Hierarchical pack tree and the fast-multipole split

## 1. Thesis and status

A large bias bank costs `O(M)` per point and a dense interaction costs `O(N M)`;
splitting the bank into a **near-field / far-field hierarchy**, where far packs
are represented by a low-order expansion of their aggregate, reduces this to
near-linear cost with a controllable, computable error.

- **Status**: concept
- **Depends on**: 01-01, 01-02, 01-06, 01-07, 02-01
- **Blocks**: 02-06, 03-07

## 2. Where it lands

`packages/omnibias-core/src/omnibias/core/hierarchy.py` for the tree and error
algebra, plus backend evaluators in `omnibias.{torch,jax}.hierarchy`. A
submodule; the tree is a data structure, not a domain.

## 3. Prior art in omnibias

- Spec 01-02's `BiasScan` — the flat bank whose cost this reduces.
- `omnibias.core.spec.make_tempered_fastpath` — the exact scale law
  `sigma_alpha^(n)(u) = alpha^n sigma^(n)(alpha u)`, which is what lets a coarse
  level represent a group of fine packs.
- `omnibias.{torch,jax}.jet` — `compose_jet`, `tower_to_jet`, `jet_to_tower`:
  the machinery for converting between derivative towers and Taylor
  coefficients, which is exactly a multipole-to-local translation in one
  dimension.
- `omnibias.difference` — `pade_certified_remainder` and the certified
  truncation machinery, useful for bounding the far-field truncation.
- `omnibias.partition` — the depth-`d` tree structure, though over regions
  rather than over bank offsets.

**Confirmed gap.** No hierarchical evaluation exists anywhere. Every bank,
every pack sum, every layer potential is evaluated densely.

## 4. Mathematics

### The setup

A bank of `M` packs at offsets `tau_j` with weights `a_j`, evaluated at
pre-activation `z`:

```
F(z) = sum_{j=1}^{M} a_j sigma^(n_j)( z + tau_j )
```

Cost `O(M)` per evaluation point, `O(N M)` for `N` points.

### The far-field expansion

Group the packs into a cluster `C` with centre `tau_C`. For an evaluation point
with `|z + tau_C|` large compared with the cluster radius `rho_C`, expand each
term about the cluster centre:

```
sigma^(n_j)(z + tau_j) = sum_{k=0}^{p} ( (tau_j - tau_C)^k / k! ) sigma^(n_j + k)( z + tau_C )
                         + R_p
```

Summing over the cluster and collecting by total order,

```
sum_{j in C} a_j sigma^(n_j)(z + tau_j)
   = sum_{m} M^C_m  sigma^(m)( z + tau_C )  +  error
```

where the **multipole moments** are

```
M^C_m = sum_{j in C, n_j <= m} a_j (tau_j - tau_C)^(m - n_j) / (m - n_j)!
```

computed once per cluster, independent of the evaluation point. So a cluster of
any size is represented by `p + 1` numbers, and evaluating its far field costs
`O(p)` rather than `O(|C|)`.

This is the classical multipole idea, and it works here because the tower is
closed under differentiation: the expansion of a `sigma^(n)` term produces only
higher-order `sigma` terms at the same point, which are free.

### Error control

The remainder after `p` terms is bounded by

```
|R_p| <= ( rho_C^{p+1} / (p+1)! ) * max_{u in hull} | sigma^(n_j + p + 1)(u) |
```

and the derivative bound is available from the interval tower in
`omnibias.core.verified`. So the truncation error is a **computed quantity**,
not a tuning parameter, and the acceptance criterion for using the far-field
expansion (the standard "well-separated" test `rho_C / distance <= eta`) can be
set from a target accuracy.

### The tree

Build a binary tree over the offset axis. At each level, clusters have half the
radius. For an evaluation point, walk the tree: use the far-field expansion for
well-separated clusters, descend into near clusters, and evaluate directly at
the leaves. Standard, and the cost is `O((N + M) p)` for a one-dimensional
offset axis.

### The local expansion (the second half of FMM)

Converting a cluster's multipole moments into a *local* expansion about an
evaluation cluster is a translation operator. In this setting it is exactly the
`tower_to_jet` / `jet_to_tower` pair: a derivative tower at one point is a Taylor
jet, and shifting a Taylor jet is a triangular linear map. So the translation
operators are the jet machinery that already exists, applied to the offset axis
instead of the input axis.

This is the piece that turns `O((N + M) p)` into a genuinely FMM-shaped
algorithm, and it is also the piece most likely to be unnecessary: for
one-dimensional offset axes with modest `M`, the simple multipole-only version
may be enough. Measure before building the translation layer.

## 5. Worked example

Bank of `M = 8` order-1 packs at offsets `tau = (1.0, 1.1, 1.2, ..., 1.7)`, all
weights `a_j = 1`, evaluated at `z = -10` (far away: the distance to the cluster
is about `11.35`, the cluster radius is `0.35`, so `rho / d = 0.031`).

Cluster centre `tau_C = 1.35`. Multipole moments with `p = 2`:

```
M_1 = sum_j a_j = 8                                            (m = n_j = 1 term)
M_2 = sum_j a_j (tau_j - 1.35)                                 
    = (-0.35 - 0.25 - 0.15 - 0.05 + 0.05 + 0.15 + 0.25 + 0.35) = 0
M_3 = sum_j a_j (tau_j - 1.35)^2 / 2
    = (0.1225 + 0.0625 + 0.0225 + 0.0025) * 2 / 2 = 0.21
```

The first moment vanishes by symmetry, which is the usual and welcome accident.

Far-field value:

```
F_far(z) = M_1 sigma'(z + 1.35) + M_2 sigma''(z + 1.35) + M_3 sigma'''(z + 1.35)
```

At `z = -10`, `u = -8.65`. In this tail `sigma'(u) = sech^2(u) ~ 4 e^{2u}`, and
each further derivative multiplies by `2`:

```
sigma'(u)   = 4 e^-17.3         = 1.2268e-7
sigma''(u)  = 2 sigma'(u)       = 2.4536e-7
sigma'''(u) = 4 sigma'(u)       = 4.9072e-7
F_far = 8 * 1.2268e-7 + 0 * 2.4536e-7 + 0.21 * 4.9072e-7
      = 9.8144e-7 + 1.0305e-7 = 1.084491e-6
```

Direct evaluation of all eight terms gives `1.087714e-6`, so the absolute error
is `3.22e-9` and the relative error `2.96e-3` — three numbers replacing eight.
The predicted bound is

```
(rho^(p+1) / (p+1)!) * max |sigma^(4)| * (number of terms)
 = (0.35^3 / 6) * 1.98e-6 * 8 = 7.146e-3 * 1.98e-6 * 8 = 1.13e-7
```

which contains the observed error with a factor of about 35 to spare: sound, and
loose in the safe direction. Going to `p = 4` (the odd moments still vanish by
symmetry, so this adds one number) drops the relative error to `4.2e-5`, which
is the convergence rate the bound predicts.

Scaling that up: a bank of `M = 1000` offsets evaluated at `N = 1000` points
costs `10^6` activation evaluations densely. With a tree of depth 7 and `p = 6`,
the count drops to roughly `N * (log M) * p` plus the leaf work — two orders of
magnitude fewer, with an error bound that is computed rather than hoped for.

## 6. Proposed API

Does not exist yet.

```python
# omnibias/core/hierarchy.py
@dataclass(frozen=True)
class Cluster:
    centre: float
    radius: float
    members: tuple[int, ...]
    children: tuple["Cluster", ...] = ()

def build_pack_tree(offsets: Sequence[float], *, leaf_size: int = 8) -> Cluster: ...
def multipole_moments(cluster, offsets, weights, orders, *, p: int) -> tuple[float, ...]: ...
def truncation_bound(cluster, *, distance: float, p: int, deriv_bound: Interval) -> Interval: ...
def separation_for_accuracy(*, radius: float, p: int, target: float) -> float:
    """The eta that meets a target accuracy, so the tree criterion is derived."""
```

```python
# omnibias/torch/hierarchy.py  (and jax twin)
def hierarchical_scan(
    z: Tensor, tree: Cluster, offsets, weights, orders, *,
    p: int = 6, eta: float = 0.5, base: str = "tanh",
) -> Tensor:
    """Near/far split evaluation. Bit-identical to the dense path within the
    stated truncation bound, and exactly equal when eta = 0 (all near)."""
```

The `eta = 0` exactness property is the key testability hook: the hierarchical
path must reduce to the dense path exactly.

## 7. Practical use cases

1. **Large scan banks** in Scan-Net (spec 02-01) at high resolution.
2. **Boundary-integral evaluation** (spec 02-06), where the dense operator is
   the binding cost.
3. **Long-range interaction models** generally: any sum over many sources
   evaluated at many targets.
4. **Adaptive refinement** (spec 03-13): a tree makes adding a pack local, so
   refinement does not rebuild the whole evaluator.
5. **Certified far-field bounds.** The truncation bound is an `Interval`, so a
   hierarchical evaluation can be part of a sound enclosure rather than only a
   speedup.

## 8. Acceptance gates

Baseline: dense evaluation.

- **G1 exact reduction.** With `eta = 0`, the hierarchical path equals the dense
  path bit-for-bit.
- **G2 bound soundness.** `truncation_bound` upper-bounds the observed error on
  a dense grid **and** a random sample, over `p = 1 .. 10` and a range of
  separations, with zero violations.
- **G3 complexity.** Measured cost scales as predicted (near-linear in `N + M`)
  over `M` spanning two decades, with the crossover point against dense
  evaluation recorded.
- **G4 accuracy at target.** Given a target accuracy, `separation_for_accuracy`
  produces a tree whose measured error meets the target on every test instance.
- **G5 parity.** torch and jax bit-identical.

## 9. Benchmark plan

- `benchmarks/pack_tree.py`: error versus `p`, cost versus `M` and `N`,
  crossover table, bound tightness.
- Smoke JSON committed; full under `$OMNIBIAS_SCRATCH/hierarchy/`.

## 10. Honesty and scope

- The packs collapse by the founding `delta -> 0` bias collapse; the hierarchy
  is an evaluation strategy on top and introduces no new limit. No temperature
  collapse appears.
- The construction above is for a **one-dimensional offset axis**. Multi-normal
  arrangements need a spatial tree and the analysis is genuinely different; do
  not claim `D`-dimensional FMM from a 1-D derivation.
- The far-field expansion is a **truncation**, so it is approximate by design.
  Every use must carry either the bound or an explicit statement that the dense
  path was used.
- Certificate tier: sound enclosure for the truncation bound.

## 11. Open questions and risks

- **The crossover may be large.** For modest `M`, dense evaluation is vectorized
  and cache-friendly while the tree is branchy. The honest deliverable may be
  "use dense below `M = 200`", and the benchmark must find that number rather
  than assume the tree always wins.
- **Batched hardware.** Tree traversal is awkward on accelerators. A
  level-by-level, fixed-shape formulation is likely necessary for the jax path
  to stay traceable, and it may cost much of the theoretical speedup.
- **Translation operators may be unnecessary.** Build the multipole-only version
  first and only add local expansions if measurement demands it.
- **Falsifier.** If the crossover exceeds the `M` any realistic use needs, this
  is an interesting derivation with no operational value.

## 12. Implementation checklist

- [ ] `packages/omnibias-core/src/omnibias/core/hierarchy.py`
- [ ] torch and jax twins, with the jax path written for fixed shapes
- [ ] `eta = 0` bit-exactness test against dense evaluation
- [ ] Truncation-bound soundness test (dense grid plus random sample)
- [ ] Complexity and crossover measurement, recorded in the artifact
- [ ] `benchmarks/pack_tree.py` plus smoke JSON
- [ ] Docs page and nav entry
- [ ] Index row in `theory/README.md`
