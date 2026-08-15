# 01-03 Hyperplane arrangement geometry

## 1. Thesis and status

Many units with different normals define a **hyperplane arrangement**: its
cells, flats and face lattice are the combinatorial object that every downstream
region-based construction (constraint sets, polytopes, graph networks) actually
uses, and the soft version is a differentiable weighting of those cells.

- **Status**: gated (G1–G4 CI; temperature collapse, sampled subgraph; cost gates smoke-earned, not in CI `all_passed`)
- **Depends on**: none
- **Blocks**: 01-08, 01-10, 02-02, 02-12, 03-02, 03-03, 03-09, 04-02, 05-02, 07-05

## 2. Where it lands

A submodule of `omnibias-partition`
(`packages/omnibias-partition/src/omnibias/partition/arrangement/`), because it
shares the audience, the dependency tier, and the soft-region domain. Promote
only if the face-lattice machinery grows its own consumers.

## 3. Prior art in omnibias

- `packages/omnibias-partition/src/omnibias/partition/_core/config.py` —
  `PartitionConfig(n_features, depth, split_kind="oblique"|"axis"|"sparse", ...)`,
  and `partition_weights(params, X, beta)` producing `2**depth` region weights
  that form a partition of unity and harden as `beta -> inf`.
- `certify_partition_gap(params, X, *, beta)` — a sound soft-to-hard gap.
- `packages/omnibias-shape/.../ops/occupancy.py` — `soft_polytope`, `soft_box`:
  intersections of soft halfspaces as occupancy fields.
- `packages/omnibias-pinn/src/omnibias/pinn/domain/_core/sdf.py` — `Halfspace`,
  and the R-function CSG `r_intersect_sdf` / `r_union_sdf` / `RCompose`, which
  give smooth exact set algebra on implicit surfaces.

**The distinction that matters.** A depth-`d` oblique partition is a **tree**:
`d` gates, `2**d` regions, each region a conjunction of exactly `d` literals in
a fixed order. An arrangement of `n` hyperplanes in `R^D` is not a tree: it has
up to `sum_{i=0}^{D} C(n, i)` full-dimensional cells and a face lattice of all
dimensions. The tree is a *subset* of arrangements with a special structure.
This spec supplies the general object and states honestly when it is affordable.

## 4. Mathematics

### Objects

Given normals `w_1 .. w_n` and offsets `b_1 .. b_n` in `R^D`, with
`h_i(x) = w_i . x + b_i`:

| Object | Definition | Dimension |
|---|---|---|
| hyperplane `H_i` | `h_i = 0` | `D - 1` |
| flat | intersection of a subset of the `H_i` | `D - |independent subset|` |
| cell (tope) | a connected component of the complement | `D` |
| face | a cell of the induced arrangement on a flat | any |
| sign vector | `s(x) = (sign h_1(x), ..., sign h_n(x))` | labels the cell |

The number of full-dimensional cells of an arrangement of `n` hyperplanes in
`R^D` is at most `sum_{i=0}^{D} C(n, i)`, attained in general position. For
`D = 2, n = 10` that is 56; for `D = 10, n = 100` it is astronomically large.
Any design that enumerates cells must therefore either keep `n` small, exploit
structure, or work implicitly.

### Soft cells

Replace `sign` by a gate. For a cell with target signs `eps in {-1, +1}^n`,

```
m_eps(x) = prod_{i=1}^{n} sigma( beta * eps_i * h_i(x) )
```

is a soft indicator that converges to the hard cell indicator as
`beta -> inf`. That is **temperature collapse**: one gate per hyperplane
hardened into a 0/1 step, distinct from the founding bias collapse.

Two properties worth stating precisely:

1. The full family `{ m_eps }` over all `2^n` sign vectors sums to `1` for every
   `beta` (expand the product), so it is a partition of unity over sign vectors,
   most of which correspond to *empty* cells. Only realizable sign vectors
   matter, and deciding realizability is itself the combinatorial problem.
2. The soft-to-hard gap for a *realizable* cell is controlled by the margin
   `min_i |h_i(x)|` and `beta`, exactly the structure that
   `certify_partition_gap` already bounds for the tree case.

### Log-domain form and the tropical link

Working with `log m_eps = sum_i log sigma(beta eps_i h_i)` turns products into
sums, and as `beta -> inf` the soft membership tends to
`-beta * sum_i max(0, -eps_i h_i)`, a max-plus expression. That is the bridge to
spec 01-08 and to `omnibias.struct`'s semiring driver: **the arrangement is a
tropical object in the hard limit and a log-domain object at finite `beta`.**

### Face lattice and the tope graph

Two cells are adjacent when their sign vectors differ in exactly one coordinate
*and* the shared face is non-empty. The resulting **tope graph** is the natural
graph for spec 02-02. Its vertices are cells, its edges are facets, and a walk
in it is a sequence of single-hyperplane crossings.

Honest complexity: constructing the full face lattice is exponential in `D`.
Practical routes are (a) small `D`, (b) restricting to a tree or laminar family,
(c) sampling cells by evaluating sign vectors at points and only materializing
the discovered subgraph, or (d) implicit access, never enumerating.

## 5. Worked example

Three lines in `R^2`:

```
h_1 = x        h_2 = y        h_3 = x + y - 1
```

Maximum cells for `n = 3, D = 2` is `C(3,0) + C(3,1) + C(3,2) = 1 + 3 + 3 = 7`,
and general position attains it. The realizable sign vectors are 7 of the 8
possible; `(-,-,+)` is empty because `x < 0` and `y < 0` force `x + y - 1 < 0`.

Soft membership at `x = (0.3, 0.3)` with `beta = 10`, target signs `(+, +, -)`:

```
h = (0.3, 0.3, -0.4)
sigma(10 * (+1)(0.3))  = sigma(3.0)  = 0.9526
sigma(10 * (+1)(0.3))  = sigma(3.0)  = 0.9526
sigma(10 * (-1)(-0.4)) = sigma(4.0)  = 0.9820
m = 0.9526 * 0.9526 * 0.9820 = 0.8912
```

The margin is `min |h_i| = 0.3`, so the deficit `1 - m` is bounded by roughly
`sum_i exp(-beta |h_i|) = 2 exp(-3) + exp(-4) = 0.1179`, consistent with the
measured `0.1088`. That bound is the certifiable statement; the exact value is
not.

## 6. Proposed API

Does not exist yet.

```python
# omnibias/partition/arrangement/_core.py
@dataclass(frozen=True)
class Arrangement:
    normals: FloatArray      # (n, D)
    offsets: FloatArray      # (n,)
    @property
    def n(self) -> int: ...
    @property
    def dim(self) -> int: ...

def sign_vector(arr: Arrangement, x: FloatArray) -> FloatArray:  # (..., n) in {-1,0,1}
    ...
def max_cells(n: int, dim: int) -> int: ...
def realized_cells(arr, samples: FloatArray) -> tuple[tuple[int, ...], ...]:
    """Sign vectors actually observed at `samples`. A lower bound, never a claim
    of completeness."""
def tope_graph(cells) -> tuple[tuple[int, int], ...]:
    """Adjacency over the discovered cells only."""
def soft_membership(arr, x, signs, *, beta) -> FloatArray: ...
def margin(arr, x) -> FloatArray: ...
def certify_cell_gap(arr, x, signs, *, beta) -> CellGapCertificate:
    """Sound bound on |soft - hard| from the margin. Mirrors
    `certify_partition_gap` for the arrangement case."""
```

Backends: `soft_membership` and `margin` get bit-identical torch and jax twins;
the combinatorial functions stay pure Python.

## 7. Practical use cases

1. **Feasible-set geometry for constrained learning.** The intersection of soft
   halfspaces is exactly an LP feasible region; specs 03-02 and 03-03 consume
   this representation.
2. **Region-wise models beyond a tree.** `RegionModels` in `omnibias-partition`
   currently combines `2**depth` tree regions; arrangement cells allow
   region-wise models whose regions are not forced into a balanced tree.
3. **Interpretable rule extraction.** A hardened sign vector is a readable
   conjunction of linear rules, and the margin says how confident the assignment
   is.
4. **Graph construction for spec 02-02.** The tope graph is the substrate for
   message passing on learned geometry.
5. **Sanity bounds.** `max_cells` gives an immediate honesty check: if a design
   claims to enumerate cells with `n = 200, D = 50`, the number tells you it
   cannot.

## 8. Acceptance gates

- **G1 combinatorial correctness.** For random arrangements with `n <= 12`,
  `D <= 4`, in general position, the number of cells discovered by dense
  sampling equals `max_cells(n, D)`; for planted degenerate arrangements it is
  strictly smaller and matches a brute-force reference.
- **G2 soundness of the gap.** `certify_cell_gap` upper-bounds the measured
  `|soft - hard|` on a dense grid **and** a random sample, with no violations,
  which is the standard the verified register already uses.
- **G3 tree agreement.** Restricted to a depth-`d` oblique tree, arrangement
  soft memberships reproduce `partition_weights` to `<= 4 ulp`.
- **G4 parity.** torch and jax bit-identical for the soft path.

## 9. Benchmark plan

- `benchmarks/arrangement_geometry.py`: cell counts versus theory, gap soundness
  sweep over `beta`, wall time versus `n` and `D` with the enumeration cutoff
  clearly recorded.
- Smoke JSON committed; full sweep under `$OMNIBIAS_SCRATCH/arrangement/`.

## 10. Honesty and scope

- `beta -> inf` here is **temperature collapse**: a soft indicator hardening
  into a 0/1 feasibility step. It is not the founding bias collapse, which is
  the `delta -> 0` coalescence of `K` biases yielding a smooth `sigma^(K-1)`.
- `realized_cells` returns what sampling found. It is a lower bound on the cell
  set and must never be presented as the complete face lattice.
- Cell counts grow combinatorially. This spec provides the vocabulary and the
  small-case tooling; it does not claim tractable arrangement learning at large
  `n` and `D`.
- The gap certificate is sound but not tight; a loose bound only widens the
  reported gap, which is the correct failure direction.

## 11. Open questions and risks

- **Realizability.** Deciding whether a sign vector is realizable is an LP
  feasibility question; wiring it to `omnibias.convex` is cheap for small `n`
  but must not be run inside a training loop.
- **Degeneracy.** Near-degenerate arrangements (three planes nearly concurrent)
  make cell discovery unstable. The margin should gate any downstream discrete
  decision.
- **Falsifier.** If every downstream consumer only ever needs the tree case, the
  general arrangement machinery is over-engineering and should stay a documented
  concept rather than shipped code.

## 12. Implementation checklist

- [ ] `packages/omnibias-partition/src/omnibias/partition/arrangement/_core.py`
- [ ] torch and jax twins for `soft_membership` and `margin`
- [ ] Cell-count test against the closed-form maximum and a brute-force reference
- [ ] Gap soundness test on dense grid plus random sample
- [ ] Agreement test versus `partition_weights` on tree-restricted arrangements
- [ ] `benchmarks/arrangement_geometry.py` plus smoke JSON
- [ ] Terminology note in any new relaxation module, and registration in
      `PENALTY_FILES` in `packages/omnibias-core/tests/test_concept_terminology.py`
- [ ] Docs page and nav entry
- [ ] Index row in `theory/README.md`
