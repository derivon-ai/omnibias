# 02-02 Arrangement graph network (Face-Net / Tope-Net)

## 1. Thesis and status

Turn a learned hyperplane arrangement into a graph — cells as nodes, shared
facets as edges — run message passing on it, then decode a discrete answer and
**certify the gap**, giving a graph neural network whose graph is learned
geometry rather than given data.

- **Status**: concept
- **Depends on**: 01-03, 01-08
- **Blocks**: 03-02, 03-09, 05-02

## 2. Where it lands

`packages/omnibias-graph/src/omnibias/graph/arrangement/` — the differentiable
graph package is the right home; the arrangement combinatorics come from spec
01-03 in `omnibias-partition`. No new package.

## 3. Prior art in omnibias

- `omnibias-graph` — differentiable spectral graph operators (Laplacians,
  spectral embedding, heat kernel) and combinatorial relaxations
  (Gumbel-Sinkhorn, SoftSort, soft top-k).
- `omnibias.struct` — the semiring / hypergraph driver, soft dynamic programming
  (Viterbi, shortest path, CTC), and `certify_soft_dp`, the sealed soft-to-hard
  gap certificate; plus `omnibias.struct.decode` for the sealed certified
  decoding chain and DAG.
- `omnibias.routing` — `certify_tour_gap`, the TSP relaxation with a
  Neumaier-Shcherbina LP gap certificate and a 2-opt decoder.
- `omnibias.partition` — `partition_weights`, `certify_partition_gap`,
  `RegionModels`.

**Confirmed gap.** All the graph machinery consumes a graph that is *given*.
Nothing constructs a graph from a learned arrangement, and nothing does message
passing over cells and facets.

## 4. Mathematics

### The graph

From spec 01-03: cells are sign vectors, and two cells are adjacent when their
sign vectors differ in exactly one coordinate and the shared face is non-empty.
Define

```
V = discovered realizable sign vectors
E = { (u, v) : Hamming(u, v) = 1, shared face non-empty }
```

Node features: cell centroid (or a representative point), cell soft mass, and
the margin. Edge features: the index of the crossed hyperplane, and the
distance-to-facet.

Both node and edge features are **differentiable in the arrangement parameters**
because they are built from `h_i(x) = w_i . x + b_i` and soft memberships, and
those have closed-form derivatives from the tower.

### Message passing

Standard, with one twist. A message across edge `(u, v)` crossing hyperplane `i`
carries the *transverse* information at that facet: the value and the jump. Since
crossing hyperplane `i` is exactly a change of sign in `h_i`, the natural edge
function is a function of `h_i` evaluated at the facet, and its derivatives are
the transverse derivatives that the OMBU tower supplies in closed form.

So the message function is not an arbitrary MLP; it is an OMBU multi-pack
evaluated at the facet. That is the architectural content: **the graph's edges
are hyperplanes, so the edge model should be a transverse operator**.

### Decode and certify

The output is typically discrete: a labelling of cells, a path through the tope
graph, a selection of facets. The pipeline is the one the repo already proves
works:

```
relax  ->  message passing at finite beta
decode ->  hard assignment / path
certify -> soft optimum plus log(N)/beta sandwiches the hard optimum
```

with the sandwich from `logsumexp_gap_bound` (spec 01-08), and the pattern
mirroring `certify_soft_dp` and `certify_tour_gap`.

### The routing reading, honestly

A walk in the tope graph is a sequence of hyperplane crossings, which makes
shortest-path and tour problems expressible. This is worth saying precisely
because it is the place where overclaiming is easiest:

- Expressing a tour on a learned arrangement graph is a **modelling** statement.
- The decoded tour comes with a gap certificate from `certify_tour_gap`, which
  is a genuine bound on that instance.
- None of this says anything about worst-case complexity. A differentiable
  heuristic plus an instance-wise gap is exactly what `omnibias.routing` and
  `omnibias.nphard` already provide, and their honesty framing applies verbatim.

## 5. Worked example

Four lines in `R^2`, forming a bounded quadrilateral cell:

```
h_1 =  x + 1        h_2 = -x + 1       h_3 =  y + 1       h_4 = -y + 1
```

All four positive is the square `(-1,1)^2`. Maximum cells for `n = 4, D = 2` is
`1 + 4 + 6 = 11`, and this arrangement (two pairs of parallel lines, so not in
general position) realizes `9`: the 3x3 grid of regions.

Graph: 9 nodes. The centre cell has 4 neighbours; edge cells have 3; corner
cells have 2. Total edges `= (4 + 4*3 + 4*2) / 2 = 12`.

Node features at the centre cell, sampled at the centroid `(0, 0)` with
`beta = 5`:

```
h = (1, 1, 1, 1),   all target signs +
soft mass = sigma(5)^4 = 0.9933071^4 = 0.9735495
margin    = 1.0
```

At a corner cell, centroid `(2, 2)`, target signs `(+, -, +, -)`:

```
h = (3, -1, 3, -1)
soft mass = sigma(15) sigma(5) sigma(15) sigma(5) = 0.9999997^2 * 0.9933071^2
          = 0.9866583
margin    = 1.0
```

Edge feature between centre and right-middle cell: the crossed hyperplane is
`h_2`, and the facet is the segment `x = 1, |y| < 1`. Evaluating a transverse
order-1 template there gives the jump signature the message carries.

A concrete task: label cells by whether a target function is positive inside
them. Message passing over 12 edges converges in 3 rounds on this graph, and the
decoded labelling comes with the `log(9)/beta = 0.4394` sandwich at `beta = 5`,
tightening to `0.0439` at `beta = 50`. The bound is loose here because the cells
are well separated; that is the expected behaviour.

## 6. Proposed API

Does not exist yet.

```python
# omnibias/graph/arrangement/__init__.py
@dataclass(frozen=True)
class ArrangementGraph:
    cells: tuple[tuple[int, ...], ...]      # sign vectors
    edges: tuple[tuple[int, int, int], ...] # (u, v, crossed hyperplane index)
    representatives: FloatArray             # one point per cell

def build_arrangement_graph(arr: Arrangement, samples: FloatArray) -> ArrangementGraph:
    """Discovery by sampling. Returns what was found; never claims completeness."""

class FaceNet(nn.Module):            # torch and jax twins
    def __init__(
        self, arrangement_dim: int, hidden: int, rounds: int, *,
        edge_template: MultiPackSpec | OpName = "grad",
        beta: float = 5.0, dtype=None,
    ) -> None: ...
    def forward(self, graph: ArrangementGraph, x: Tensor) -> Tensor: ...

def certify_facenet_gap(logits, *, beta, n_terms) -> GapCertificate:
    """Reuses `logsumexp_gap_bound`; does not fork the constant."""
```

## 7. Practical use cases

1. **Region-wise modelling with learned regions**, going beyond the balanced
   tree that `RegionModels` currently combines.
2. **Piecewise-linear function learning with structure**, where the graph makes
   the piece adjacency explicit and message passing enforces consistency across
   facets.
3. **Combinatorial layers on learned geometry**: assignment, matching or path
   problems where the graph itself is a function of parameters, so the
   *geometry* is trainable end to end.
4. **Interpretable clustering.** Cells are conjunctions of linear rules; the
   graph shows which rules are adjacent.
5. **A substrate for topology features** (spec 03-09): Euler characteristic and
   persistence read directly off this graph.

## 8. Acceptance gates

Baselines: a fixed `k`-NN graph neural network on the same points, and
`RegionModels` on a depth-matched oblique tree.

- **G1 graph correctness.** For `n <= 12`, `D <= 4`, the discovered graph
  matches a brute-force face-lattice computation exactly on node count, edge
  count and adjacency.
- **G2 gap soundness.** The reported sandwich contains the true hard optimum on
  every instance of a randomized suite, dense-grid and random-sample checked.
- **G3 task skill.** On a piecewise-function regression suite, Face-Net beats
  both baselines in relative `L2` at matched parameter count, with skill `> 0`,
  over five seeds.
- **G4 scaling honesty.** The benchmark records the `n`, `D` at which graph
  discovery becomes the bottleneck, and the artifact states the cutoff rather
  than quietly avoiding large cases.

## 9. Benchmark plan

- `benchmarks/arrangement_graph.py`: graph correctness, gap soundness, task
  skill, and a scaling table with an explicit cutoff.
- Smoke JSON committed; full under `$OMNIBIAS_SCRATCH/facenet/`.

## 10. Honesty and scope

- `beta -> inf` here is **temperature collapse**: soft cell memberships harden
  into 0/1 assignments. It is not the founding bias collapse (`delta -> 0`
  coalescing biases into `sigma^(K-1)`), which is what forms the transverse edge
  templates. Both appear in this architecture, in different places, and must be
  named separately wherever they are discussed.
- Graph discovery by sampling gives a **subgraph**. Any claim that depends on the
  full face lattice must either restrict to small cases with brute-force
  verification or be withheld.
- The routing and combinatorial readings are differentiable heuristics with
  instance-wise gap certificates. They carry no complexity-theoretic content
  whatsoever: nothing here bears on P versus NP.
- The certificate tier is a sound gap bound, not a theorem-prover tier.

## 11. Open questions and risks

- **Combinatorial explosion** is the central risk. Without structure, the graph
  is unusable beyond small `n` and `D`. The mitigation (sampling, laminar
  restriction) weakens the theory, and the spec should be judged on whether the
  restricted version still wins.
- **Graph changes during training.** As hyperplanes move, cells appear and
  disappear, so the graph is not fixed. Message passing over a changing graph is
  a genuine difficulty; a fixed-topology phase followed by a geometry-only phase
  may be necessary.
- **Falsifier.** If a `k`-NN graph network matches Face-Net everywhere, the
  learned-geometry graph is not adding information.

## 12. Implementation checklist

- [ ] `packages/omnibias-graph/src/omnibias/graph/arrangement/`
- [ ] torch and jax twins for `FaceNet` with a parity test
- [ ] Graph-correctness test versus brute force for small cases
- [ ] Gap soundness test reusing `logsumexp_gap_bound`
- [ ] Changing-topology test: assert the model detects and reports it
- [ ] Terminology note in any relaxation module plus `PENALTY_FILES` registration
- [ ] `benchmarks/arrangement_graph.py` plus smoke JSON
- [ ] Docs page and nav entry
- [ ] Index row in `theory/README.md`
