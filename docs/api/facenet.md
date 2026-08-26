# Face-Net (02-02)

Message passing on a **sampled subgraph** of the arrangement tope
graph, then a certified decode. Sampling is a lower bound, never a
complete face lattice. `beta -> inf` is temperature collapse, not
founding `delta -> 0`. The gap reuses `logsumexp_gap_bound`. Sound
gap, not P vs NP, not theorem-prover.

G1/G2 are CI-gated on `n<=12`. G3 vs a Face-Net GNN + `RegionModels` is
**leftover-recorded** unearned (leftover #28): a 0-hop cell-centroid
readout versus k-NN is measured, but named G3 needs message passing
and `RegionModels` at matched parameter count. Sampled
`build_arrangement_graph` wall vs `n`/`D` is **leftover-recorded**
(leftover #20; G1 tooling refuses `n>12` or `D>4`); the previous
untimed G4 / `g3_vs_knn` stubs are withdrawn. Cost and G3 are not in
CI `all_passed`. Status is **gated**, not shipped. See theory spec
02-02. Combinatorics come from [arrangement.md](arrangement.md).

## Graph layer

::: omnibias.graph.arrangement
    options:
      show_root_heading: false
      heading_level: 3
