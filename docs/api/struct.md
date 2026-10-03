# omnibias-struct

Differentiable structured computation.

`omnibias.struct` provides soft dynamic programming on chains, acyclic graphs
and shared semiring/hypergraph representations.

- `ChainTrellis`, `DAG`, `Hypergraph`: structural inputs.
- `MaxPlusSemiring`, `LogSemiring`, `CountingSemiring`: evaluation algebras.
- `omnibias.struct.torch` and `.jax`: differentiable implementations.
- `DPGapCertificate`, `certify_soft_dp`: soft-versus-hard value bounds.

Log-sum-exp temperature controls smoothing. The bound depends on the counted
finite alternatives and the inverse temperature; it does not assert that an
arbitrary learned decoder is correct. Consult the operation's signature for
tensor layout and its associated small-instance oracle.

Install this distribution with `pip install omnibias-struct`; select its
backend extras when needed. See [guarantees](../guarantees.md).

<!-- BEGIN GENERATED API INVENTORY -->

Version **0.1.0a1** · Python **>=3.10** · **3 - Alpha** · AGPL-3.0-or-later **or commercial**

<details markdown="1">
<summary>Public modules and top-level exports</summary>

[Source](https://github.com/derivon-ai/omnibias/tree/main/packages/omnibias-struct/src/omnibias/struct). Modules below are relative to `omnibias.struct`; underscored modules are internal.

`decision`, `decision.torch`, `decode`, `jax`, `jax.align`, `jax.attention`, `jax.distributions`, `jax.dtw`, `jax.eisner`, `jax.jets`, `jax.monotonic`, `jax.mtt`, `jax.parse`, `jax.plan`, `jax.select`, `jax.semiring`, `jax.soft_dp`, `jax.tropical`, `select`, `torch`, `torch.align`, `torch.attention`, `torch.curvature`, `torch.distributions`, `torch.dtw`, `torch.eisner`, `torch.jets`, `torch.monotonic`, `torch.mtt`, `torch.parse`, `torch.plan`, `torch.select`, `torch.semiring`, `torch.soft_dp`, `torch.tropical`, `verified`.

Exports from `omnibias.struct`:

`AcyclicMDP`, `AlignmentLattice`, `BinaryGrammar`, `CTCLattice`, `ChainTrellis`, `CountingSemiring`, `DAG`, `DPGapCertificate`, `DTWLattice`, `HyperEdge`, `Hypergraph`, `LSEBeta`, `LogSemiring`, `MaxPlusSemiring`, `SelectionCertificate`, `Semiring`, `TropicalGapCertificate`, `TropicalLinear`, `TropicalPathResult`, `TropicalSchedule`, `argmax_stability_margin`, `as_tropical_schedule`, `best_derivation`, `best_parse_tree`, `best_projective_tree`, `beta_for_confidence`, `brute_force_align`, `brute_force_arborescence`, `brute_force_cky`, `brute_force_ctc`, `brute_force_dtw`, `brute_force_eisner`, `brute_force_entropy`, `brute_force_gotoh`, `brute_force_kbest`, `brute_force_local_align`, `brute_force_mas`, `brute_force_optimal_return`, `brute_force_partition`, `brute_force_projective`, `brute_force_shortest_path`, `brute_force_soft_align`, `brute_force_soft_dtw`, `brute_force_soft_gotoh`, `brute_force_soft_local_align`, `brute_force_soft_mas`, `brute_force_value`, `brute_force_viterbi`, `build_chart`, `build_gotoh_dag`, `build_local_dag`, `certify_argmax`, `certify_soft_dp`, `certify_tropical_gap`, `count_alignments`, `count_arborescences`, `count_derivations`, `count_parse_trees`, `count_paths`, `count_projective_trees`, `ctc_best`, `ctc_best_alignment`, `derivation_weight`, `dual_subdivision`, `eisner_hypergraph`, `enumerate_derivations`, `from_dag`, `hard_align`, `hard_cky`, `hard_dtw`, `hard_eisner`, `hard_gotoh`, `hard_local_align`, `hard_mas`, `hard_matrix_tree`, `hard_value`, `hard_value_iteration`, `homotopy_gap_bound`, `iter_arborescences`, `iter_projective_trees`, `kbest_derivations`, `log_num_paths`, `logsumexp_gap_bound`, `mass_concentration_bound`, `matrix_tree_marginals`, `matrix_tree_partition`, `max_arborescence`, `newton_polytope`, `path_follow`, `relaxed_value`, `relaxed_weights`, `sample_derivations`, `seal_selection_certificate`, `semiring_value`, `shortest_path`, `soft_cky`, `soft_eisner`, `soft_value`, `stepwise_gap_bound`, `surrounding_tropical`, `tropical_anneal_descent`, `tropical_value`, `viterbi`.

</details>

<!-- END GENERATED API INVENTORY -->
