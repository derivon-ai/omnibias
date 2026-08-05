# omnibias-struct

Certified **differentiable dynamic programming** on **PyTorch** and **JAX**
(bit-identical twins, float64): soft Viterbi, shortest-path, and CTC, plus soft-DTW,
sequence alignment (global, **local Smith-Waterman**, and **affine-gap Gotoh**), soft value
iteration / planning, monotonic alignment search (MAS), and structured attention -- all on
one soft-DP substrate, with higher-order and second-order derivatives *through* the
recursion, a verified interval path, and a sealed certified-decoding proof.

A **semiring / hypergraph driver** generalises the substrate: every DP is a semiring
reduction over the derivations of a weighted hypergraph, so the tree / grammar families
--- **CKY inside-outside parsing**, **Eisner projective dependency** parsing, and the
**matrix-tree** non-projective marginals --- and the **distribution operators** (path
entropy, exact sampling, exact k-best) all lift onto it with no bespoke recursion. The
driver reproduces the hand-written soft-DP layers bit-for-bit (the additive-safety proof),
so the new families are purely additive.

Exact hard DP is not differentiable (its `argmax` gradient is a.e. zero), so the sound
differentiable object is a **relaxation + a certified gap**, and two limits are kept
rigorously apart:

- **`beta -> inf` (feasibility / temperature).** The hard `max` combine becomes
  `lse_beta(a) = beta^-1 log sum_i exp(beta a_i)`; since `lse_beta >= max` and
  `lse_beta -> max`, the soft DP anneals to exact hard DP.
- **`delta -> 0` (the founding bias collapse tower): the exact differentiation
  engine.** Pairwise `lse_beta(a, b) = a + beta^-1 softplus(beta (b - a))`, and
  `softplus^(n) = sigma^(n-1)` is the closed-form omnibias tower; propagated through
  `compose_jet` it gives the exact log-sum-exp / softmax jets, whose first-order
  sensitivity is the softmax marginal.

The gap is closed-form: `V* <= V_beta <= V* + log(N)/beta` (`N` = exact path count),
self-checked against brute-force hard DP on small instances. It certifies the
*relaxation* error (the `beta -> inf` axis), **not** model correctness.

## Backend-agnostic core

::: omnibias.struct
    options:
      show_root_heading: false
      heading_level: 3
      members:
        - ChainTrellis
        - DAG
        - CTCLattice
        - DTWLattice
        - AlignmentLattice
        - AcyclicMDP
        - count_paths
        - count_alignments
        - log_num_paths
        - viterbi
        - shortest_path
        - ctc_best
        - ctc_best_alignment
        - hard_dtw
        - hard_align
        - hard_mas
        - hard_value_iteration
        - brute_force_viterbi
        - brute_force_shortest_path
        - brute_force_ctc
        - brute_force_partition
        - brute_force_dtw
        - brute_force_soft_dtw
        - brute_force_align
        - brute_force_soft_align
        - brute_force_mas
        - brute_force_soft_mas
        - brute_force_optimal_return
        - certify_soft_dp
        - DPGapCertificate
        - logsumexp_gap_bound
        - stepwise_gap_bound

## Semiring / hypergraph driver

The keystone: a topologically-ordered weighted `Hypergraph` (arity-0/1/2 `HyperEdge`s) plus a
`Semiring` protocol with the three canonical reductions -- `MaxPlusSemiring` (hard optimum,
`beta -> inf`), `LogSemiring` / `LSEBeta` (the soft relaxation), and `CountingSemiring` (the
exact derivation count `N`). `semiring_value` runs one topological sweep; `from_dag` lifts any
`DAG` onto the driver; `enumerate_derivations` / `brute_force_value` are the flat oracles. The
differentiable `semiring_value` / `semiring_marginals` live on the backends (below) and are
pinned bit-for-bit to `soft_viterbi` / `soft_shortest_path` / `soft_dtw` / `soft_align`.

::: omnibias.struct
    options:
      show_root_heading: false
      heading_level: 3
      members:
        - Hypergraph
        - HyperEdge
        - Semiring
        - MaxPlusSemiring
        - LogSemiring
        - LSEBeta
        - CountingSemiring
        - semiring_value
        - hard_value
        - soft_value
        - count_derivations
        - best_derivation
        - enumerate_derivations
        - derivation_weight
        - brute_force_value
        - from_dag

## Tree / grammar families (parse / Eisner / matrix-tree)

Backend-agnostic constructors + hard / brute-force / counting oracles for the three tree
families. **CKY** (`BinaryGrammar`, `build_chart`, `hard_cky`, `soft_cky`, `count_parse_trees`)
parses a CNF grammar; **Eisner** (`eisner_hypergraph`, `hard_eisner`, `count_projective_trees`)
is the `O(n^3)` projective-dependency span DP; both lift onto the driver, so the differentiable
`soft_inside` / `inside_outside` and `soft_eisner` / `eisner_marginals` come from the backends.
**Matrix-tree** (`matrix_tree_partition`, `matrix_tree_marginals`, `max_arborescence`,
`count_arborescences`) is the *exact* Kirchhoff-determinant partition for non-projective
dependency -- honestly **not** an `lse_beta` relaxation; its `beta -> inf` gap is taken against
the Chu-Liu/Edmonds maximum arborescence.

::: omnibias.struct
    options:
      show_root_heading: false
      heading_level: 3
      members:
        - BinaryGrammar
        - build_chart
        - hard_cky
        - best_parse_tree
        - soft_cky
        - count_parse_trees
        - brute_force_cky
        - eisner_hypergraph
        - hard_eisner
        - best_projective_tree
        - soft_eisner
        - count_projective_trees
        - brute_force_projective
        - brute_force_eisner
        - iter_projective_trees
        - matrix_tree_partition
        - matrix_tree_marginals
        - hard_matrix_tree
        - max_arborescence
        - count_arborescences
        - iter_arborescences
        - brute_force_arborescence

## Distribution operators

Exact quantities over the Gibbs path distribution `p_beta(D) ~ exp(beta * score(D))` on any
driver hypergraph. The pure-numpy oracles here (`brute_force_entropy`, `sample_derivations`,
`kbest_derivations`, `brute_force_kbest`) back the differentiable / exact backend ops
`path_entropy`, `sample_paths`, `gumbel_relaxed_sample`, `topk_paths`, and `topk_free_energy`.

::: omnibias.struct
    options:
      show_root_heading: false
      heading_level: 3
      members:
        - brute_force_entropy
        - sample_derivations
        - kbest_derivations
        - brute_force_kbest

## Local + affine-gap alignment

Smith-Waterman **local** alignment (free start/end via source/sink 0-edges) and Gotoh
**affine-gap** alignment (3-state match / gap-x / gap-y lattice), both lifted onto the shared
shortest-path substrate. Backend `soft_local_align` / `soft_gotoh` (+ `*_marginals`) are the
differentiable twins.

::: omnibias.struct
    options:
      show_root_heading: false
      heading_level: 3
      members:
        - build_local_dag
        - hard_local_align
        - brute_force_local_align
        - brute_force_soft_local_align
        - build_gotoh_dag
        - hard_gotoh
        - brute_force_gotoh
        - brute_force_soft_gotoh

## Soft-DP layers (torch)

Sequence models (`soft_viterbi`, `soft_shortest_path`, `soft_ctc`) with closed-form
forward-backward marginals (`*_marginals`, `soft_viterbi_transition_marginals`) and
batched (`*_batched`) twins; the application layers `soft_dtw`, `soft_align`,
`soft_local_align`, `soft_gotoh`, `soft_value_iteration`, `soft_mas`, and
`structured_attention`; the semiring driver (`semiring_value`, `semiring_marginals`) and the
tree / grammar families built on it (`soft_inside` / `inside_outside`, `soft_eisner` /
`eisner_marginals`, `soft_matrix_tree` / `matrix_tree_marginals`); the distribution
operators (`path_entropy`, `sample_paths`, `gumbel_relaxed_sample`, `topk_paths`,
`topk_free_energy`); the exact log-sum-exp / softmax jets (`logsumexp_beta`, `softmax_beta`,
`pairwise_lse_jet`); and the higher-order / second-order-through-the-DP surface
(`dp_value_jet`, `chain_lse_jet`, `hypergraph_lse_jet`, `cky_lse_jet`, `eisner_lse_jet`,
`chain_hessian`, `chain_value_and_hvp`, `chain_sharpness`). Batched (`*_batched`) twins now
cover `soft_align`, `soft_mas`, `soft_value_iteration`, `structured_attention`,
`soft_inside`, and `soft_eisner`.

::: omnibias.struct.torch
    options:
      show_root_heading: false
      heading_level: 3

## JAX twin

The JAX backend (`omnibias.struct.jax`) is the bit-identical twin of
`omnibias.struct.torch`; the soft-DP values, closed-form forward-backward marginals, and
the tower-built jets are checked against the torch backend and against `jax.grad` /
finite differences to `rtol=1e-9` in float64 (the second-order *curvature* helpers are
torch-only -- JAX users obtain the same Hessians via `dp_value_jet`).

## Verified interval soft-DP

Outward-rounded interval enclosures of the soft-DP value, built on
`omnibias.core.verified` (`exp_iv` / `ln_iv` / `softplus_iv`): the chain / DAG values plus
`dtw_value_iv`, `align_value_iv`, and `ctc_value_iv`, the certified forward-backward
`chain_marginals_iv`, and the interval Kirchhoff determinant `matrix_tree_partition_iv`.
Soundness is tested against a dense grid **and** a random sample. Scope is a local box around
the given point.

::: omnibias.struct.verified
    options:
      show_root_heading: false
      heading_level: 3

## Certified decoding

`certify_decoding` proves the winner-vs-runner-up margin stays `> 0` over an input
`epsilon`-ball (a deviation-tracking reduced min-plus interval DP), so the decoded path is
provably stable under the perturbation; `certify_decoding_dag` generalises the same
margin-sign proof from the chain to an arbitrary DAG / DTW shortest path. The verdict is
v1-sealed (`seal_decoding_certificate` / `check_decoding_certificate`,
`seal_dag_decoding_certificate` / `check_dag_decoding_certificate`) and optionally
Lean-checkable on the finite margin-sign fact.

::: omnibias.struct.decode
    options:
      show_root_heading: false
      heading_level: 3

## Certified selection (measure-mode collapse)

The `beta -> inf` collapse of a Gibbs law `p_beta(i) ~ exp(beta s_i)` over `N` logits onto a
Dirac at the mode (`argmax`). The contribution is not the softmax but a **sound, closed-form
certificate** of how far the collapse has gone: `certify_argmax` seals a `SelectionCertificate`
with three honest sub-claims over sorted logits `s_(1) >= s_(2) >= ...`, margin `m = s_(1) -
s_(2)` -- the value gap `max <= lse_beta <= max + log(N)/beta` (reusing `certify_soft_dp`), the
mode-mass lower bound `p_max >= 1/(1 + (N-1) e^{-beta m})` (checked against the exact `p_max`;
`beta_for_confidence` inverts it), and the optional `L^inf` argmax-stability radius `m/2` (stable
iff `m > 2 eps`). `seal_selection_certificate` emits a tamper-evident v1 certificate. This is the
*feasibility / temperature / measure* sense of collapse (the `beta -> inf` axis), **not** the
founding `delta -> 0` bias collapse -- the tower only differentiates `lse_beta`; do not conflate.

The backend twins (`omnibias.struct.{torch,jax}.select`) add the differentiable selection ops
and exact Gibbs moments: `soft_max_value` (`= lse_beta`), `soft_argmax` / `gibbs_mean`
(`= softmax`), `gibbs_covariance` (`= diag(p) - p p^T = hessian(lse_beta)/beta`),
`gibbs_cumulants_directional` (higher directional cumulants via the exact log-partition jet
`compose_jet`), and `certified_argmax(logits, beta, *, eps=None) -> (soft_output, certificate)`.

::: omnibias.struct.select
    options:
      show_root_heading: false
      heading_level: 3

## Differentiable decision layer

`omnibias.struct.decision` makes the certified selection above **embeddable** in
a network: `DecisionLayer` (torch `nn.Module`) has a relaxed
`softmax(beta·scores)` forward with an exact autograd backward, and `.certified`
returns the `soft_output` plus the sealed `SelectionCertificate`. The
backend-neutral helpers (`best_index`, `certified_decision`, `decision_regret`)
support a predict-then-optimize workflow: train through the relaxed decision on
**decision regret** and read off the sound `log(N)/beta` gap plus the `L^inf`
argmax-stability radius. This is the same feasibility / measure-mode `beta ->
inf` axis as `struct.select`, **not** the founding `delta -> 0` bias collapse.

::: omnibias.struct.decision
    options:
      show_root_heading: false
      heading_level: 3

::: omnibias.struct.decision.torch
    options:
      show_root_heading: false
      heading_level: 3

Status: Alpha (`0.1.0a1`).
