# omnibias-discrete

Shared discrete optimization seam.

`omnibias.discrete` contains the common representation and relaxation machinery
used by external discrete-optimization consumers.

- `DiscreteProblem`, `DiscreteSolution`: problem and result contracts.
- `AnnealSchedule`: temperature progression.
- `omnibias.discrete.torch` and `.jax`: differentiable relaxation backends.
- `decode`, `round_relaxed`, `one_flip_descent`: obtain feasible candidates.
- `brute_force_min`: a small-problem reference oracle.
- `certify_gap`: combine a feasible value and a justified lower bound.

A rounded candidate or a small relaxation loss is not an optimality proof.
Report feasibility and the certified gap separately.

Install this distribution with `pip install omnibias-discrete`; select its
backend extras when needed. See [guarantees](../guarantees.md).

<!-- BEGIN GENERATED API INVENTORY -->

Version **0.1.0a2** · Python **>=3.10** · **3 - Alpha** · AGPL-3.0-or-later **or commercial**

<details markdown="1">
<summary>Public modules and top-level exports</summary>

[Source](https://github.com/derivon-ai/omnibias/tree/main/packages/omnibias-discrete/src/omnibias/discrete). Modules below are relative to `omnibias.discrete`; underscored modules are internal.

`certify`, `jax`, `jax.relaxation`, `matroid`, `proposers`, `torch`, `torch.relaxation`.

Exports from `omnibias.discrete`:

`AnnealDescentProposer`, `AnnealSchedule`, `DiscreteProblem`, `DiscreteSolution`, `GapCertificate`, `INIT_THETA_SCALE`, `TightenedGap`, `UnionFind`, `boolean_constraints`, `brute_force_min`, `certify_gap`, `decode`, `energy`, `flip_deltas`, `gershgorin_min_eig_lower`, `initial_theta`, `is_binary`, `is_forest`, `lasserre_lower_bound`, `mean_normalized_regret`, `negative_coeff_lower_bound`, `one_flip_descent`, `round_relaxed`, `spo_plus_subgradient`, `tighten_gap`.

</details>

<!-- END GENERATED API INVENTORY -->
