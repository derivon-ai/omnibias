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
