---
name: omnibias-discrete
description: Maintain shared discrete problem, relaxation, decoding and optimality-gap infrastructure.
---

# Maintaining omnibias-discrete

Owned implementation: [source](../../../packages/omnibias-discrete/src/omnibias/discrete);
public surface: [API](../../../docs/api/discrete.md).
Load [shared numerical contracts](../../../docs/development/numerical-contracts.md)
when changing mathematics or tensor behavior.

`_core/` defines the problem seam and annealing primitives; backend directories
implement unrolled relaxations. `certify.py` combines lower bounds with feasible
objective values, `matroid.py` provides supported selection structure, and
`proposers.py` owns search proposals. Named application encodings belong in their
consumer repositories.

Follow the objective from its continuous relaxation to the binary decoder and
certificate. Preserve diagonal, constant-term and variable-order conventions.
For minimization, a feasible candidate gives an upper bound and a certified
relaxation gives a lower bound; test the inequality direction on exhaustive small
instances. Failure to close the gap is a valid result, not permission to stamp
an optimum.

Annealing state and random proposals should be explicit and reproducible. Decoder
changes need feasibility checks before reporting an objective, including empty
supports and ties. Optional certificate backends must be loaded only when the
selected route needs them. The soundness regression and brute-force oracle tests
are essential for bound changes; relaxation and shared-seam tests cover downstream
compatibility. Keep support-selection reports consistent for optional PINN adapters.

From the omnibias repository, start with:

```bash
uv run pytest packages/omnibias-discrete/tests/test_shared.py packages/omnibias-discrete/tests/test_decode.py packages/omnibias-discrete/tests/test_gap_soundness_regression.py -q
```
