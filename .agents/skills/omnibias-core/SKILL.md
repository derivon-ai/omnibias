---
name: omnibias-core
description: Maintain pure-Python activation coefficients, Taylor combinatorics, verified numerics and certificate machinery in omnibias-core.
---

# Maintaining omnibias-core

Owned implementation: [source](../../../packages/omnibias-core/src/omnibias/core);
public surface: [API](../../../docs/api/core.md).
Load [shared numerical contracts](../../../docs/development/numerical-contracts.md)
when changing mathematics or tensor behavior.

Start with the smallest mathematical definition that owns the change: `polynomials.py`
for activation coefficients, `bell.py` for composition, `multi_index.py` for mixed
indices, `spec.py` for activation metadata. Consumers and tensor backends must
reuse those definitions. Keep exact integer or rational calculations exact until
the documented numerical boundary; do not replace a recurrence with fitted data.

`verified/` owns enclosure arithmetic and its assumptions. `proof/` serializes
and checks finite obligations; the Lean kernel is a separate executable checker.
Trace a proposed verification change through all three layers before altering a
verdict. Preserve refusal behavior for invalid domains, inconclusive bounds and
missing checkers. Core's dependency tests must continue to work without installed
Torch, JAX or Keras.

`planner.py`, `plan_algorithms.py` and `plan_calibration.py` own the operator planner shared by both
back ends; `precision.py` and `conditioning.py` own the policies it applies. A change to what `Plan.to_config` writes
needs a `PLAN_FORMAT` bump, a `PLAN_MIGRATIONS` step and the frozen `tests/data/plan_format_1.json` kept loading.
`conformance.py` and its `fixtures/` are the public correctness contract of any engine; `deprecation.py` holds the
schedule of promoted names. Calibration files under `calibration/` ship only for a device class whose pre-registered
dispatch gate passed. See [stability](../../../docs/stability.md).
`engines.py` is the process-global registry behind `register_engine`: an engine is generic, never serves a structured request and is
chosen by predicted cost or `force=`. See [interop](../../../docs/interop.md).

The planner is training-aware: `OperatorRequest.differentiate` (`none`, `params`, `params2`), `accuracy`, `resolve_planning`
and `expected_uses="auto"` are in `planner.py`, `Work.training(**backward)` and `second_derivative()` build the objective
from a program's forward work, and `plan_calibration._predict` must stay the same formula as `CostModel.predict_work`.
A work model counts what the executor builds, including trace and compile size: `forward_polylaplacian` unrolls
`sum_{s<=p} C(D, s)` support jets and is refused above `POLYLAPLACIAN_MAX_SUPPORTS`; `taylor_closed` serves `Delta` and
`Delta^2` only, the latter up to `TAYLOR_CLOSED_MAX_CHANNELS` (its streams stack `4 m^2 + 1` channels; 16,385 compile, 262,145 held 98 GiB of
host memory for 40 minutes, nothing between was measured); `fused_streams` is forced-only where a measurement says it loses and refuses what its shared-memory bound
cannot hold. `polarization.py` holds the mixed-partial plans and `demand.py` the demand vocabulary (`QuantitySpec`; the
back ends' `experimental/demand.py` add `streams_quantities` and `streams_residual`). A candidate that cannot be timed (its compile fails or runs out of memory) is a failed measurement: record it in `executor_failures` and
rank it last, never raise it out of `find`. A change to `plan_algorithms.py` or `plan_calibration.py` after a registered measurement round is an
amendment of that round (`omnibias_experiments/dispatch_gate_*/PREREGISTRATION.md`), not a silent edit.

`polarization.py` plans are sparse and memoized (`PolarizationPlan` keeps `rows/cols/slots/values`; `dense_is_cheap` and
`dense_floats()` serve small ones; the `Fraction` tables materialize only on request): a consumer contracts the entries that exist
(gather, then segment-sum), never a dense `(R, K, n)` table, and a change of its summation order is oracle-gated, not bit-identical.
`index_position_view` returns the shared read-only table (`index_position` still returns a fresh dict; copy a view before mutating it) and
`polylaplacian_support_batches` groups supports of one size, which share their terms. `find` is bounded: the leader runs first, followers
stop after one run slower than `FIND_CUT_FACTOR` times the best (only when the timer accepts `deadline`), a follower whose predicted peak
exceeds `host_memory_limit()` is skipped (`find_memory_limit`), a failed candidate is never cached, finished candidates are journaled in
`plans.json.journal` (`PlanCache.journal_*`) so an interrupted `find` resumes, and `measurement_context` and the bucketed `expected_uses`
are part of the cache key. None of that changes `PLAN_FORMAT`. Cost-model changes belong in the dated amendments of the dispatch gate
(`ROUND3.sha256` hashes `planner.py` and `plan_algorithms.py`; regenerate it with the amendment).
Performance edits are gated against an immutable snapshot with the paired A/B harness
(`omnibias_experiments/perf_audit_20261026/PROTOCOL.md`): bit-identical where the change claims to be a refactor, oracle-gated where
a summation order changes, never slower (paired median at most 1.02) and never larger in temporaries plus constants.

For coefficient work, exercise algebraic identities and independent values across
orders, including zero and negative input orders. For enclosure work, include
endpoint, degeneracy and cancellation cases that can expose unsound rounding.
Use the existing certificate audit and package-boundary tests when touching
serialization, imports or public exports.

From the omnibias repository, start with:

```bash
uv run pytest packages/omnibias-core/tests/test_polynomials_invariants.py packages/omnibias-core/tests/test_bell.py packages/omnibias-core/tests/test_multi_index.py -q
```
