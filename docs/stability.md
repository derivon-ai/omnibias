# Stability: what is stable, how names are retired

The planned derivative operators began under `omnibias.<backend>.experimental`. The part a user calls — the planner front ends,
the plan format, the precision and conditioning policies and the conformance harness — is now promoted to stable modules.
The programs the planner chooses among stay experimental: they are reached through the planner, and each carries measured but
scoped evidence in [performance](performance.md). This page states what the promotion promises, and what it does not.

## The stable surface

| Module | Names |
|---|---|
| `omnibias.jax.operators` | `compile_operator`, `compile_residual`, `CompiledOperator`, `CompiledResidual`, `OperatorOutput`, `Residual`, `Quantities`, `ConditioningPolicy`, `PrecisionPolicy`, `Plan`, `PlanCache`, `PLAN_FORMAT`, `match_dense_chain`, `describe_layers`, `layers_of`, `bucket_size`, `enable_compilation_cache`, `StagedProgram`, `ExportedProgram`, `shard_points`, `distributed_residual_gram`, `distributed_residual_gauss_newton`, `distributed_residual_ntk_diagonal`, `conformance_engine` |
| `omnibias.torch.operators` | `compile_operator` (the planner front end, formerly `compile_planned_operator`), `compile_residual`, `CompiledOperator`, `CompiledResidual`, `OperatorOutput`, `ConditioningPolicy`, `PrecisionPolicy`, `Plan`, `PlanCache`, `PLAN_FORMAT`, the three `distributed_residual_*` functions, `conformance_engine` |
| `omnibias.core.planner` | `OperatorRequest`, `Descriptor`, `Plan`, `PlanCache`, `CostModel`, `select`, `Refusal`, `PLAN_FORMAT`, `PLAN_MIGRATIONS`, `migrate_plan_config`, `resolve_planning` |
| `omnibias.core.precision`, `omnibias.core.conditioning` | the policies the front ends accept |
| `omnibias.core.conformance` | the fixtures, `run_correctness`, `dispatch_regret`, `reference_engine` |
| `omnibias.core.deprecation` | the schedule of retired names |
| `omnibias.core.engines`, `register_engine` / `unregister_engine` in both `operators` modules | third-party engines as planner executors ([interop](interop.md)) |
| `omnibias.jax.folx_compat`, `omnibias.torch.deepxde_adapter`, `omnibias.torch.pina_adapter` | interfaces to third-party APIs; each is tested against the release named in [interop](interop.md), and a newer release is outside the promise until it is tested |

Stable means three things. Signatures and result structures follow the [deprecation policy](#deprecation-policy) below. A plan
written by one release is read by the next ([plan format](#the-plan-format)). And the conformance fixtures are public and
versioned, so a claim of correctness can be re-checked by anyone.

Stable does **not** mean fast. Which program the planner picks, and how fast it is, depends on the device, the shape and the
calibration; the evidence that it is right on a given device is the [dispatch gate](#the-dispatch-protocol), and where that gate
has not been run the cost model uses generic defaults and says so in `explain()`. Programs may be added, retired or reranked
within a release series; the *result* of a request does not change, only the program that computes it.

The request keywords added with the training-aware planner (`differentiate`, `accuracy`, `uses_hint`, `safe_find`,
`expected_uses="auto"`, `mode="auto"`, `compile_policy="auto"`) are optional, and their defaults keep the earlier behaviour: a call
that does not pass them plans a forward evaluation and returns the result it returned before; the program that computes it can
differ, because the registry gained programs (and `folx`, when installed, is now a candidate). The names are stable; the ranking
they produce is not (see above). `differentiate="params2"` ranks by an unvalidated model of the second derivative. The compiled training step
(`omnibias.jax.experimental.training_step`, `omnibias.torch.experimental.training_step`) and the second-order step modules
(`omnibias.jax.optim_pinn`, `omnibias.torch.optim_pinn`) are **not** part of the stable surface: they are new, their names and
signatures may change in a release series, and they carry no promise beyond their tests.

The two back ends do not offer identical behaviour. PyTorch has no `projected_streams` residual program (the planner refuses it
with that reason), and its hardware calibration is not measured; see [operator planner](operator-planner.md) and
[precision](precision.md).

## The plan format

`Plan.to_config()` is a JSON-serialisable record of the descriptor, the chosen algorithm, every candidate with its predicted
cost, every refusal and the library versions. It carries an integer `format` (`PLAN_FORMAT`, currently 2: format 2 added the `objective` and `uses_source` keys, and a
format-1 configuration is migrated by giving them the values `"forward"` and `"explicit"`).

- The integer changes when a key is removed or renamed, or a meaning changes. Adding an optional key does not change it.
- A configuration with an older format is read through `PLAN_MIGRATIONS`, one step per version, chained. If a link is missing,
  loading raises `ValueError` naming it.
- A configuration from a *newer* format is refused rather than guessed at.
- A `PlanCache` file written under another format, or for another calibration, never returns an entry: the key changes, so an
  upgrade re-plans instead of reusing a stale decision.
- `tests/data/plan_format_1.json` is a frozen sample that must keep loading (it now loads through the 1 to 2 migration). A change
  that breaks it needs a format bump and a migration, not an edit of the sample.

```python
from omnibias.core.planner import PLAN_FORMAT, PLAN_MIGRATIONS, migrate_plan_config

print("format", PLAN_FORMAT, "migrations", sorted(PLAN_MIGRATIONS))
assert migrate_plan_config({"format": PLAN_FORMAT, "algorithm": "projected"})["format"] == PLAN_FORMAT

for config in ({"format": PLAN_FORMAT + 1}, {"format": 0}, {"format": "1"}):
    try:
        migrate_plan_config(config)
    except ValueError as error:
        print("refused:", error)
```

## Deprecation policy

A name leaves `experimental` in three steps, recorded as data in `omnibias.core.deprecation`:

1. **Silent alias** from `since` (0.5.0). The old name and the new one resolve to the *same object*; nothing that worked changes.
2. **`DeprecationWarning`** from `warn_from` (0.6.0). Importing or reading the old name warns, names the replacement and the
   release that removes it.
3. **Removal** in `remove_in` (0.7.0): at least two minor releases after the stable name first shipped.

The schedule is compared with the installed version with the pre-release suffix ignored, so `0.6.0rc1` already follows the
`0.6.0` rules. `tests/test_deprecation_schedule.py` fails the build if an alias is past its removal release, if the stable
module no longer provides a name it promised, or if the order `since ≤ warn_from < remove_in` is broken.

Names that changed meaning, not only location, are listed here. In PyTorch, `experimental.compile_planned_operator` becomes
`operators.compile_operator`; `experimental.compile_operator` is a different function (it compiles an `OperatorSpec` with the
structural compiler and plans only when given an operator name) and is **not** promoted.

```python
from omnibias.core.deprecation import ALIASES
import omnibias.jax.operators  # registers its aliases
import omnibias.torch.operators  # registers its aliases

by_backend = {}
for alias in ALIASES.values():
    by_backend.setdefault(alias.old.split(".")[1], []).append(alias)
for backend, aliases in sorted(by_backend.items()):
    schedule = {(a.since, a.warn_from, a.remove_in) for a in aliases}
    print(backend, len(aliases), "aliases; since/warn/remove:", sorted(schedule))
```

Experimental code outside the promoted set has no such schedule. Anything left under `experimental` may change without notice,
as it says in each module's docstring.

## Conformance

`omnibias.core.conformance` replays public fixtures against **any** derivative engine, not only omnibias: a hand-written
function, another library or a reference implementation. An engine is a callable `engine(quantity, case) -> array` for the
Laplacian (`"lap"`), the divergence (`"divergence"`), `log|det(I + J)|` (`"logdet"`) or the solution of `(I + J) y = rhs`
(`"solve"`).

The fixtures hold exact float64 weights and points (hexadecimal floats) and references computed in 90-digit `mpmath`
arithmetic by central differences of the network itself, so no derivative formula in this package decides the expected value.
The reported error is `max |got − want| / max(1, |want|)` entrywise.

```python
import numpy as np

from omnibias.core import conformance as C

results = C.run_correctness(C.reference_engine, tolerance=1e-13)
print(C.summarize(results))


def sloppy(quantity, case):
    want = C.reference_engine(quantity, case)
    return want + 1e-6 * np.maximum(1.0, np.abs(want))  # an engine that is accurate to six digits


bad = C.run_correctness(sloppy, tolerance=1e-11)
print(len(C.summarize(bad)["failed"]), "of", len(bad), "case/quantity pairs fail at 1e-11")
```

Each back end supplies `conformance_engine()`, which plans every request with its `compile_operator` and so tests the
dispatching front end, not one hand-picked program. `tests/test_conformance.py` runs both. Conformance is correctness at the
fixture sizes in float64: it says nothing about speed, large inputs or float32.

## The dispatch protocol

A planner that is correct but picks slowly is a regression. `dispatch_regret(timings, chosen)` takes the steady-state time of
every program that could serve each case and the program the planner chose, and reports the ratio to the best one. The gate
registered before the measurement (`omnibias_experiments/dispatch_gate_20261022/PREREGISTRATION.md`) is:

- a calibration set and a held-out set of descriptors, disjoint, fixed before any timing;
- median regret at most 1.10 and worst regret at most 1.5 on the held-out set;
- the fallback (generic) program, where it is chosen, at least as fast as the strongest alternative within a lower95 of 0.95;
- the cost of calibrating and compiling amortised within 100 uses for at least 90 % of held-out cases;
- a multiprecision qualification of the programs the planner may choose;
- a calibration file ships for a device class only if the gate passes on it.

The outcome on the A100, including what failed, is in [operator planner](operator-planner.md#dispatch-regret). A third round is
registered for the current planner, in both the forward and the value-and-gradient mode, and has not been run, so the dispatch of the
current registry has no registered held-out result. A device class
without a shipped calibration file runs on the generic cost model, and `explain()["cost_model"]` says which one is in use (`"defaults"` for the generic one, `"defaults+cpu class prior"` and `"defaults+counted torch launches"` for the two device-class
priors that are set from the hardware or counted, not fitted).

## What stability does not cover

- It does not establish convergence of any PDE solve or the quality of any trained model.
- It does not make float32 or TF32 results equal to float64; [precision](precision.md) gives the measured errors and the refusals.
- It does not extend to the experimental programs, the benchmarks in `docs/benchmarks` or the consumer repositories.
- Hardware evidence is per measured device. A device that was not measured is unmeasured, not assumed to behave like its neighbour.
