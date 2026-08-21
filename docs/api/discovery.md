# Discovery engine

`omnibias.core.proof.discovery` is the proposer layer on
[`ProofMachine`](../cookbook/proof-machine.md). The checker is exact. A
proposer only emits candidates. Cookbook:
[Finite discovery engine](../cookbook/discovery-loop.md).

## Statement, family, loop

::: omnibias.core.proof.discovery
    options:
      show_root_heading: false
      heading_level: 3

## Catalog

Core never imports holonomic / combinatorics / symbolic / pinn. Owning
packages call `register_catalog` at import time.

| Mode | Meaning |
| --- | --- |
| `exact_search` | `run_discovery` + exact `check` |
| `exact_replay` | existing `verify_*` |
| `enclosure` | certified residual / SOS / gauge payload |
| `empirical` | float residual or SINDy; not `ExactCheck` |

Core proposers: `score_guided`, `coordinate_newton`, `onehot_anneal`.
Optional: `omnibias.discrete.proposers.AnnealDescentProposer` (not a
`get_proposer` name).

::: omnibias.core.proof.catalog
    options:
      show_root_heading: false
      heading_level: 3

## Condition language

Candidates may be typed conditions (`ConditionHypothesis`) in a finite
`GrammarSpec`. Typed constructors grow the grammar. `KindMetaFamily`
tries registered sorts. An incomplete grammar miss is
`search_incomplete` — never “no condition exists.”

::: omnibias.core.proof.condition
    options:
      show_root_heading: false
      heading_level: 3

## Shared observation / class loop

`Observation` is the one JSON-able payload every sort binder views.
A tag is optional. `sample_x` and `poly_constraints` are extra views.
`bind_sorts` returns `None` for a sort that does not apply.
`select_class` collects certified hits, optionally grows one
constructor step on an exact miss, and ranks them (`exact` ≻
`enclosure` ≻ `empirical`). `ClassMemory` (optional JSON) /
`FrequencyGate` only propose an order. Soft RMSE never outranks an
exact identity. The driver never marks a grammar complete.
`omnibias.symbolic.ingest` packs raw tables.
`omnibias.symbolic.classloop.discover_observation` loads owning
packages and runs the bilevel loop.

::: omnibias.core.proof.observe
    options:
      show_root_heading: false
      heading_level: 3

## Exact lift

`residual_identically_zero` / `integer_null_space` adjudicate over
`Fraction`. A float residual is not an `ExactCheck`.

::: omnibias.core.proof.lift
    options:
      show_root_heading: false
      heading_level: 3

## Ingest and stack entry

`omnibias.symbolic.ingest` packs raw tables. A tag is never required.
`discover_observation` imports owning packages and runs the bilevel
loop. PINN training stays opt-in.

::: omnibias.symbolic.ingest
    options:
      show_root_heading: false
      heading_level: 3

::: omnibias.symbolic.classloop
    options:
      show_root_heading: false
      heading_level: 3
