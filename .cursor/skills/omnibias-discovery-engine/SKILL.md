---
name: omnibias-discovery-engine
description: Add a discoverable FiniteFamily or ConditionHypothesis — statement, exact check, catalog mode, exhaustion, honesty keys, and a CI budget. Use when wiring a new search, condition-language sort, Observation binder, or replay kind through run_discovery, select_class, or register_catalog.
---

# Add a discoverable family

The discovery engine is `omnibias.core.proof.discovery` plus
`omnibias.core.proof.catalog`. `ProofMachine` adjudicates.

## Why nested AD fails

A float SVD or a residual minimum is not an exact check over Q. Generic
search notebooks plant a known witness and call it discovery. Without a
finite `Statement`, a typed `ConditionHypothesis`, and a CI budget,
searches are unreproducible and uncatalogued.

## What only this tower unlocks

A named family, an exact `Q` check, catalog modes (`exact_search`,
`exact_replay`, `enclosure`, `empirical`), exhaustion that reports
`search_incomplete` at `budget == 0`, and honesty keys that record what the
search actually earned. Soft residuals stay in `empirical` / `enclosure`
modes. Uniqueness is span-scoped.

## Use

1. Write a `Statement`: finite `obligation`, named `parent`, `parent_status`,
   `existential` (default True).
2. Implement `FiniteFamily`: `complete`, optional `cardinality()`, `origin`,
   `neighbors`, `score`, exact `check` → `ExactCheck` with an `honesty` dict.
3. `check` is exact Q / a finite predicate. Lie symmetry: float SVD proposes,
   then `exact_symmetry_report` snaps with `as_fraction` and `rank_collapse`.
4. `register_catalog(CatalogEntry(...), factory)` from the owning package at
   import. Core does not import the package.
5. Pick a mode. Replay stays `verify_*`.
6. CI proposer is `score_guided`. Optional proposers are instances
   (`AnnealDescentProposer()`).
7. Cheap in CI. `budget == 0` → `search_incomplete`.

Honesty keys on a family record earned claims (Jacobian n=2 via
`escalate_n2_result`, Keller sweep membership, H* separator, Erdős template
predicates). Pack tables: `omnibias.symbolic.ingest`.

## Extend

Compose with `omnibias-holonomic`, `omnibias-combinatorics`,
`omnibias-symbolic`, `omnibias-frontier`. Tests live next to the family.

```bash
python -m pytest packages/omnibias-core/tests -q -k discovery
```

## Next invention

A new FiniteFamily whose exact check is over Q, whose catalog mode is honest,
and whose first CI hit is a real witness rather than a planted origin.

## Further references

- `docs/api/discovery.md`
- `docs/cookbook/discovery-loop.md`
