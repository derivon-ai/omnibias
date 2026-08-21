---
name: omnibias-dev-discovery-engine
description: Add a discoverable FiniteFamily or ConditionHypothesis to omnibias -- statement, exact check, catalog mode, exhaustion, honesty keys, and a CI budget. Use when wiring a new search, condition-language sort, Observation binder, or replay kind through run_discovery, select_class, or register_catalog. For contributors modifying omnibias itself, not for consumers using it.
---

# Add a discoverable family

The discovery engine is `omnibias.core.proof.discovery` plus
`omnibias.core.proof.catalog`. `ProofMachine` still adjudicates.

## Checklist

1. Write a `Statement`: finite `obligation`, named `parent`,
   `parent_status`, and `existential` (default True).
2. Implement `FiniteFamily`: `complete`, optional `cardinality()`,
   `origin`, `neighbors`, `score` (heuristic), exact `check` returning
   `ExactCheck` with an `honesty` dict.
3. `check` is exact `Q` / a finite predicate. Float residuals belong in
   catalog mode `empirical` or `enclosure`, never as a forged
   `ExactCheck`.
4. `register_catalog(CatalogEntry(...), factory)` from the owning
   package at import. Core must not import the package.
5. Pick a mode: `exact_search`, `exact_replay`, `enclosure`, `empirical`.
   Replay stays `verify_*`. Do not plant a known witness as a fake
   search origin.
6. CI default proposer is `score_guided`. Do not add names to
   `get_proposer`. Optional proposers are instances
   (`AnnealDescentProposer()`).
7. Budget: cheap in CI. `budget == 0` must be `search_incomplete`.
8. Honesty keys that stay False if present:
   `jacobian_conjecture_proof_claim`, `jacobian_n2_claim`,
   `dgg_congestion_theorem_refuted`, `erdos_183_claim`,
   `erdos_146_claim`, `erdos_180_claim`,
   `ten_proofs_formalization_claim`, `navier_stokes_proof_claim`,
   `no_condition_exists_claim`, `unnamed_condition_complete_claim`.
9. `discovered_by_omnibias` is True only on a non-replay hit.
10. Docs say “witness in this family / grammar.” Never “omnibias
    refuted Keller / Goemans / Erdős” and never “no condition exists.”
11. Condition-language sorts register with `register_condition_sort`.
    `GrammarSpec.complete` defaults False. An incomplete grammar miss
    is `search_incomplete`. Snap via `residual_identically_zero`; do
    not forge `ExactCheck` from a float residual.
12. Observation binders are `Observation → FiniteFamily | None`.
    `None` means the sort does not apply and forces an incomplete
    grammar. Binders read the table (`sample_x`, column names,
    `poly_constraints`), not planted tags. Pack via
    `omnibias.symbolic.ingest`. `select_class` collects and ranks
    certified hits (`exact` ≻ `enclosure` ≻ `empirical`) and may
    grow one constructor step on an exact miss. Gates / `ClassMemory`
    (optional JSON) only propose an order. They never write
    `ExactCheck`. The driver never sets `GrammarSpec.complete=True`.
    `discover_observation` is the stack entry point. Do not
    auto-register invented checkers. PINN training stays opt-in and
    out of default CI.

## Verdicts

- Existential + hit → `PROVED`
- Existential + exhausted complete miss → `BLOCKED` (`no witness in enumerated family`)
- Universal + hit → `DISPROVED`
- Universal + exhausted complete miss → `PROVED` of that finite universal
- Any non-exhausted miss → `BLOCKED` / `search_incomplete`

`Characterization.unique_in_family` is true only after an exhausted
complete walk with one solution. First-hit is not uniqueness.
