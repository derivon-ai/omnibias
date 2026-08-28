# 09-30 Inequality engine

## 1. Thesis and status

A **front door** for systems of inequalities: one `Conjecture` kind
dispatches to linear / polynomial / Boolean / finite-CSP backends.
The optimizer **proposes**; an exact `Q` / GF(2) / enclosure /
tiny complete enum **proves**. Soft RMSE is never an `ExactCheck`.

- **Status**: shipped (G1–G4 CI; G5 leftover-recorded, not in CI
  `all_passed`; export, not a new LP/CSP algorithm)
- **Depends on**: 03-02, 03-03, 09-01, 01-14
- **Blocks**: none

### Operator card

- **Benefit.** One payload and one `ProofMachine` kind instead of
  four disconnected solvers.
- **How it works.** `propose → rationalize → check`. Missing
  backend or an inconclusive SDP / IPM / anneal is `BLOCKED`.
- **Strength.** Locked 12-instance catalog agrees with named
  oracles. Not a complexity-class win.
- **When to use.** Adjudicating a named finite inequality system.
- **When not.** As a CAD / SMT / Gurobi replacement, a P vs NP
  claim, or a general theorem prover.
- **Accuracy floor.** Exact `Q` / GF(2) / SOS LDLᵀ / tiny enum.
  Float infeasible is not empty.

## 2. Where it lands

`omnibias.core.proof.inequality` (protocol + loader) plus thin
adapters in `omnibias.convex.inequality`, `omnibias.sos.inequality`,
`omnibias.boolean.inequality`, and `omnibias.discrete.csp.inequality`.
No new package. Core never imports those backends; they register at
import. Not a seventh `OperatorBlock` role.

## 3. Prior art in omnibias

- Spec 03-02 — arrangement LP front end; not a new LP algorithm
- Spec 03-03 — finite-domain CSP; not a complete solver
- `omnibias.sos.proofmachine` — `sos_global_nonneg` /
  `sos_nonneg_on_set`; SDP proposes, interval LDLᵀ proves
- `omnibias.boolean` — `solve_system` / `is_satisfiable`
- `omnibias.core.proof.catalog` / `load_discovery_stack` — import-time
  registration
- `Observation.poly_terms` / `poly_constraints` — existing monomial
  encoding

**Confirmed gap.** No shared `inequality_system` kind, no
propose → rationalize → check loop, and no locked cross-sort catalog.
09-30 is **not** 03-02, **not** 03-03, and **not** the existing
`sos_*` kinds.

## 4. Mathematics

An inequality system is a pair `(sort, existential, data)`.

- Existential + exact witness → `PROVED`
- Existential + complete tiny miss (Boolean `consistent=False`,
  CSP `brute_force_sat` empty, linear vertex-enum empty under
  `VERTEX_ENUM_MAX_*`) → `DISPROVED`
- Universal + SOS / Positivstellensatz seal → `PROVED`
- Universal + rational point with `p(x) < 0` → `DISPROVED`
- Else → `BLOCKED`

Temperature collapse (`beta -> inf`) may appear in a linear or CSP
**proposer**. The check is not temperature collapse. Founding bias
collapse is not in play. Enclosure Collapse may appear in the SOS
LDLᵀ path: `width -> 0` of a sound enclosure yields a point plus a proof.
There is **no** Farkas certificate: `InfeasibleProblemError` is `BLOCKED`.

## 5. Worked example

Unit box `0 <= x <= 1`: `A = [[1], [-1]]`, `b = [1, 0]`. Soft
membership proposes a point; rationalize onto `Q`; slacks are
nonnegative. Verdict `PROVED`. Asserting `p_equals_np_claim` is
`BLOCKED`.

`x^2 + y^2` global nonnegativity: SOS documents the boundary Gram
as inconclusive. Verdict `BLOCKED`, never a false proof.

## 6. Proposed API

```python
# omnibias.core.proof.inequality — implemented
INEQUALITY_KIND = "inequality_system"

@dataclass(frozen=True)
class InequalitySystem:
    sort: Literal["linear", "polynomial", "boolean", "csp"]
    existential: bool
    data: Mapping[str, object]
    name: str = ""

def load_inequality_stack() -> tuple[str, ...]
def build_inequality_machine() -> ProofMachine
def solve_inequality(system: InequalitySystem) -> Verdict
```

Pure Python in core. Adapters may use numpy. No default-dtype issue.

## 7. Practical use cases

1. **Linear SAT** of a tiny box (worked example).
2. **Boolean UNSAT** of `x ∧ ¬x` on the 1-bit cube.
3. **SOS plant** `p = 1` globally nonnegative.
4. **CSP SAT** with a witnessed one-hot vertex.
5. **Not** deciding arbitrary real-algebraic sentences.

## 8. Acceptance gates

- **G1 dispatch.** One kind, four sorts; missing sort/backend →
  `BLOCKED`.
- **G2 catalog.** 12/12 locked instances match the published
  table; replay agrees.
- **G3 honesty.** Forged `p_equals_np_claim` / `new_lp_algorithm_claim`
  / unsupported `complete_solver` → `BLOCKED`.
- **G4 pipeline.** At least one linear and one polynomial instance
  record propose → rationalize → check (not a planted witness).
  Boolean may skip propose.
- **G5 leftover-recorded.** Enclosure width vs naive interval on
  one locked polynomial. Not in CI `all_passed`. Not complexity SOTA.

## 9. Benchmark plan

- `benchmarks/inequality_facade.py`
- Smoke: `docs/benchmarks/inequality_facade_smoke.json`
- `--full`: `$OMNIBIAS_SCRATCH/inventions/inequality_facade/`

## 10. Honesty and scope

- Not a complete solver, not P vs NP, not a new LP algorithm.
- Float infeasible ≠ empty. No Farkas / Motzkin certificate.
- `theorem_prover_verified` is unforged (v1 stays false).
- Finite rational Lean only. Not NS, YM, RH, Jacobian `n=2`, or
  CCF stretch.
- Temperature collapse is the proposer axis only.

## 11. Open questions and risks

- **High-D linear emptiness** stays `BLOCKED` until a Farkas
  primitive exists.
- **SOS degree explosion** stays inconclusive.
- **Falsifier.** A float residual sealed as `ExactCheck`, or a
  `DISPROVED` from `InfeasibleProblemError`.

## 12. Implementation checklist

- [x] `omnibias.core.proof.inequality`
- [x] Four owning-package adapters
- [x] Locked catalog + facade tests
- [x] Smoke JSON with a `gates` block
- [x] Docs / mkdocs / AGENTS.md
- [x] `__all__` updates
- [x] Index row in `theory/README.md`

---

## Repo invariants this spec must respect

Check these before submitting an implementation.

- **Pure core**: no torch, jax, tensorflow or keras imports from
  `omnibias.core`. Pure-Python math lives there and every backend imports it.
- **Bit-identical twins**: torch and jax implementations must agree exactly.
  Polynomial coefficients come from `omnibias.core.polynomials`; never fork them
  per backend.
- **Default dtype**: use the framework default
  (`torch.get_default_dtype()` / `keras.config.floatx()`), never a hardcoded
  `float32`.
- **Vendor-neutral language**: no scheduler commands, vendor names, internal
  hostnames, usernames, or absolute local paths in tracked files. Artifacts go
  to `$OMNIBIAS_SCRATCH`, defaulting to a repo-relative `artifacts/`.
  `packages/omnibias-core/tests/test_no_leakage.py` enforces this across the
  whole readable surface.
- **Terminology**: distinguish the founding bias collapse (`delta -> 0`, yields a
  derivative) from temperature collapse (`beta -> inf`, yields a 0/1 step). The
  older penalty phrasings are retired and guarded by `tests/test_terminology.py`.
  Every new `**/relaxation.py` must carry the cross-reference note and be listed
  in `PENALTY_FILES` in
  `packages/omnibias-core/tests/test_concept_terminology.py`.
- **Executable docs**: if any part of this lands in `docs/` or a package README,
  every fenced Python block is executed by `tests/test_docs_snippets.py`. Verify
  calls against real signatures; opting out of a block needs a directive with a
  stated reason.
- **Earned flags**: `theorem_prover_verified` is set only by a genuine
  `lake build` pass of the Mathlib-free kernel, and `mathlib_verified` only by a
  genuine pass of the Mathlib-backed project. Asserting either without a pass
  blocks the verdict.
- **Typing tier**: the strict CI gate covers `core`, `torch`, `jax` and
  `ferminet`. Newly authored modules should be written strict-clean regardless
  of tier; curated beta modules can be added to
  `scripts/mypy_strict_allowlist.txt` once
  `mypy --strict --follow-imports=silent <file>` is clean.
