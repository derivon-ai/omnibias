# 06-01 Acceptance gates and benchmarks

## 1. Thesis and status

Every spec in this tree carries acceptance gates, and this file is the single
definition of what a gate *is* — so that fifty-four specs cannot each invent a
weaker standard, and so that a reader can tell a result from a demo by looking
at one JSON block.

- **Status**: designed
- **Depends on**: none
- **Blocks**: 06-02, 06-03, 07-01

The listed successors are the specs that build on this one directly. Every other
spec in the tree is bound by it too: each one's section 8 must conform to the
protocol defined here.

## 2. Where it lands

`benchmarks/_gates.py` — which already exists and already implements the
protocol. This spec adds the *extensions* the theory program needs
(certificate-coverage gates, parity gates, cost gates) to that module, not a new
one. Nothing about this earns a package.

## 3. Prior art in omnibias

This is the one spec in the tree where the prior art is nearly complete.

- `benchmarks/_gates.py` — the live protocol. Its docstring already states the
  three ordered questions, and it provides `require_reference_valid`,
  `require_skill`, `require_rel_l2`, `gates_block`, plus the CCF-specific
  `ccf_absolute_gates` and the IPM / Boussinesq scaffold gates. The
  `honesty` sub-block with `navier_stokes_proof_claim: False` is already
  emitted by the scaffold gates.
- `benchmarks/_common.py` — `provenance()` (schema, UTC timestamp, hardware
  class, dependency versions, config, starting RSS), `write_json()` into
  `docs/benchmarks/`, `median_time_ms()`, `rss_mb()`.
- `docs/benchmarks/*.json` — 28 committed artifacts; `*_smoke.json` is the CI
  variant, the bare name is the `--full` run.
- `docs/benchmarks/pinn_four_gap_matrix.md` — the capability matrix pattern.
- `.github/workflows/ci.yml` — per-package test jobs, `mkdocs build --strict`,
  the docs-snippet runner, the skills drift check.

**Confirmed gap**, and it is narrow: there is no gate helper for *certificate
coverage* (does the enclosure contain the truth, over `N` instances), none for
*torch/jax parity* as a benchmark gate rather than a unit test, and none for
*cost* (a method that wins on accuracy at `100x` the compute has not won).

## 4. Mathematics

### The three ordered questions, restated

From `benchmarks/_gates.py`, in order, because the order is load-bearing:

1. **Is the reference physically valid?** A comparison against a diverged
   reference is not a result at any accuracy. `require_reference_valid` enforces
   finiteness plus a physical invariant (for the periodic heat equation, the
   maximum principle `max|u(t)| <= max|u(0)|`). A failure raises
   `INVALID EXPERIMENT` rather than recording a number.
2. **Does it beat the zero predictor?** Nash-Sutcliffe skill
   `1 - MSE(pred)/MSE(0)`, required positive. A method with negative skill is
   worse than outputting zeros, and reporting its relative improvement over
   another failing method is the classic way to make a non-result look like a
   result.
3. **Is the absolute error below a named threshold?** `require_rel_l2` with a
   `max_rel_l2` chosen from the problem, not from the observed value.

### What is not a gate

- A relative improvement over a weak or unnamed baseline.
- `isfinite`, `not nan`, "training converged", or any existential check.
- A threshold set after seeing the result. If the threshold moved to accommodate
  the number, the gate has been deleted, not passed.
- A single-seed result for anything stochastic.
- An aggregate across benchmarks that hides per-benchmark losses.

### The three extensions this spec adds

**(a) Certificate coverage.** For any spec that produces a sound enclosure, the
gate is not "the enclosure is tight" but "the enclosure *contains the truth* in
`100%` of `N` instances". Coverage below `100%` is a soundness bug, categorically
different from a wide interval:

```
require_enclosure_coverage(enclosures, truths, *, n_min=1000)
    -> raises if any truth falls outside; reports median width separately
```

Width is reported, never gated together with coverage, so that a tight-but-
unsound implementation cannot trade one against the other.

**(b) Parity.** Where a spec ships torch and jax twins, bit-identity is a
repository invariant, and the benchmark should record it rather than leaving it
to a unit test that a benchmark reader never sees:

```
require_backend_parity(a, b, *, name)  -> exact equality, not allclose
```

**(c) Cost.** Accuracy at unbounded compute is not a win:

```
require_cost_parity(method_ms, baseline_ms, *, max_ratio)
```

Any accuracy gate in this tree that does not sit beside a cost gate is
incomplete, and the two must be reported in the same artifact.

### Seeds and reporting

Stochastic results report `n_seeds >= 5`, and the artifact carries **every
seed's value**, not only the mean. The gate is applied to the worst seed unless
the spec states otherwise and says why. Reporting a mean across seeds while
gating on the best is fabrication.

### The artifact contract

Every benchmark writes, via `_common.write_json`:

```json
{
  "schema": "<name>-v1",
  "generated_utc": "...",
  "hardware_class": "commodity x86-64 CPU (...)",
  "versions": {"python": "...", "numpy": "...", "jax": "...", "torch": "..."},
  "config": {"...": "..."},
  "seeds": [0, 1, 2, 3, 4],
  "per_seed": [{"...": "..."}],
  "baseline": {"name": "LightGBM 4.x", "...": "..."},
  "gates": {"all_passed": false, "entries": [{"name": "...", "passed": false}]},
  "honesty": {"<claim_flag>": false}
}
```

`all_passed: false` is a perfectly acceptable committed artifact. **A benchmark
that cannot fail is not a benchmark**, and the tree should contain failing
artifacts as evidence that the gates bind.

## 5. Worked example

**A gate that binds, and the same gate rewritten to not bind.**

Suppose a new operator claims to solve a heat-equation transfer task. The
reference is an integrator output; the prediction is the model's.

The correct gate block:

```python
from _gates import (
    gates_block, require_reference_valid, require_rel_l2, require_skill,
)

entries = [
    require_reference_valid(u_ref, u0_max_abs=float(np.max(np.abs(u0))),
                            name="heat_reference"),
    require_skill(u_pred, u_ref, min_skill=0.0, name="heat_skill"),
    require_rel_l2(u_pred, u_ref, max_rel_l2=1e-3, name="heat_rel_l2"),
]
block = gates_block(entries)
```

Run it and suppose the outcome is `rel_l2 = 4.2e-3`, so `all_passed` is
`false`. There are exactly two honest responses: improve the method, or report
the failure. There is a third, dishonest one that is very easy to write:

```python
# Every line below deletes a gate while appearing to keep it.
require_rel_l2(u_pred, u_ref, max_rel_l2=5e-3)          # threshold moved to fit
require_rel_l2(u_pred, u_baseline, max_rel_l2=1e-3)     # compared to a weaker arm
if np.isfinite(u_pred).all(): passed = True             # existential check
rel = rel_l2(u_pred, u_ref) / rel_l2(u_base, u_ref)     # ratio of two failures
```

Each of these produces a JSON with `all_passed: true` and each is worthless. The
protection is not vigilance, it is that the threshold is written down in the
spec's section 8 **before** the implementation exists, and a reviewer can
compare the two.

**The CCF gate as the model to copy.** `CCF_STRETCH_RESIDUAL_GATE = 1e-13` sits
in `_gates.py` as a module constant, against a documented achieved floor many
orders of magnitude above it. That gate has not passed. It has also never been
weakened, and the campaign reports its distance from it. That is the standard:
**the threshold is a property of the problem, and the result is a property of
the method.** Spec 07-03 works against exactly this constant.

**A certificate-coverage gate, contrasted.** For an enclosure method:

```python
require_enclosure_coverage(encl, truth, n_min=1000)   # must be 100%, no tolerance
report_median_width(encl)                             # reported, never gated with it
```

A `99.9%` coverage over `1000` instances is one soundness violation, which
means the implementation is wrong. Writing the gate as `>= 99%` would convert a
correctness bug into a passing number, which is precisely the failure mode the
sound-enclosure register exists to prevent.

## 6. Proposed API

Additions to `benchmarks/_gates.py`; everything else there already exists.

```python
def require_enclosure_coverage(
    enclosures: Sequence[tuple[float, float]],
    truths: Sequence[float],
    *,
    n_min: int = 1000,
    name: str = "enclosure",
) -> dict[str, Any]:
    """Coverage must be exactly 1.0. Raises on the first miss with its index.
    Median and max width are reported in the verdict but never gated jointly."""

def require_backend_parity(
    a: Any, b: Any, *, name: str = "parity",
) -> dict[str, Any]:
    """Exact equality between torch and jax outputs. Not allclose."""

def require_cost_parity(
    method_ms: float, baseline_ms: float, *, max_ratio: float, name: str = "cost",
) -> dict[str, Any]: ...

def require_all_seeds(
    per_seed: Sequence[dict[str, Any]], *, key: str, expected: float, tol: float,
    name: str = "all_seeds", min_seeds: int = 5, direction: str | None = None,
) -> dict[str, Any]:
    """Gate the worst seed. Records every seed's value in the verdict.
    Default: |value - expected| <= tol for every seed. With direction='min'
    require value >= expected; with direction='max' require value <= expected.
    Refuses n_seeds < min_seeds (default 5) unless overridden."""

# Landed with Wave-0 falsifier A6 (04-01 G2); shared by later scaling gates.
def require_scaling_exponent(
    x, y, *, expected: float, tol: float, min_decades: float, name: str,
) -> dict[str, Any]:
    """Least-squares slope of log(y) vs log(x). Raises if decades < min_decades,
    if any value is non-positive/non-finite, or if |fitted - expected| > tol.
    Reports decades, n_points and r2 (never gated jointly)."""

def require_rel_error(
    measured: float, expected: float, *, max_rel: float, name: str,
) -> dict[str, Any]:
    """Scalar |measured - expected| / |expected| <= max_rel."""

def require_within_stderr(
    measured: float, reference: float, stderr: float, *,
    max_sigmas: float = 3.0, name: str,
) -> dict[str, Any]:
    """|measured - reference| <= max_sigmas * stderr."""

# Landed with Wave-0 falsifier A7 (05-01 G7); validity guard, not a soft gate.
def require_capture_rate(
    n_captured: int, n_total: int, *, min_rate: float = 1.0, name: str,
) -> dict[str, Any]:
    """Require capture rate >= min_rate. Raises RuntimeError with
    INVALID EXPERIMENT on failure so a broken regime cannot report an exponent."""
```

## 7. Practical use cases

1. **Reviewing a spec's section 8** against a single definition instead of a
   remembered convention.
2. **Reviewing a pull request** by reading the committed artifact's `gates`
   block rather than the prose summary.
3. **Detecting gate erosion** over time: the thresholds are module constants, so
   a change to one is a visible diff.
4. **Making failure publishable.** A committed artifact with
   `all_passed: false` is a legitimate, informative deliverable, which is what
   makes the honest path the low-friction one.

## 8. Acceptance gates

This spec's own gates, since a gate protocol that is not itself gated is an
opinion.

- **G1 self-test.** Each new helper has a test that constructs a deliberately
  failing input and asserts the raise, and a passing input and asserts the
  verdict. A helper that cannot fail is rejected.
- **G2 coverage strictness.** `require_enclosure_coverage` raises on a single
  miss out of `10 000` synthetic instances. Asserted by test.
- **G3 parity exactness.** `require_backend_parity` fails on a `1`-ulp
  difference. Asserted by test, because `allclose` creeping in later is the
  predictable regression.
- **G4 artifact schema.** Every artifact in `docs/benchmarks/` validates against
  the schema, including the required `gates`, `baseline` and `seeds` keys. A CI
  test enumerates the directory so a new artifact cannot skip validation.
- **G5 no unnamed baselines.** The schema check rejects a `baseline` block
  without a `name`, so "compared to a baseline" cannot survive review.

## 9. Benchmark plan

Not a benchmark itself. The deliverable is:

- extended `benchmarks/_gates.py` with self-tests in
  `tests/test_gates_protocol.py` (lives under root `tests/`, **not**
  `packages/omnibias-core/tests/`: `_gates.py` imports numpy at module scope,
  core declares no numpy dependency, and the only CI job that runs core tests
  is the numpy-free `core` job — a core-tests path would skip everywhere),
- a schema validator run over `docs/benchmarks/*.json` in CI,
- this file referenced from `benchmarks/README.md`.

The three helpers `require_scaling_exponent`, `require_rel_error`, and
`require_within_stderr` landed with Wave-0 falsifier A6. The validity guard
`require_capture_rate` (raises `RuntimeError` / `INVALID EXPERIMENT`) landed
with Wave-0 falsifier A7. The worst-seed helper `require_all_seeds` landed with
the A7 hardening pass (and is reused by Wave-0 A4). The three remaining
extensions (`require_enclosure_coverage`, `require_backend_parity`,
`require_cost_parity`) and the schema validator remain open.

## 10. Honesty and scope

- A passing gate is **evidence about one benchmark on one hardware class**, not
  a general claim. The `hardware_class` and `versions` fields exist so nobody
  reads more into it.
- Gate thresholds encode a judgement about what is scientifically interesting.
  They can be revised, but only *upward* (stricter) without discussion, and any
  loosening needs the reason recorded in the diff.
- Empirical gates are the **lowest rung** of the claim ladder in spec 06-02. A
  passing benchmark is not a sound enclosure and is not a proof; artifacts must
  not carry `theorem_prover_verified` or `mathlib_verified`, which are earned
  only by a Lean pass.
- The `honesty` sub-block pattern (already used by `ccf_absolute_gates` and
  `ipm_boussinesq_scaffold_gates`) is how forbidden claims are pinned to
  `False` inside the artifact, and Group 07 specs must use it.
- No collapse of either kind appears in this spec; it is pure protocol.

## 11. Open questions and risks

- **Threshold drift by attrition.** The real risk is not a single loosened
  threshold but many small ones over years. A periodic audit that diffs
  thresholds against their introducing commit is the mitigation, and it is not
  yet written.
- **Cost gates and hardware.** `max_ratio` is meaningful only within one
  hardware class; comparing a CPU smoke ratio to a GPU full-run ratio is
  meaningless and the schema should make the class explicit at comparison time.
- **Seed count.** Five seeds is a compromise. For high-variance results it is
  too few, and the spec cannot fix that centrally — individual specs must raise
  it and say why.
- **Coverage gates need a truth.** `require_enclosure_coverage` presumes a known
  true value, available only for synthetic instances. For real problems the
  soundness argument is the interval arithmetic itself, and the gate cannot
  substitute for it.
- **Falsifier.** If the schema validator finds existing artifacts that cannot be
  brought into conformance, the schema is wrong and must be adjusted to reality
  rather than the artifacts quietly excluded.

## 12. Implementation checklist

- [x] Extend `benchmarks/_gates.py` with `require_scaling_exponent`,
      `require_rel_error`, `require_within_stderr` (Wave-0 A6)
- [x] Extend `benchmarks/_gates.py` with `require_capture_rate` (Wave-0 A7;
      raises `RuntimeError` / `INVALID EXPERIMENT`)
- [x] Extend `benchmarks/_gates.py` with `require_all_seeds` (A7 hardening /
      Wave-0 A4; worst-seed gate, refuses `n_seeds < 5` without override)
- [ ] Extend `benchmarks/_gates.py` with `require_enclosure_coverage`,
      `require_backend_parity`, `require_cost_parity`
- [x] `tests/test_gates_protocol.py` with a deliberately-failing case per
      landed helper (path corrected: not under core tests)
- [ ] Artifact schema validator enumerating `docs/benchmarks/*.json` in CI
- [ ] Reject `baseline` blocks without a `name`
- [ ] Backfill `seeds` / `per_seed` / `baseline` into existing artifacts, or
      record why an artifact is exempt
- [x] Reference this file from `benchmarks/README.md`
- [ ] Threshold-audit script diffing gate constants against their introducing
      commit
- [x] Index row in `theory/README.md`
