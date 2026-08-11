# 01-11 Rational exactness and new Lean obligations

## 1. Thesis and status

The consistency conditions of collapse are **finite identities over the
rationals**, which is exactly the payload class the existing Lean bridge already
discharges, so `theorem_prover_verified` becomes reachable for the new
mathematics itself rather than only for downstream numerical enclosures.

- **Status**: designed
- **Depends on**: 01-01, 01-04
- **Blocks**: 07-01, 07-05

## 2. Where it lands

`packages/omnibias-core/src/omnibias/core/proof/obligations/rational_stencil.py`
plus a Lean lemma file in `formal/omnibias-verified-kernel/`. No new package:
the certificate and bridge machinery already exists and this is a new obligation
class inside it.

## 3. Prior art in omnibias

- `packages/omnibias-difference/src/omnibias/difference/_core/stencil.py` —
  `signs_exact(order, delta) -> tuple[Fraction, ...]` and `offsets_exact`. The
  weights are already exact rationals, not floats. This is the fact the whole
  spec rests on.
- `packages/omnibias-core/src/omnibias/core/proof/certificate.py` — certificate
  format v1: canonical, hash-sealed JSON, `verify_certificate_digest`.
- `packages/omnibias-core/src/omnibias/core/proof/lean_check.py` — the bridge
  that extracts a certificate's finite rational obligation, emits Lean chaining
  the kernel's `ZInterval` soundness lemmas, and runs `lake build`.
- `formal/omnibias-verified-kernel/` — the Mathlib-free Lean 4 kernel:
  sound `ZInterval` algebra plus a finite rational obligation checker,
  `sorry`-free.
- `formal/omnibias-analytic/` — the Mathlib-backed checker feeding the distinct
  `mathlib_verified` tier.

**Confirmed gap.** Every obligation the bridge discharges today concerns a
*numerical enclosure produced by a run* (a spectral-gap positivity, the sign of
an enclosed quantity). No obligation asserts a property of the **method** itself.
The stencil consistency conditions are the natural first such obligation, and
they are finite and rational, so they fit the kernel's scope exactly.

## 4. Mathematics

### The obligations

For a stencil with scale-free weights `A_{i,p}` at nodes `c_i` (spec 01-04),
consistency for target order `q` with `N` data pairs is the finite system

```
C_j :   sum_{(i,p)} A_{i,p} c_i^(j-p) / (j-p)!  =  [j == q],      j = 0 .. N-1
```

Each `C_j` is an equality between two rational numbers. There are `N` of them.
Nothing infinite, nothing analytic, no limits: **a finite conjunction of
rational identities.**

Three obligation families follow.

1. **Stencil consistency.** `C_0 .. C_{N-1}` hold for the generated weights.
   This certifies that the weight generator produced a scheme of the claimed
   order.
2. **Leading-coefficient value.** `C_N = sum A_{i,p} c_i^(N-p) / (N-p)!` equals
   the rational the generator reported. This certifies the error constant, which
   is what the truncation bound multiplies.
3. **Multi-pack support consistency.** For a `MultiPackSpec`, the incidence
   matrix satisfies the Polya inequalities and the confluent Vandermonde
   determinant is a specific nonzero rational. This certifies poisedness
   *exactly*, upgrading spec 01-01's numerical rank test.

### Why this is in scope for the kernel

The kernel is deliberately Mathlib-free (Lean 4 core only) so CI can check it
cheaply, and its stated scope is finite, rational obligations. Infinite analytic
obligations, limits, continuum statements and asymptotics are explicitly out of
scope and are not expressed in Lean at all.

Each `C_j` is a rational equality. A determinant of a rational matrix is a
rational. A Polya inequality is a comparison of integers. All three families sit
inside the scope. What is emphatically **not** in scope, and must never be
smuggled in:

- the statement `f_K(z; delta) -> sigma^(K-1)(z)` as `delta -> 0` (a limit);
- the truncation bound `|E| <= |C_N| h^r M_N` as a statement about all `f` (a
  quantified analytic statement about a function class);
- any claim about the activation's analyticity.

The honest split: **Lean certifies the algebra; the analysis stays in the
enclosure register.** The truncation bound becomes a certified statement only in
the sense that its *rational constant* is verified and its *interval evaluation*
is sound; the underlying Taylor theorem is a mathematical fact used, not proved.

### What this buys

Today a `theorem_prover_verified` verdict says "the numbers this run produced
satisfy this sign condition". With these obligations it can also say "the weights
this method uses satisfy their consistency conditions exactly". The first is
about an instance; the second is about the tool. That is a meaningful upgrade in
what the flag means, and it is honest because both are still finite rational
facts.

### The obligation payload

Following the existing `rational_identity` payload shape:

```json
{
  "kind": "rational_stencil_consistency",
  "nodes": [["-1", "1"], ["0", "1"], ["1", "1"]],
  "orders": [[0], [0], [1]],
  "weights": [[["-2", "3"]], [["2", "3"]], [["1", "3"]]],
  "target_order": 1,
  "conditions": [
    {"j": 0, "lhs": ["0", "1"], "rhs": ["0", "1"]},
    {"j": 1, "lhs": ["1", "1"], "rhs": ["1", "1"]},
    {"j": 2, "lhs": ["0", "1"], "rhs": ["0", "1"]}
  ],
  "leading_coeff": ["5", "18"]
}
```

Rationals are transported as integer numerator/denominator pairs, never as
floats, so the Lean side never sees a decimal.

## 5. Worked example

Use the Birkhoff scheme from spec 01-04: nodes `(-1, 0, 1)` in units of `h`,
orders `({0}, {0}, {1})`, target `q = 1`, weights `A = (-2/3, 2/3, 1/3)`.

Conditions, all exact:

```
j = 0:  (-2/3)(1) + (2/3)(1) + (1/3)(0)                = 0   = [0 == 1]  OK
j = 1:  (-2/3)(-1) + (2/3)(0) + (1/3)(1)               = 1   = [1 == 1]  OK
        ( -2/3 * (-1)^1 / 1! ) + 0 + ( 1/3 * (1)^0 / 0! ) = 2/3 + 1/3 = 1
j = 2:  (-2/3)(1/2) + (2/3)(0) + (1/3)(1)              = -1/3 + 1/3 = 0  OK
```

Leading coefficient:

```
C_3 = (-2/3)(-1)^3/3! + (2/3)(0) + (1/3)(1)^2/2! = 1/9 + 1/6 = 5/18
```

Every quantity is a fraction of small integers. The emitted Lean obligation is a
conjunction of three equalities and one value assertion over `Rat`, which the
Mathlib-free kernel discharges by `decide`-style evaluation on the rational
arithmetic it already proves sound.

Poisedness for the same scheme: the confluent Vandermonde determinant is

```
det | 1   1   0 |
    | -1  0   1 |     =  -3/2   != 0
    | 1/2 0   1 |
```

so the scheme is poised, and the certificate carries `-3/2` as the witness.

## 6. Proposed API

Extends existing machinery; the certificate and bridge already exist.

```python
# omnibias/core/proof/obligations/rational_stencil.py
def stencil_consistency_obligation(stencil: IrregularStencil) -> Obligation:
    """Finite conjunction of rational equalities C_0 .. C_{N-1} plus C_N."""

def poisedness_obligation(spec: MultiPackSpec) -> Obligation:
    """Polya inequalities plus a nonzero confluent Vandermonde determinant."""

def seal_stencil_certificate(stencil, *, run_lean: bool = True) -> Certificate:
    """Builds the v1 certificate, hashes it, and (optionally) drives
    `omnibias.core.proof.lean_check`. `theorem_prover_verified` is set only on a
    genuine kernel pass."""
```

Lean side: a new lemma file in `formal/omnibias-verified-kernel/` proving the
soundness of the rational-equality evaluator used for this payload class, kept
`sorry`-free, and wired into the existing `lake build` target so CI checks it.

## 7. Practical use cases

1. **Certifying the tool, not only the run.** A user asking "how do I know your
   sixth-order stencil is sixth order?" gets a kernel-checked answer.
2. **Poisedness with no numerical caveat.** Spec 01-01's representation claims
   currently rest on a float rank test; this replaces it with an exact,
   machine-checked witness.
3. **Regression protection for the generator.** If a refactor perturbs a weight,
   the obligation fails loudly at the proof layer, not silently at the fifth
   decimal.
4. **A credible base for the frontier track.** Group 07 spends its credibility on
   what the certificate flags mean; widening the verified surface to include the
   method's own algebra strengthens every downstream claim.
5. **A template for future obligation classes**, for example the moment
   conditions of spec 01-05, which are the same shape.

## 8. Acceptance gates

- **G1 kernel pass.** For a curated set of at least 20 stencils (uniform,
  irregular, Birkhoff), `lake build` succeeds and `theorem_prover_verified` is
  `True`.
- **G2 negative control.** For deliberately corrupted weights, the Lean build
  **fails** and the flag stays `False`. A bridge that cannot fail is not a
  bridge.
- **G3 no-toolchain degradation.** With no Lean toolchain present, the pipeline
  runs, produces a sealed certificate, and leaves `theorem_prover_verified` as
  `False` without raising.
- **G4 tamper evidence.** Editing any field of a sealed certificate makes
  `verify_certificate_digest` fail.
- **G5 tier separation.** `mathlib_verified` remains `False` throughout; nothing
  in this spec touches the Mathlib-backed project, and the two flags are never
  set by the same code path.

## 9. Benchmark plan

Not a performance feature. The measurements that matter are:

- Lean build wall time per obligation, recorded so CI cost stays bounded.
- Obligation size (number of conditions, integer magnitudes) versus `N`, since
  exact rational arithmetic can blow up coefficients.

Both go into `docs/benchmarks/rational_stencil_smoke.json`.

## 10. Honesty and scope

- **Only the algebra is proved.** The limit `delta -> 0`, Taylor's theorem, and
  any statement quantified over a function class are **not** expressed in Lean
  and must not be described as verified. The kernel's scope is finite rational
  obligations, and this spec stays inside it.
- `theorem_prover_verified` is earned **only** by a genuine kernel pass. It can
  never be asserted by the certificate, and asserting it without a pass blocks
  the verdict. That existing invariant is untouched here.
- `mathlib_verified` is a **distinct tier** produced by the Mathlib-backed
  project. The two are never conflated, and this spec does not produce it.
- The founding bias collapse is the mathematical context; the *limit* itself is
  precisely the part that is out of Lean scope. Say "the collapse weights are
  verified", never "the collapse is verified".
- No temperature collapse appears in this spec.

## 11. Open questions and risks

- **Coefficient blowup.** Exact rationals from a large Vandermonde solve can have
  enormous numerators, making the Lean term large and the build slow. Measure
  and cap; consider emitting the obligation in a modular form if needed.
- **CI cost.** The kernel is cheap by design, but 20 obligations per run is not
  free. Keep the CI set small and run the full set in the heavier tier.
- **Scope creep.** The temptation to "just also prove the convergence" is exactly
  what the honesty stack forbids. The reviewer's job is to reject any Lean file
  that mentions a limit.
- **Falsifier.** If the obligations never catch a real regression and cost real
  CI time, they are ceremony; keep them only if the negative-control discipline
  is maintained.

## 12. Implementation checklist

- [ ] `packages/omnibias-core/src/omnibias/core/proof/obligations/rational_stencil.py`
- [ ] Lean lemma file in `formal/omnibias-verified-kernel/`, `sorry`-free
- [ ] Wire into the existing `lake build` target and CI kernel check
- [ ] Curated obligation set (uniform, irregular, Birkhoff) as a test fixture
- [ ] Negative-control test asserting the Lean build fails on corrupted weights
- [ ] No-toolchain degradation test
- [ ] Tamper-evidence test on the sealed certificate
- [ ] Tier-separation test asserting `mathlib_verified` stays `False`
- [ ] Docs page and nav entry
- [ ] Index row in `theory/README.md`
