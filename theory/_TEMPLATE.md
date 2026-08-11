# <spec number> <title>

> Copy this file, keep every heading, delete the guidance in angle brackets.
> A spec is finished when an implementer who has never read this conversation
> can build the thing and know whether they succeeded.

## 1. Thesis and status

<One sentence. What is the object, and what does it buy?>

- **Status**: concept | designed | gated | shipped
- **Depends on**: <other spec numbers, or "none">
- **Blocks**: <spec numbers that need this first, or "none">

## 2. Where it lands

<Existing submodule, new submodule, or new package. Justify against the "earn
independent existence" rule in AGENTS.md: a new distribution needs a distinct
domain, a distinct dependency or maturity tier, or a distinct audience.
Defaulting to a submodule is the right answer most of the time.>

## 3. Prior art in omnibias

<Exact module paths of what already exists, and the precise delta this adds.
Cite real symbols. If a capability is genuinely absent, say "Confirmed gap" and
name the nearest precedent so nobody re-derives it. Never guess: check.>

## 4. Mathematics

<Definitions, derivation, lemma or theorem statements, and the limits taken.
State which limit is in play: the founding bias collapse (spread `delta -> 0`,
`K` biases coalesce, output is a smooth `sigma^(K-1)`) or temperature collapse
(`beta -> inf`, one gate hardens into a 0/1 feasibility step). Give error terms.>

## 5. Worked example

<Small and concrete, with actual numbers. A reader should be able to check it by
hand or in ten lines of numpy. Show the expected output.>

## 6. Proposed API

<Signatures. Note the torch and jax twins explicitly: they must be bit-identical
and must share coefficients from `omnibias.core.polynomials`. State dtype policy
(framework default, never a hardcoded float32) and any tracing constraints for
jax. Mark clearly that this API does not exist yet.>

## 7. Practical use cases

<Three to six. Each one: the problem, why this primitive helps specifically, and
the expected win over the obvious alternative. Vague use cases are worse than
none, because they hide the fact that nobody needs the feature.>

## 8. Acceptance gates

<Falsifiable, absolute thresholds and the named baseline to beat. Follow the
three questions in `benchmarks/_gates.py`: is the reference physically valid,
does the method beat the zero predictor (skill > 0), and is the absolute error
below a named threshold? A relative improvement over a weak baseline is not a
gate.>

## 9. Benchmark plan

<CPU smoke artifact plus a `--full` multi-seed run, mirroring the existing
benchmarks. Name the artifact path under `$OMNIBIAS_SCRATCH` (default
`artifacts/`), the committed summary JSON, and the CI job.>

## 10. Honesty and scope

<What is NOT claimed. The two-collapse note if either limit appears. The
certificate tier if any: empirical gate, sound enclosure, `theorem_prover_verified`
(earned only by a genuine Lean kernel pass), or `mathlib_verified` (a distinct
tier, never conflated with the kernel one).>

## 11. Open questions and risks

<Where this could fail, what is unproven, what would falsify it. A spec with no
risks section has not been thought about hard enough.>

## 12. Implementation checklist

- [ ] <files to create, with paths>
- [ ] <tests, including a regression test for every behavioral change>
- [ ] <torch/jax parity test if both backends are touched>
- [ ] <docs page and mkdocs nav entry if user-visible>
- [ ] <CI job>
- [ ] <regenerate the `__all__` block of any touched `__init__.py`>
- [ ] <index row in `theory/README.md` updated>

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
  calls against real signatures; opting out needs a directive with a stated
  reason.
- **Earned flags**: `theorem_prover_verified` is set only by a genuine
  `lake build` pass of the Mathlib-free kernel, and `mathlib_verified` only by a
  genuine pass of the Mathlib-backed project. Asserting either without a pass
  blocks the verdict.
- **Typing tier**: the strict CI gate covers `core`, `torch`, `jax` and
  `ferminet`. Newly authored modules should be written strict-clean regardless
  of tier; curated beta modules can be added to
  `scripts/mypy_strict_allowlist.txt` once
  `mypy --strict --follow-imports=silent <file>` is clean.
