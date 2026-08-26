# 08-09 Certified step (proof-carrying update)

## 1. Thesis and status

A parameter update is **legal** only if an `omnibias.verify` enclosure
of a named input-output property (Lipschitz or output box) remains
inside a declared bound after the step, so training can refuse an
update that would break a certificate.

- **Status**: shipped (G1–G3 CI; empty is a reject, not robustness)
- **Depends on**: 08-01
- **Blocks**: none

### Operator card

- **Benefit.** The iterate always carries a sound I/O bound, or the
  step is rejected. Distinct from 08-04 (root of `F` vs property of
  `f_theta`).
- **How it works.** Propose `theta'` (any optimizer). Run
  `lipschitz_bound` and/or `taylor_output_bounds` on a named input cell.
  Accept iff the sealed upper bound is `<=` the declared cap. Empty or
  exploding enclosure ⇒ reject (not "robust").
- **Strength.** Tiny nets where interval / Taylor-model bounds stay
  tight (1-D Lipschitz toy).
- **When to use.** After a model is already small-Lipschitz and the
  product is a network-plus-certificate. Never as the CCF path.
- **When not.** Deep nets (interval explosion); claiming robustness
  without a nonempty enclosure; stretch `1e-13`.
- **Accuracy floor.** Enclosure width. Vacuous bounds refuse every
  step.

## 2. Where it lands

A callback in `omnibias.verify` (e.g. `omnibias.verify.train_step`)
invoked from `omnibias.{torch,jax}.optim`. Not a package. Verify
already owns the certificates; this spec is the **policy**.

## 3. Prior art in omnibias

- `omnibias.verify._core.certificates.lipschitz_bound`
- `omnibias.verify._core.taylor.taylor_output_bounds`
- `omnibias.verify._core.surrogate_bounds.mollified_lipschitz_iv`
- Spec 08-04 — unique **zero** of a residual map. Do not merge.

**Confirmed gap.** Optimizers do not consult verify enclosures before
committing `theta'`.

## 4. Mathematics

Let `P(theta)` be a sound enclosure of a scalar property (Lipschitz
constant on a cell `X`, or `sup_{x in X} |f_theta(x)|`). Outward-rounded
interval / Taylor-model arithmetic guarantees
`true P in enclosure`. The step constraint is

```
sup enclosure(P(theta'))  <=  P_max
```

with `P_max` fixed before the run. If the enclosure is empty or
`+inf`, the predicate is **false** (reject).

Activations use the same `sigma` tower as the differentiable register
when the bound is Taylor-model based (bias collapse). Soft-gate
Lipschitz (`mollified_lipschitz_iv`) is temperature-style in `beta` if
that path is chosen — name the limit in the artifact
(`bias_collapse` vs `temperature_collapse`).

## 5. Worked example

**Affine scalar.** `f(x) = w x`, cell `X = [0, 1]`, Lipschitz `|w|`.
`P_max = 2`. From `w = 1.5`, a GN step to `w' = 2.5` has
`|w'| = 2.5 > 2` ⇒ reject. A step to `w' = 1.8` ⇒ accept.
The implementer must record one accepted and one rejected step in the
smoke artifact.

## 6. Proposed API

Does not exist yet.

```python
# omnibias/verify/train_step.py  — proposed
@dataclass(frozen=True)
class CertifiedStepConfig:
    property: str = "lipschitz"   # lipschitz | output_box
    p_max: float = 2.0
    cell: Any = None

@dataclass(frozen=True)
class CertifiedStepResult:
    accepted: bool
    bound_hi: float | None
    reason: str   # "ok" | "violates" | "vacuous"

def certified_accept(
    net, theta_trial, *, config: CertifiedStepConfig,
) -> CertifiedStepResult:
    """Does not exist yet."""
```

Verify core stays pure Python. Torch/jax only supply `theta_trial`.

## 7. Practical use cases

1. **1-D Lipschitz toy** — G1.
2. **Proof-carrying last-layer polish** (08-07) under a Lipschitz cap.
3. **Not** CCF / fluids stretch.

## 8. Acceptance gates

- **G1 accept/reject.** On the section-5 affine toy (or equivalent):
  100% of accepted steps have sealed Lipschitz `<= P_max`; at least
  one proposed step is rejected and would have violated `P_max`.
- **G2 vacuous reject.** A net / cell pair known to explode intervals
  returns `accepted=False`, `reason="vacuous"`, not `accepted=True`.
- **G3 no robustness slogan.** Artifact `robust_without_enclosure`
  is `false`.

## 9. Benchmark plan

- `benchmarks/certified_step.py`
- Smoke: `docs/benchmarks/certified_step_smoke.json`
- `--full`: `$OMNIBIAS_SCRATCH/training/certified_step/`
- CI smoke after implementation.

## 10. Honesty and scope

- Sound enclosure of an I/O property, not a root (08-04) and not a
  global min of a deep nest.
- Empty enclosure ⇒ reject. That is not a robustness claim.
- Not CCF stretch; not Clay. Hilbert × dictionary remains `~1e-1`.
- The bound uses the chain rule (jet / Lipschitz composition); it does
  not skip it.
- Name which collapse the bound uses (`beta` mollifier vs jet Taylor).

## 11. Open questions and risks

- **Explosion** is the predicted failure on deep nets. G2 makes that
  a tested reject, not a surprise.
- **Falsifier.** If an accepted step has true Lipschitz `> P_max`,
  soundness is broken — do not ship.

## 12. Implementation checklist

- [x] `omnibias.verify.train_step.certified_accept`
- [x] Tests: G1 toy, G2 vacuous, soundness (grid + random)
- [x] `benchmarks/certified_step.py` plus smoke JSON
- [x] `__all__` update
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
