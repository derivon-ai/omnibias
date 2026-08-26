# 09-25 World-model-as-jet

## 1. Thesis and status

Predict the **next N-jet of a trajectory** and plan on the Taylor
polynomial plus a Lohner remainder — a model-based loop whose planner
is a remainder statement, not sampled imagination.

- **Status**: shipped (G1–G4 CI; G4 Lohner-path purity; not NS global regularity)
- **Depends on**: 07-06, 09-01
- **Blocks**: none

### Operator card

- **Benefit.** Dynamics jets + validated remainder in one object.
- **How it works.** A tower field predicts `(x, Dx, ...)` at
  `t+dt`. `lohner_step` / `lohner_flow` encloses the true flow of a
  named vector field. Control / plan uses `T_N` and refuses actions
  whose remainder box exits a safe set.
- **Strength.** Low-dimensional analytic ODEs (harmonic oscillator,
  logistic).
- **When to use.** After 07-06 / dynamics users want a learned jet
  head.
- **When not.** As global regularity of NS. Not CCF. Not a general
  RL SOTA world model.
- **Accuracy floor.** Lohner remainder width; learned jets add model
  error.

## 2. Where it lands

`omnibias.dynamics` + `omnibias.core.verified.lohner`. No new package.

## 3. Prior art in omnibias

- `omnibias.core.verified.lohner.lohner_step`, `lohner_flow`,
  `LohnerSet`
- Spec 07-06 — validated orbits (closed-form Jacobians in validated
  flow)
- `omnibias.{torch,jax}.jet`

**Confirmed gap.** No learned next-jet head wired to a Lohner plan
loop.

## 4. Mathematics

For `x' = f(x)`, the N-jet in time is determined by Faà di Bruno /
Lie derivatives. Learned `f_theta` uses **bias collapse** for spatial
jets. Lohner is interval QR flow (sound enclosure). No temperature
collapse. Global existence stays external (06-02 / 07-02).

## 5. Worked example

Harmonic oscillator `x''=-x`, `x(0)=1`, `x'(0)=0`. At `dt=0.1`,
`x(0.1)=cos(0.1)≈0.995004`. A jet world-model of order 2 about `t=0`
is `1 - t^2/2`; at `0.1` that is `0.995`. Remainder vs `cos` is
`≈4.16e-6`. Lohner on the linear field must contain `cos(0.1)`.

## 6. Proposed API

Does not exist yet. Core Lohner stays pure Python. Learned head:
default dtype twins.

```python
# omnibias/dynamics/_core/jet_world.py — proposed
@dataclass(frozen=True)
class JetWorldConfig:
    jet_order: int = 2
    dt: float = 0.1

def predict_next_jet(state_jet, f, *, config: JetWorldConfig):
    """Taylor step. Pair with lohner_step for a box."""
```

## 7. Practical use cases

1. **Oscillator** one-step (worked example).
2. **Refuse** a step whose Lohner box exits `|x|<=1.1`.
3. **Not** 3-D fluid world models as NS.

## 8. Acceptance gates

- **G1.** Order-2 Taylor of the oscillator matches `0.995` to
  `1e-12` relative to `1 - dt^2/2`.
- **G2 skill.** Lohner box at `dt=0.1` contains `cos(0.1)` (sound)
  and `width < 0.05`. A learned `f_theta` is optional; if present,
  `|pred - cos| < 1e-3` (five seeds) and skill vs `x=0` positive.
- **G3 honesty.** `navier_stokes_proof_claim` is `false`;
  `continuum_claim` is `false`.
- **G4 purity.** Lohner path imports neither torch nor jax.

## 9. Benchmark plan

- `benchmarks/world_model_jet.py`
- Smoke: `docs/benchmarks/world_model_jet_smoke.json`
- `--full`: `$OMNIBIAS_SCRATCH/inventions/world_model_jet/`

## 10. Honesty and scope

- Finite-time enclosure on a named ODE. Not global regularity. Not
  stretch. Lean flags not asserted.
- Certificate tier: sound Lohner + empirical learned head.

## 11. Open questions and risks

- **Learned `f`** can make Lohner vacuous. G2 allows a frozen-`f`
  arm.
- **Falsifier.** G1 Taylor coefficient wrong.

## 12. Implementation checklist

- [x] Jet-world helpers in `omnibias.dynamics`
- [x] Oscillator Lohner test
- [x] `benchmarks/world_model_jet.py` plus smoke JSON
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
