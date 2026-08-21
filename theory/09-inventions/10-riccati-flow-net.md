# 09-10 Riccati flow net

## 1. Thesis and status

A layer that is the **time-`t` flow of the founding Riccati ODE**
`sigma' = sigma(1-sigma)` (or `tanh' = 1-tanh^2`), not `sigma(Wx+b)`.
Depth is integration time.

- **Status**: gated
- **Depends on**: 09-01
- **Blocks**: none

### Operator card

- **Benefit.** The Riccati identity *is* the dynamics; jets of the
  flow are closed form.
- **How it works.** Hidden state `s(t)` solves the Riccati ODE with
  `s(0)` an affine of the input. Output is `s(T)`. `T` is a (possibly
  learned) depth.
- **Strength.** 1-D monotone maps and logistic-growth features.
- **When to use.** After first-bet cluster; as a contrast to 08-08 DEQ
  and to CNF.
- **When not.** As a DEQ (`u = sigma(W u + x)`). As a CNF (density
  ODE). Not CCF.
- **Accuracy floor.** Closed form of logistic / tanh flow; learned `T`
  can saturate.

## 2. Where it lands

`omnibias.{torch,jax}.architectures`. Fails the package test.

## 3. Prior art in omnibias

- `omnibias.core.polynomials` — Riccati-derived tower coefficients.
- Spec 08-08 — implicit `u = sigma(W u + x)`, IFT with exact `sigma'`.
- `omnibias.score.flow.integrate_cnf` — density transport, not Riccati
  state.

**Confirmed gap.** No layer whose forward is the closed-form Riccati
flow in time.

## 4. Mathematics

Logistic: `ds/dt = s(1-s)`, solution
`s(t) = s0 / (s0 + (1-s0) e^{-t})` for `s0 in (0,1)`. Tanh flow
`dtanh = 1-tanh^2` is `tanh(artanh(s0)+t)`. These are **not** bias
collapse (no `delta -> 0` pack). Bias collapse still supplies jets
*of* `s(t)` in `s0` if needed.

No temperature collapse.

## 5. Worked example

`s0 = 0.25`, `t = 1`. Logistic:
`s(1) = 0.25 / (0.25 + 0.75/e) ≈ 0.25 / (0.25 + 0.275909) ≈ 0.475021`.
The implementer matches this to `1e-12`. `ds/ds0` at this point is a
named closed form the jet head must match.

## 6. Proposed API

Does not exist yet. Bit-identical torch / jax twins; default dtype.

```python
# omnibias/torch/architectures/riccati_flow.py  (and jax twin) — proposed
@dataclass(frozen=True)
class RiccatiFlowConfig:
    family: str = "logistic"   # logistic | tanh
    t: float = 1.0

def riccati_flow(s0, *, config: RiccatiFlowConfig):
    """Closed-form flow. Raises if s0 not in (0,1) for logistic."""
```

## 7. Practical use cases

1. **Logistic growth regression** (scalar `s0 -> s(T)`).
2. **Ablation vs** one `sigmoid` layer of the same width (must win on
   the logistic target or G2 fails).
3. **Not** ImageNet.

## 8. Acceptance gates

- **G1 closed form.** Worked example to `1e-12`.
- **G2 skill.** Fit `T` and an affine on 32 samples of the logistic
  flow: RMSE `< 1e-6` (five seeds). A vanilla `sigmoid` MLP of width 8
  is the baseline; Riccati-flow must be strictly better on median RMSE.
- **G3 distinct.** Artifact `claimed_deq` and `claimed_cnf` are
  `false`.
- **G4 parity.** torch / jax bit-identical on G1.

## 9. Benchmark plan

- `benchmarks/riccati_flow_net.py`
- Smoke: `docs/benchmarks/riccati_flow_net_smoke.json`
- `--full`: `$OMNIBIAS_SCRATCH/inventions/riccati_flow/`

## 10. Honesty and scope

- Not bias collapse (unless a jet-in-`s0` head is added). Not
  temperature collapse. Not DEQ. Not CNF. Not CCF.
- Certificate tier: empirical.

## 11. Open questions and risks

- **Saturation** as `T -> inf` (`s -> 1`). G2 uses moderate `T`.
- **Falsifier.** G2: a sigmoid MLP already represents the flow.

## 12. Implementation checklist

- [x] `omnibias.{torch,jax}.architectures` Riccati flow
- [x] Domain checks for logistic
- [x] `benchmarks/riccati_flow_net.py` plus smoke JSON
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
