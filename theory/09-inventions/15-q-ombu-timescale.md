# 09-15 q-OMBU / timescale hybrid

## 1. Thesis and status

Alternate Jackson q-derivative or Hilger delta layers with ordinary
`sigma` layers, with a **named limit** `q -> 1` / `mu -> 0` that
recovers the ordinary tower.

- **Status**: gated
- **Depends on**: 09-01
- **Blocks**: none

### Operator card

- **Benefit.** Discrete-continuous hybrid nets with a contract on the
  continuum limit.
- **How it works.** A block is either `q_derivative` /
  `delta_derivative` or an OMBU `sigma` cell. A residual head
  `q_derivative_residual` / `delta_derivative_residual` measures the
  limit error.
- **Strength.** q-series toys and time-scale dynamic equations.
- **When to use.** After first-bet; when qcalculus / timescale users
  want a stacked net.
- **When not.** As a claim that q-deformation solves a continuum PDE.
  Not CCF.
- **Accuracy floor.** The measured limit residual.

## 2. Where it lands

`omnibias.qcalculus` and `omnibias.timescale` as torch / jax layers
(those packages already have twins). No new package.

## 3. Prior art in omnibias

- `omnibias.qcalculus.{torch,jax}.q_derivative`,
  `q_derivative_limit`, `q_derivative_residual`
- `omnibias.timescale.{torch,jax}.delta_derivative`,
  `delta_derivative_limit`, `delta_derivative_residual`
- `omnibias.timescale._core.timescale.TimeScale`, `reals`
- Ordinary OMBU `OperatorBlock`

**Confirmed gap.** No stacked hybrid architecture that *trains* with
q / Hilger layers and *gates* the named limit.

## 4. Mathematics

Jackson q-derivative `D_q f(z) = (f(qz)-f(z))/((q-1)z)` (z ≠ 0)
satisfies `D_q -> d/dz` as `q -> 1`. Hilger delta `-> d/dt` as grain
`mu -> 0`. Those limits are **not** bias collapse and **not**
temperature collapse. Ordinary `sigma` layers in the stack still use
bias collapse for `sigma^(n)`.

## 5. Worked example

`f(z)=z^2`, `q=1.01`, `z=2`. `D_q f = (q^2 z^2 - z^2)/((q-1)z) =
z(q+1) = 2*2.01 = 4.02`. Ordinary derivative `4`. Residual `0.02`.
`q_derivative_limit` must return `4`. The implementer matches
`4.02` and `4` to `1e-12`.

## 6. Proposed API

Does not exist yet. Bit-identical torch / jax twins; default dtype.

```python
# omnibias/qcalculus/torch/hybrid.py  (and jax / timescale twins) — proposed
@dataclass(frozen=True)
class QOMBUConfig:
    q: float = 1.01
    n_q_layers: int = 1
    n_sigma_layers: int = 1

def q_ombu_forward(x, params, *, config: QOMBUConfig) -> tuple:
    """Returns (y, limit_residual)."""
```

## 7. Practical use cases

1. **Sanity:** hybrid net on `z^2` reports the worked residual.
2. **Linear dynamic equation** on a named time scale vs `reals()`.
3. **Not** a continuum NS claim via `mu -> 0`.

## 8. Acceptance gates

- **G1.** Worked `D_q z^2` matches `4.02` to `1e-12`.
- **G2 skill.** As `q` in `{1.1, 1.01, 1.001}`, `|D_q z^2 - 4|` is
  monotone decreasing at `z=2`. If not, G2 fails.
- **G3 honesty.** `continuum_claimed_from_q_limit` is `false`.
- **G4 parity.** torch / jax bit-identical on G1.

## 9. Benchmark plan

- `benchmarks/q_ombu_timescale.py`
- Smoke: `docs/benchmarks/q_ombu_timescale_smoke.json`
- `--full`: `$OMNIBIAS_SCRATCH/inventions/q_ombu/`

## 10. Honesty and scope

- Named limits are not bias collapse and not temperature collapse.
- Not CCF. Not a continuum theorem.
- Certificate tier: empirical.

## 11. Open questions and risks

- **q=1** is a removable singularity; implementations must use
  `q_derivative_limit`, not divide by zero.
- **Falsifier.** G2 non-monotone residual as `q -> 1`.

## 12. Implementation checklist

- [x] Hybrid wrappers in qcalculus / timescale
- [x] Limit-residual tests
- [x] `benchmarks/q_ombu_timescale.py` plus smoke JSON
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
