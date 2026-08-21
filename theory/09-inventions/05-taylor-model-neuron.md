# 09-05 Taylor-model neuron

## 1. Thesis and status

A unit whose output is a **`TaylorModel` / interval**, not a float, so
the forward pass is the same algebraic object as
`omnibias.core.verified`.

- **Status**: gated
- **Depends on**: 01-10, 09-01
- **Blocks**: 09-24

### Operator card

- **Benefit.** Hidden state carries a remainder enclosure.
- **How it works.** Affine image of a `TaylorModel`; `sigma` is applied
  via a polynomial part plus an interval remainder (existing TM
  arithmetic). Depth is composition of TMs.
- **Strength.** Tiny monotone nets where boxes stay non-vacuous.
- **When to use.** First-bet after FTC / jet-token; pair with 09-24.
- **When not.** Deep residual nets; CCF; as a replacement for float
  PINNs.
- **Accuracy floor.** Enclosure explosion past shallow depth.

## 2. Where it lands

Pure-Python construction in `omnibias.core.verified` (already hosts
`TaylorModel`). Thin torch / jax adapters that wrap float midpoints for
training *if* a hybrid mode is needed. No new package.

## 3. Prior art in omnibias

- `omnibias.core.verified.taylor_model.TaylorModel`
- `omnibias.core.verified.taylor_model_mv.TaylorModelMV`
- `omnibias.verify._core.taylor.taylor_output_bounds`
- `omnibias.verify._core.certificates.lipschitz_bound`
- Spec 08-09 — certifies a *step*, not a TM hidden state.

**Confirmed gap.** No layer API whose activations *are* `TaylorModel`
objects.

## 4. Mathematics

On a box `X`, `f(x) = p(x-x0) + I` with `f(X) subset p(X-x0) + I`.
Composition uses TM arithmetic already in the core. `sigma` for
sigmoid / tanh uses the closed-form tower for the polynomial part
(bias collapse coefficients) and an interval bound on the remainder.

No temperature collapse. Error is the TM remainder `I`, which must
contain the true range (soundness), not be tight.

## 5. Worked example

`sigma = sigmoid`, `X = [0, 0.2]`, order-1 TM about `0.1`.
`sigmoid(0.1) ≈ 0.524979`, `sigmoid'(0.1) ≈ 0.249376`. A sound TM must
enclose `sigmoid([0,0.2]) subset [0.5, 0.549834]` (true endpoints).
The implementer checks `I` contains the range of the remainder
`sigmoid(x) - (s + s'(x-0.1))` on `X`. Width of `I` is recorded; G2
fails if the enclosure is vacuous (`I` contains `[-1e6, 1e6]`).

## 6. Proposed API

Does not exist yet. Core is pure Python. Adapters use default dtype.

```python
# omnibias/core/verified/tm_neuron.py — proposed
@dataclass(frozen=True)
class TMNeuronSpec:
    order: int = 1
    activation: str = "sigmoid"

def tm_dense(tm_in: TaylorModel, w: float, b: float, spec: TMNeuronSpec) -> TaylorModel:
    """Affine + TM-sigma. Must not import torch/jax."""
```

## 7. Practical use cases

1. **1-unit certified bound** on `sigmoid(wx+b)` over a box (vs
   `taylor_output_bounds` on a float net).
2. **Two-layer** TM nest for 09-24.
3. **Not** ImageNet-scale enclosures.

## 8. Acceptance gates

- **G1 soundness.** Worked example: the TM range contains a dense grid
  of true `sigmoid` values and a random sample (core verified contract).
- **G2 non-vacuous.** `width(I) < 0.1` on that box.
- **G3 depth honesty.** A 6-layer random TM nest is reported; if
  `width(I) > 1e3`, the artifact field `enclosure_exploded` is `true`
  (allowed; not a pass of a depth claim).
- **G4 no backend leak.** The core module imports neither torch nor jax.

## 9. Benchmark plan

- `benchmarks/taylor_model_neuron.py`
- Smoke: `docs/benchmarks/taylor_model_neuron_smoke.json`
- `--full`: `$OMNIBIAS_SCRATCH/inventions/tm_neuron/`

## 10. Honesty and scope

- Sound enclosure tier. `theorem_prover_verified` is **not** asserted.
- Bias collapse supplies polynomial coefficients of `sigma^(n)`.
- Not 08-09. Not CCF. Not a deep-net certificate.
- Navier–Stokes stays external.

## 11. Open questions and risks

- **Explosion** is the predicted failure. G3 records it.
- **Falsifier.** G2 failure on the one-unit box means TM-sigma is not
  ready even at depth 1.

## 12. Implementation checklist

- [x] `omnibias.core.verified` TM-neuron helpers
- [x] Soundness tests (grid + sample)
- [x] `benchmarks/taylor_model_neuron.py` plus smoke JSON
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
