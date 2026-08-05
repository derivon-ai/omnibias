# omnibias-formal

**Status: Alpha (0.1.0a1).**

Mathlib-backed formal checking for omnibias certificates. This package drives the
Lean project [`formal/omnibias-analytic`](../../formal/omnibias-analytic) to
discharge a certificate's finite obligations against Mathlib, and reports a
**`mathlib_verified`** tier.

## Two trust tiers, never conflated

- **`theorem_prover_verified`** (existing, `omnibias.core.proof.lean_check`) is
  earned by the tiny, hand-auditable, **Mathlib-free** kernel
  (`formal/omnibias-verified-kernel`) over integer `ZInterval` arithmetic.
- **`mathlib_verified`** (this package) is earned by the **Mathlib-backed**
  project over `ℚ` / `ℝ`. It is a larger, honestly-labeled trust base. It never
  sets `theorem_prover_verified`, and a green build never implies `unproven_claim`.

## Phase 0 scope

The enclosed-quantity **sign** obligation over `ℚ`: a v1 `interval` certificate
with a strictly positive lower endpoint (or strictly negative upper endpoint) is
turned into a Lean theorem discharged by the project's proven `enclosed_pos` /
`enclosed_neg` lemmas, emitting the rational endpoint directly (no integer-scaling
hack) and closing the endpoint sign with `norm_num`.

Sum-of-squares / `positivity` / `nlinarith` obligations, real-interval and PDE
margins, and the faithful global-regularity statement restatements are later phases.

## Usage

```python
from omnibias.core.proof.certificate import interval_certificate
from omnibias.core.verified.interval import Interval
from omnibias.formal import check_certificate, mathlib_check_available

cert = interval_certificate("H ω₀(0)", Interval(0.5, 2.0))
result = check_certificate(cert)   # available=False (verified=False) without a Lean toolchain
```

`check_certificate` is tamper-evident (a mismatched `digest` is rejected before
any Lean is emitted) and degrades gracefully: with no `lake` on PATH it returns
`available=False`, `verified=False`, and never raises.

## Tests

```bash
python -m pytest packages/omnibias-formal/tests -q
```

The fast suite runs **without** a Lean toolchain (obligation generation, tamper,
and graceful-degradation paths). The end-to-end `lake build` assertion is gated
behind `mathlib_check_available()` and runs in the dedicated `lean-analytic` CI
job.

## License

Dual-licensed: AGPL-3.0-or-later OR a commercial licence from Derivon
(`LicenseRef-omnibias-Commercial`). See [`LICENSE`](LICENSE),
[`../../LICENSING.md`](../../LICENSING.md), and
[`../../COMMERCIAL-LICENSE.md`](../../COMMERCIAL-LICENSE.md). Contact
info@derivon.ai for commercial terms.
