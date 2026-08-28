---
name: omnibias-certificate-lean
description: Work on omnibias certificates and the Mathlib-free Lean loop — the hash-sealed v1 certificate format, the finite-obligation bridge, and the ZInterval kernel. Use when sealing or verifying certificates or wiring theorem_prover_verified from a genuine lake build.
---

# Certificates and the Mathlib-free kernel

omnibias runs one derivative tower in three registers — differentiable,
rigorous, and **formal**. This skill owns the tiny Mathlib-free kernel and
`theorem_prover_verified`.

## Why nested AD fails

A float residual is not a kernel proof. JSON that self-attests
`theorem_prover_verified` is forgeable. Generic proof assistants pulled into
CI with Mathlib are too heavy for the fast path. Nested AD has no finite
rational obligation at all.

## What only this tower unlocks

Hash-sealed certificate v1; a bridge that extracts a *finite, rational*
obligation (sign, spectral-gap positivity, PD inertia, stencil poisedness,
interval replay trace, domain-subdivision coverage); Lean that chains proven
`ZInterval` lemmas; `lake build` on `formal/omnibias-verified-kernel` (Lean 4
core only, sorry-free). With no toolchain the flag stays `False` and nothing
errors.

Sibling: `omnibias-formal-agent` owns Mathlib-backed `mathlib_verified`.

## Use

- Certificate format: `omnibias.core.proof.certificate`
  (`verify_certificate_digest`).
- Bridge: `omnibias.core.proof.lean_check`.
- Kernel: `formal/omnibias-verified-kernel`.
- Replay / subdivision: `omnibias.core.proof.replay` → `Omnibias/Replay.lean`,
  `Omnibias/Subdivision.lean`. Docs: `docs/api/lean_replay.md`.
- Proof engine: `omnibias.core.proof.engine`. `prove(..., lean_check=True)`
  fires Lean when `generate_obligation` already applies.

`theorem_prover_verified` is set only on a genuine kernel `lake build`.
Verdict carries `certificate_schema_version`. For small finite maps use
kernel-reduced `by decide`. Keep the kernel Mathlib-free; prove more by
feeding more finite reductions.

```bash
python -m pytest packages/omnibias-core/tests -q -k "certificate or lean or proof"
# where a Lean toolchain exists:
cd formal/omnibias-verified-kernel && lake build
```

## Extend

Compose with `omnibias-formal-agent`, `omnibias-verify`, `omnibias-sos`,
`omnibias-formal`. New sealed payloads round-trip `verify_certificate_digest`.
Test the well-formed path and graceful degradation without Lean.

## Next invention

A new finite obligation class (replay vocabulary or subdivision shape) that
the kernel checks with `by decide` and that a shipped certificate actually
emits.

## Further references

- `docs/scope-and-guarantees.md`
- `docs/api/lean_replay.md`
