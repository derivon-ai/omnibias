---
name: omnibias-dev-certificate-lean
description: Work on omnibias certificates and the Lean formal loop -- the hash-sealed v1 certificate format, the finite-obligation bridge, and the Mathlib-free ZInterval kernel. Use when sealing or verifying certificates or touching theorem_prover_verified, which is earned only by a genuine lake build pass and must never be forged. For contributors modifying omnibias itself, not for consumers using it.
---

# Certificates and the formal loop

omnibias runs one derivative tower in three registers -- differentiable,
rigorous, and **formal**. The formal register's credibility depends entirely on
never overstating what was proven.

## Where it lives

- Certificate format v1: `omnibias.core.proof.certificate` -- canonical, hash-sealed JSON for `Interval` / `TaylorModel` enclosures; tamper-evident via `verify_certificate_digest`.
- Formal bridge: `omnibias.core.proof.lean_check` -- extracts a certificate's finite, rational obligation (spectral-gap positivity, enclosed-quantity sign, PD inertia vector), emits Lean chaining the kernel's proven `ZInterval` lemmas, and runs `lake build`.
- The kernel: `formal/omnibias-verified-kernel` (Lean 4 core, **Mathlib-free** so CI kernel-checks it cheaply).

## The rules you must not break

- `theorem_prover_verified` is set **only** on a genuine kernel `lake build` pass. It can never be forged by the certificate, and asserting it without a pass must block the verdict. With no Lean toolchain the bridge **degrades gracefully** -- the flag stays `False`, nothing errors.
- `Verdict` carries `certificate_schema_version` and the kernel-earned `theorem_prover_verified`; keep both honest.
- The kernel is deliberately Mathlib-free. **Do not widen the kernel's trust base.** The way to "prove more" is more finite reductions feeding the kernel, never trusting a bigger axiom set.
- Infinite / analytic obligations (limits, continuum statements, asymptotics) are **out of scope** and are not expressed in Lean at all. Do not add them; an obligation the kernel cannot decide does not belong here.

## Checklist

- Any new sealed payload is tamper-evident (round-trips through `verify_certificate_digest`) and carries the schema version.
- Test both paths: a well-formed obligation, and graceful degradation when no Lean toolchain is present.

```bash
python -m pytest packages/omnibias-core/tests -q -k "certificate or lean or proof"
# kernel check (only where a Lean toolchain exists):
cd formal/omnibias-verified-kernel && lake build
```
