# Hilbert XVI: the machine-checked obligation ledger

Turning "Hilbert's 16th problem" into a stampable flag is exactly the wrong
shape for a repository that must stay honest for years. Instead,
`omnibias.dynamics.hilbert16_ledger` follows the discipline already proven
out by
[`omnibias.core.proof.obligations.convergence_ledger`](../omnibias-core/src/omnibias/core/proof/obligations/convergence_ledger.py):
every piece of the problem this repository or the literature actually
addresses is one explicit, dated `H16Obligation`, and the parent claims are
**derived** from that entry set, never asserted by hand.

## 1. What one obligation looks like

```python
@dataclass(frozen=True)
class H16Obligation:
    name: str
    statement: str
    status: H16Status  # BLOCKED | CONDITIONAL | DISCHARGED_LOCAL_SCOPE | DISCHARGED
    external_premises: tuple[str, ...] = ()
    evidence_digest: str = ""
    source: str = ""
```

An entry `fully_discharged` only when `status == "DISCHARGED"` **and**
`external_premises` is empty -- a `CONDITIONAL` entry that leans on an
external published theorem, or a `DISCHARGED` entry that still cites one,
never counts.

## 2. The four kinds of entry the shipped ledger carries

- **The six research gates `G1`-`G6`** from
  [HILBERT16-PROGRAM.md](HILBERT16-PROGRAM.md), each `BLOCKED` with the
  concrete obstruction recorded as an `external_premises` string (for `G1`,
  the exact counterexample: `X'=-eps*s*X, Y'=r*Y` admits no fixed decay
  exponent uniformly bounding the kappa-sensitivity as `s/r -> 0`).
- **Fifteen dated Design-Roussarie-Rousseau (DRR) cases**
  (`I_2^1`, `I_4^1`, `I_12^1`, `I_13^1`, `I_14^1`, `I_6b^1`, `H_13^3`,
  `DI_2b`, `H_14^3`, `DF_1a`, `DF_1b`, `DF_2a`, `DF_2b`, `DH_1`, `DH_2`),
  each carrying its literature source. Cases resting on a genuinely
  published finite-cyclicity theorem (e.g. Rousseau-Shan-Zhu 2016 for
  `I_12^1`/`I_13^1`) are `CONDITIONAL` on that external theorem, never
  independently replayed by this repository's Lean kernel; cases the
  literature itself leaves open (Huzak 2018's `DF_1b`/`DF_2b`/`DH_1`/`DH_2`)
  or only preprint-claimed (`H_14^3`) are `BLOCKED`.
- **The Part-A 22-oval octic target**, `BLOCKED`: the shipped 16-oval baseline
  is an unnested grid-polynomial verifier demo, not progress toward 22; the
  target scheme remains open.
- **This plan's genuinely new, strictly local results** --
  `local_resonant_normal_form`, `local_bautin_basis`,
  `local_bautin_stabilization_barrier`,
  `local_songling_lower_bound_audit`,
  `local_part_a_polygon_sos`,
  `local_collar_membership`, `local_entry_exit_product`,
  `local_weighted_section`, `local_quasihomogeneous_dichotomy`,
  `local_ln_format_barrier`, `local_abelian_return_transfer`,
  `local_fold_imap`, `local_shrinking_root_imap`,
  `local_canonical_zeta`, `local_fold_zeta`, `local_physical_c2`,
  `local_z_x_gap`, `local_z_v_bound`, `local_z_slow_v`, `local_fold_z_x`, `local_stage_b`, `local_stage_a`, `local_chi_b`, `local_dx_e_leading`, `local_dx_e_unif`, `local_stage_c`, `local_stage_c_exit`, `local_stage_c_th`, `local_stage_c_gap`, `local_stage_c_env`, `local_stage_c_if`, `local_stage_c_int`, `local_stage_c_lo`, `local_stage_c_k`, `local_stage_c_boot`, `local_stage_c_rect`, `local_stage_c_hit`, `local_stage_c_sec`, `local_stage_c_oneshot`, `local_stage_c_oneshot_eps`, `local_stage_c_eps_span`, `local_stage_c_origin`, `local_stage_c_origin_span`, `local_stage_c_origin_iface`, `local_stage_c_origin_near`, `local_stage_c_origin_x32`, `local_stage_c_compare`, `local_stage_c_uniform`, `local_stage_c_interface`, `local_sep_spre`, `local_dx_e_off`, `local_dx_e_ray`, `local_dx_e_near`, `local_dx_e_open`, `local_outgoing_corridor`, `local_post_corridor`,
  `local_height_envelope`, `local_q_ratio_c2`,
  `local_k_zeta_remainder`, `local_kill_zeta`,   `local_cancelled_n`,
  `local_height_mix`, `local_orbit_th`, `local_th_integral`,
  `local_vh_orbit`, `local_e_out_section`,   `local_e_out_eps`,
  `local_e_out_speed`, `local_e_sigma_speed`,   `local_e_sigma_in`,
  `local_e_sigma_hit`, `local_e_sigma_from0`, `local_e_sigma_unif`,
  `local_e_sigma_wall`, `local_e_sigma_box`, `local_e_sigma_span`,
  `local_e_sigma_pack`, `local_e_sigma_eps`,   `local_e_sigma_oneshot`,
  `local_e_sigma_oneshot_eps`, `local_e_sigma_eps_span`,
  `local_e_sigma_eps_lo` -- each
  `DISCHARGED_LOCAL_SCOPE` with *empty*
  `external_premises`: they are real, sealed, machine-checked wins on one
  declared instance, but `DISCHARGED_LOCAL_SCOPE` is a status distinct from
  `DISCHARGED` specifically so a local win can never leak into a parent
  claim.

## 3. Aggregate status vs. derived parent flags

`check_ledger` computes the coarse aggregate:

```text
BLOCKED      if any entry is BLOCKED
CONDITIONAL  else if any entry is CONDITIONAL, DISCHARGED-with-premises,
                or DISCHARGED_LOCAL_SCOPE
PROVED       only if every entry is DISCHARGED with no external premises
```

and returns the exact list of blocking entries, so a benchmark or a caller
can print precisely what remains rather than a single opaque flag.

`derived_parent_flags` is stricter and is the only place the three parent
claims are computed:

```python
gates_ok   = all(e.fully_discharged for e in ledger.gates())
cases_ok   = all(e.fully_discharged for e in ledger.drr_cases())
part_a_ok  = all(e.fully_discharged for e in ledger.part_a())
part_b_ok  = gates_ok and cases_ok
full       = part_a_ok and part_b_ok
```

`payload_earns_parent_claim(payload, key)` re-derives a flag from a stored
payload's own `entries` rather than trusting a stored boolean, so a
serialized certificate cannot forge `full_hilbert16_solved` by editing the
top-level flag field alone -- the regression suite checks this directly by
flipping a stored flag to `True` by hand and confirming the derived replay
still reports `False`.

## 4. The certificate

`certify_h16_ledger` seals one `H16LedgerCertificate` per ledger snapshot:
the ledger's own content digest, the `check_ledger` result, and the derived
parent flags, all inside the honesty block alongside the boilerplate
`dulac_truncated_model_only` / `physical_return_membership_proved` /
`graphic_finite_cyclicity_proved` / `drr_case_closed` flags (every one
`False`). `verify_h16_ledger` replays `certify_h16_ledger` from the stored
ledger and checks digest equality plus the certificate seal -- it does not
trust any stored derived field.

## 5. What this is not

This ledger does not attempt, and cannot be made to attempt, an actual
proof of any BLOCKED or CONDITIONAL entry. `full_hilbert16_solved` is
`False` on every ledger this module ships, by construction, and a
dedicated regression test exists specifically to catch the day someone
accidentally makes that stop being true (e.g. by flipping a status without
discharging the underlying mathematics). Nothing in this module or its
certificate makes any claim about `H(2) < \infty`, a Roussarie graphic's
finite cyclicity, or a DRR case's closure -- those all remain exactly as
open as [HILBERT16-PROGRAM.md](HILBERT16-PROGRAM.md) and
[HILBERT16-DULAC-CYCLICITY.md](HILBERT16-DULAC-CYCLICITY.md) already
record.

## 6. Reproduction

```bash
python -m pytest packages/omnibias-dynamics/tests/test_hilbert16_ledger.py -q
python -m benchmarks.hilbert16_ledger
```

The smoke artifact is `docs/benchmarks/hilbert16_ledger_smoke.json`; its
four gates use the ``gh`` prefix so they do not collide with research gates
``G1``-``G6``:

- GH1: the shipped ledger is open (six gates, fifteen DRR cases, Part A, and
  the three local-scope entries all present; every parent flag `False`);
- GH2: certificate round-trip (`certify_h16_ledger` / `verify_h16_ledger`
  agree, and `payload_earns_parent_claim` matches the derived flags);
- GH3: a tampered stored flag is rejected by the derived replay;
- GH4: a synthetic, fully-discharged ledger genuinely earns
  `full_hilbert16_solved = True`, proving the derivation path is not
  vacuously always-false.
