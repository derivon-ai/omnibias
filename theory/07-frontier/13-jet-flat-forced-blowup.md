# 07-13 Jet-flat forced concentrating field

## 1. Thesis and status

On the locked 07-09 scales there is an axis-regular polynomial swirl
whose regularized leading stress \(T_0\) vanishes at the axis after one
exact-\(\mathbb{Q}\) correction, with core \(\|u\|_\infty \asymp
\tau^{-A}\) and core energy \(O(\tau^{1/2-3h})\to 0\). Force is that
residual times a mollifier cutoff; a from-rest ramp is \(0\) at
\(t=0\). This is a different, weaker object than Clay (C)/(D).

- **Status**: shipped (G1–G6 CI; founding bias collapse, not temperature collapse; not Clay (A)/(B); not a forced-blowup reproof)
- **Depends on**: 07-09, 01-05
- **Blocks**: none

## 2. Where it lands

`omnibias.pinn.certified.forced_flat` in `omnibias-pinn`. No new
package: same domain and audience as 07-09. Pure Python plus
`Fraction`; no torch/jax. Do not grow
`omnibias.pinn.certified.navier_stokes`. Do not switch the headline
plant to 1-D Burgers: the 3-D anisotropic volume is why energy stays
bounded.

## 3. Prior art in omnibias

- `omnibias.pinn.certified.anisotropic` — `SimilarityScales`,
  `AxisRegularProfile`, `apply_T_b`, `DX`. Linear swirl \(F=c(1+aX)\);
  no \(X^2\) coefficient and no paper (4.8)–(4.11) \(T_0\).
- `omnibias.core.pulse_envelope` / `omnibias.core.mollifier.tail_bound`
  — occupancy ramp and certified cutoff tail (01-05, 07-12).
- `omnibias.core.proof.obligations.stress_cone` — cone membership, not
  this axis jet.

**Confirmed gap.** An explicit concentrating field with a leading-order-
flat force, bounded core energy, and a from-rest ramp was not locked.

## 4. Mathematics

Locked scales \(h=1/200\), \(A=1/2+h\), \(D=1/2-h\). Family
\(F=c(1+aX+bX^2)\) with the 07-09 \(U=\eta X\) and \(\Pi=\int F^2\).
Paper (4.8)–(4.11): \(H=2XF\), \(l=D_X\log H=1+XF_X/F\), sources
\(S_q,S_n\) with \(C_Q=C_N=0\), so \(Q_s(0)=S_q(0)/2\) and
\(N_s(0)=S_n(0)\). Then \(T_0=F(p_s-s)\) with
\(p_s=(XQ_s/L,\,XN_s/(LE))\) and \(s=(a,-b_s)\).

At the axis, \(W=1\), \(l=1\), \(H_c=0\) give \(S_q=-(1+h)\),
\(S_n=0\). Raw \(T_0(0,0)=(0,0)\) because of factors of \(X\). The
load-bearing object is the regularized jet
\[
\lim_{X\to 0} T_{r\theta}/X = FQ_s/L + 2F_X = c\bigl(2a-(1+h)/2\bigr),
\qquad
\lim_{X\to 0} T_{rz}/\sqrt{2X} = N_s/(2L).
\]
The \(X^2\) coefficient \(b\) does not enter. The named extra
coefficient is the linear slope \(a=(1+h)/4\), which zeros the jet over
\(\mathbb{Q}\). Uncorrected \(a=1\) is the negative control
\(599/400\neq 0\).

Core \(\|u\|_\infty\asymp\tau^{-A}\): the \(\tau\)-independent
prefactor \(F(X_c)\) at \(X_c=1\) is exact; two locked \(\tau\) share
it. Core energy on the similarity box \([0,1]\times[-1,1]\) uses the
volume element and \(|u|^2\sim q^{-2A}\) to produce the paper’s
\(\tau^{1/2-3h}\) power; the prefactor
\(\int_0^1\int_{-1}^1 2XF^2\,d\eta\,dX\) is exact over \(\mathbb{Q}\).
The 3-D anisotropic volume is why this tends to \(0\); 1-D viscous
Burgers is the wrong plant.

Cutoff: `mollifier.tail_bound` contains `true_outside_mass`. From-rest
ramp: occupancy \(P=st\) with \(s=0\) at \(t=0\). Remaining \(q\)-powers
of the residual and \(C^\infty\) extension of the force through \(t=1\)
stay leftover.

This is founding **bias-collapse** arithmetic (exact identities). It is
not temperature collapse.

## 5. Worked example

\(h=1/200\), \(c=1\), uncorrected \(a=1\), \(b=0\). Axis jet
\((599/400,\,0)\). Corrected \(a=201/800\) gives \((0,0)\). Energy
prefactor of the uncorrected plant is \(17/3\); exponent
\(97/200>0\). Locked \(\tau=1\) and \(\tau=1/4\) share the
\(\|u\|_\infty\) prefactor \(F(1)=2\). From-rest ramp value \(0\);
on-window occupancy \((s,t)=(1,1)\) has value \(1\).

## 6. Proposed API

```
JetFlatProfile(h, c, a, b)
leading_stress_T0(profile, X, eta) -> (T_rtheta, T_rz)
axis_T0(profile) -> regularized T0 jet at (0,0)
correct_axis_stress(profile) -> (corrected_profile, a)
core_linfty_scale(tau) -> tau^{-A} monomial
core_energy_scale(tau) -> tau^{1/2-3h} * exact prefactor
forced_field(tau, cutoff, ramp) -> honesty payload + scales
```

Honesty: `forced_blowup_reproof_claim=False`. No torch/jax in this
module. Catalog kind `jet_flat_forced_blowup`, parent
`"Navier-Stokes forced blowup (Clay C/D)"`,
`parent_status="already_true"`.

## 7. Practical use cases

- Lock a concentrating blowup with a leading-order-flat force as finite
  identities, without the paper’s joining and pulse cycle.
- Use the uncorrected axis jet as a named negative control.
- Keep energy bounded by staying on the 3-D anisotropic volume.

## 8. Acceptance gates

- **G1.** Two locked \(\tau\): \(\|u\|_\infty\) prefactor identical;
  exponent \(-A\). The ratio is \((\tau_1/\tau_2)^{-A}\) as a monomial
  identity, not an evaluated irrational.
- **G2.** Core energy prefactor exact; exponent \(1/2-3h>0\), so the
  monomial decreases as \(\tau\downarrow 0\).
- **G3.** Uncorrected `axis_T0` is nonzero (named negative control).
- **G4.** Corrected `axis_T0=(0,0)` over \(\mathbb{Q}\).
- **G5.** Mollifier tail contains `true_outside_mass`; ramp vanishes at
  \(t=0\).
- **G6.** Honesty flags false; leftover named: \(C^\infty\) through
  \(t=1\), pulses, joining, uniqueness, remaining \(q\)-powers.

## 9. Benchmark plan

`benchmarks/forced_flat_blowup.py` writes
`docs/benchmarks/forced_flat_blowup_smoke.json`. `--full` writes under
`$OMNIBIAS_SCRATCH` (default `artifacts/`). CI runs the smoke.

## 10. Honesty and scope

Not Clay (C)/(D): the force need not extend smoothly through the
singular time. Not a second proof of the OpenAI theorem. Unforced
(A)/(B) stays external. Not a CCF residual. Not a continuum regularity
claim. `forced_blowup_reproof_claim=False`.

## 11. Open questions and risks

- Off-axis \(T_0\) needs the integral (4.10); leftover.
- A 1-parameter solve in \(b\) cannot hit the axis jet; the extra
  coefficient is \(a\). Falsifier: corrected jet nonzero over
  \(\mathbb{Q}\).
- Remaining \(q\)-powers, uniqueness, and \(C^\infty\) through \(t=1\)
  stay leftover.

## 12. Implementation checklist

- [x] `omnibias.pinn.certified.forced_flat`
- [x] Tests + smoke JSON
- [x] Docs page and mkdocs nav entry
- [x] CI job
- [x] Index row in `theory/README.md`

## 13. Parent problem and the exact reason it stays an external obligation

**Parent: Navier-Stokes forced blowup (Clay C/D)**, already true in the
world. This spec ships a **different weaker object**: a concentrating
field with bounded core energy and a leading-order-flat force, written
as finite identities. It is not a second proof of Clay (C)/(D) and does
not reproduce the OpenAI argument. Unforced Navier-Stokes regularity
(Clay A/B) stays an external obligation.
