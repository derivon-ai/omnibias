# Formal verification used by the Hilbert-sixteenth argument

The checked statements below support the written passage arguments. The
actual physical hypotheses and uniform singular remainders are proved in
the linked analytic notes and have not been formalized. None of these
builds verifies full graphic coverage or Hilbert XVI, and the curve/surface
classification is a separate program.

| Checked source | What its Lean theorem establishes | What must still connect it to the physical count |
| --- | --- | --- |
| [Hilbert16Rolle.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16Rolle.lean) | An actual derivative with at most one or two zeros excludes three or four ordered function zeros. A strictly negative derivative at every zero implies at most one zero. | The actual fixed-field displacement, its derivative hypotheses, and its connected admission interval |
| [Hilbert16Parabola.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16Parabola.lean) | Fourteen exact algebraic statements, including the physical field restriction, rational Darboux identity, cubic-speed/cofactor cancellation, and signed endpoint cancellation | Existence and capture of the regular passage, uniform tail estimates, matched-endpoint sensitivity, and the sharp multiplier estimate |
| [Hilbert16Resonance.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16Resonance.lean) | Seven abstract statements: strict-convex zero exclusion, positive-factor zero equivalence, the joint-interval three-zero implication and its positive-curvature specialization, a curvature error budget, an increasing-core whole-interval count, and construction of an increasing core from a nonpositive convex gap | The actual singular and regular derivative estimates; continuity and connected admission; selection of the clipped sublevel interval and its outer negative-at-zero comparisons |
| [Hilbert16ReturnMap.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16ReturnMap.lean) | Eight conditional real-variable implications: exponential normalization, the actual weighted derivative, first and mixed event-time identities, first/second derivative zero bounds, interval-sign transfer, and non-isolation of an identity's interior zeros | Actual differentiability, event existence and uniqueness, a nonzero event normal velocity, and the asserted derivative enclosures; no general passage-class membership theorem is inferred |
| [Hilbert16Scale.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16Scale.lean) | Eight statements about the actual exponential parameter derivative and diagonal limit, fixed-product paths and their derivatives, the second weighted polynomial expression, cancellation, and a pole margin | The directed primitive remainder bounds and membership of actual singular passages in the represented class; physical uniformity is additional |
| [Hilbert16ChiScale.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16ChiScale.lean) | Seven exact linear-saddle statements: the χ identification, χ and kappa derivatives, scaled-kappa and reciprocal-separation identities, the sensitivity ratio, and the frozen-exponent obstruction | Membership of an actual quadratic passage in the linear model; uniform physical remainders; G1 and G4 |
| [Hilbert16SaddleNode.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16SaddleNode.lean) | Six exact identities: the double-root quadratic, the wall at the equilibrium, vanishing-separation linear exit, vanishing χ, the `sigma * kappa` factorization on a χ-locus, and the shrinking-root product | Physical C2 remainders, the fold-scale tension, outgoing continuation as `r1 -> 0`, G1 and G4 |
| [Hilbert16TwoBlowup.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16TwoBlowup.lean) | Four exact W-coordinate identities: `epsilon * W = h^epsilon`, the outgoing W-ratio, `d log W / d tau = -u`, and the two-scale product | Physical C2 remainders, a covering of `sep = exp(-1/epsilon^2)`, G1 and G4 |
| [Hilbert16ScaleDichotomy.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16ScaleDichotomy.lean) | Nine exact identities: blow-up height and height ratio, fold-scale `epsilon^4`, the affine leading event exponent with first derivative and vanishing second difference, the joint-axis sum and exclusion, and `epsilon log(1/sep) = 1/epsilon` on the kill sequence | Physical C2 remainders, a third compact scale, G1 and G4 |
| [Hilbert16LNCell.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16LNCell.lean) | Six elementary statements: first and iterated logarithmic derivatives of natural-power monomials, two explicitly conditional order-two Cauchy consequences, and a log-strip perturbation margin | Holomorphic extension and sup bounds for the actual map, physical LN/exp membership, first-hit completeness, overlap matching, G1 and G4 |
| [Hilbert16EntryExit.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16EntryExit.lean) | Slow-line two-root and double-root partial fractions, height-dominated `dx/dy`, the tracked `sep^2` × outgoing-factor logarithm, and the kill-sequence sign | Physical C2 remainders, `sep = 0` remainder, chart O, complete first-hit, G1 and G4 |
| [Hilbert16StageB.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16StageB.lean) | Kill-line `r1=1-sep/2`; naive `|Delta x|` majorant; tracked exponent `7/4` at `eps=1/16` | Stage A `dx_e/dkappa`, Stage C, C2, first-hit, G1 and G4 |
| [Hilbert16StageA.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16StageA.lean) | Kill-line `B_-(a)=theta(1+theta)sep^2`; right-wall slope; worst-case `a=3/8` at `sep=1` | `dx_e/dkappa`, Stage C, first-hit, G1 and G4 |
| [Hilbert16ChiB.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16ChiB.lean) | `(1+theta)/theta=9`; decay `c=1/16`; worst-case `(K+1)/r1=8` | uniform-in-`chi` `dx_e`, Stage C, first-hit, G1 and G4 |
| [Hilbert16DxELeading.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16DxELeading.lean) | Prefactor `3/8`; slope half `3/8`; threshold net `3/16` | Stage C, first-hit, G1 and G4 |
| [Hilbert16DxEUnif.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16DxEUnif.lean) | Extra coefficient `3/32`; written `c=1/16` weaker by `1/32`; lift `27/32` | Stage C orbit, first-hit, `dx_e` off the kill line, G1 and G4 |
| [Hilbert16StageC.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16StageC.lean) | Written `a_min=1/2`; integrating factor `4`; declared floor `1/4` | outgoing orbit, first-hit, C2, G1 and G4 |
| [Hilbert16StageCExit.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16StageCExit.lean) | `T_e/eps^2=1/8` at `x=1/2`; `eps y0=1/256`; gap room `31/256` | outgoing orbit, first-hit, C2, G1 and G4 |
| [Hilbert16StageCTh.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16StageCTh.lean) | Midpoint product `-sep^2/4`; worst `T_h=3/4`; room `1/4` | outgoing orbit, first-hit, C2, G1 and G4 |
| [Hilbert16StageCGap.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16StageCGap.lean) | Wall `1/8-1/16=1/16`; `h_1` cube `1/4096`; exact wall `T-h=1/4096` | outgoing orbit, first-hit, C2, G1 and G4 |
| [Hilbert16StageCEnv.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16StageCEnv.lean) | Wall `T_e/eps^2=1/8` below 1; declared `c=1/32`; `c+room=1` | C≠0 orbit, first-hit, C2, G1 and G4 |
| [Hilbert16StageCIf.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16StageCIf.lean) | `2*3=6`; `2*6=12`; edge `12(1/4-1/16)=9/4` | first-hit, C2, G1 and G4 |
| [Hilbert16StageCInt.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16StageCInt.lean) | `C eps=1/8`; `1-alpha=7/8`; slope `16/7` | lower envelope, first-hit, C2, G1 and G4 |
| [Hilbert16StageCLo.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16StageCLo.lean) | `1/2-1/32=15/32`; `1+1/16=17/16`; wrapping `15/512` | tight ratio, first-hit, C2, G1 and G4 |
| [Hilbert16StageCK.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16StageCK.lean) | `6*(1/16)=3/8`; `3*2=6`; `6-16/7=26/7` | `T-h=O(eps)`, first-hit, C2, G1 and G4 |
| [Hilbert16StageCBoot.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16StageCBoot.lean) | `1+6=7`; `2*6+3=15`; `2*7*3=42` | first-hit, C2, G1 and G4 |
| [Hilbert16StageCRect.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16StageCRect.lean) | `1+1=2`; `2*2=4`; `(1/16)*(1/4)=1/64` | first-hit of the large section, C2, G1 and G4 |
| [Hilbert16StageCHit.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16StageCHit.lean) | `64*(1/64)=1`; `64*3=192`; `1-1/4096=4095/4096` | Lohner, signed-label section, chart O, C2, G1 and G4 |
| [Hilbert16StageCSec.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16StageCSec.lean) | `1/4-1/12=1/6`; `(1/2)/2=1/4`; `1/64+1/256=5/256` | Lohner, chart O, C2, G1 and G4 |
| [Hilbert16StageCOneshot.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16StageCOneshot.lean) | `(1/4)/(1/16)=4`; `120*(1/20)=6`; `80*(1/20)=4` | every `eps`, chart O, C2, G1 and G4 |
| [Hilbert16StageCOneshotEps.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16StageCOneshotEps.lean) | `(1/4)/(1/20)=5`; `(1/4)/(1/25)=25/4`; `160*(1/20)=8` | every `eps`, chart O, C2, G1 and G4 |
| [Hilbert16StageCEpsSpan.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16StageCEpsSpan.lean) | `23/400+1/200=1/16`; `4*(1/800)=1/200`; `800*(1/16)=50` | every `eps`, chart O, C2, G1 and G4 |
| [Hilbert16StageCOrigin.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16StageCOrigin.lean) | `1-2/2=0`; `1-(3/2)/2=1/4`; `1-(7/4)/2=1/8` | every `r1`, complete first-hit on chart O, C2, G1 and G4 |
| [Hilbert16StageCOriginSpan.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16StageCOriginSpan.lean) | `3/2+1/2=2`; `8*(1/16)=1/2`; `16*(1/2)=8` | every `r1`, complete first-hit on chart O, C2, G1 and G4 |
| [Hilbert16StageCOriginIface.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16StageCOriginIface.lean) | `7/4+1/4=2`; `8*(1/32)=1/4`; `32*(1/4)=8` | every `r1`, complete first-hit on chart O, C2, G1 and G4 |
| [Hilbert16StageCOriginNear.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16StageCOriginNear.lean) | `15/8+1/8=2`; `8*(1/64)=1/8`; `64*(1/8)=8` | every `r1`, complete first-hit on chart O, C2, G1 and G4 |
| [Hilbert16StageCOriginX32.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16StageCOriginX32.lean) | `31/16+1/16=2`; `8*(1/128)=1/16`; `128*(1/16)=8` | every `r1`, complete first-hit on chart O, C2, G1 and G4 |
| [Hilbert16StageCCompare.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16StageCCompare.lean) | `2*1-1^2=1`; `(1/4)/(1/32)=8`; `310*(1/40)=31/4` | every `eps`, Lohner tube, complete first-hit on chart O, C2, G1 and G4 |
| [Hilbert16StageCUniform.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16StageCUniform.lean) | `1/4+7/4=2`; `70*(1/40)=7/4`; `2*(4-1/4)/(1/16)=120` | Lohner tube, `eps>1/16`, shrinking interface, complete first-hit on chart O, C2, G1 and G4 |
| [Hilbert16StageCInterface.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16StageCInterface.lean) | `1+(1/2)*(1/2-2)=1/4`; `60*(1/40)=3/2`; `5*(1/4)/(1/16)^2=320` | Lohner tube, height-section flag, `eps>1/16`, complete first-hit on chart O, C2, G1 and G4 |
| [Hilbert16SepSpre.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16SepSpre.lean) | `2^48 * 2^{-48}=1`; `4/(1/2)=8`; `(3/16)(4-11/5)=27/80` | `dx_e` off the kill line, uniform-in-chi, Stage C, first-hit, G1 and G4 |
| [Hilbert16DxEOff.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16DxEOff.lean) | `1-5/8=3/8`; `1/(1/2)=2`; `11/5+2=21/5` | `lambda1 < -4`, `lambda1` in `(-2, 0)`, Stage C, first-hit, G1 and G4 |
| [Hilbert16DxERay.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16DxERay.lean) | `2*1=2`; `(3/8)/2=3/16`; `11/5+2=21/5` | `lambda1` in `(-2, 0)`, Stage C, first-hit, G1 and G4 |
| [Hilbert16DxENear.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16DxENear.lean) | `3/4-5/8=1/8`; `1/(1/4)=4`; `11/5+3=26/5` | `lambda1` in `(-3/2, 0)`, Stage C, first-hit, G1 and G4 |
| [Hilbert16DxEOpen.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16DxEOpen.lean) | `1-(5/8)*(8/5)=0`; `11/5+5=36/5`; `3/16-1/20=11/80` | fixed `eps=1/16`, Stage C, first-hit, G1 and G4 |
| [Hilbert16WeightedSection.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16WeightedSection.lean) | `r q tau_q=-1`; `r q^2 tau_qq=1`; chart-O interface speed is proportional to `r1=q` | Negative assessment only: no physical C2 overlap, first-hit completeness, G1, or G4 |
| [Hilbert16QuasihomogeneousDichotomy.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16QuasihomogeneousDichotomy.lean) | Kill-sequence event and W-ratio log expansions; incompatibility of `b>=1` with `b<=0`; exact moving-section cancellation | Frozen-section one-scale no-go only: moving sections, multistage atlases, physical C2, G1, and G4 remain open |
| [Hilbert16LNFormatBarrier.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16LNFormatBarrier.lean) | Exact `tau=n`, log-W `=2n`, direct chain norm `=3n`, and strict norm growth | Direct-format obstruction only: normalized physical-return LN membership, uniform analytic domains, G3, and Hilbert XVI remain open |
| [Hilbert16AbelianReturnTransfer.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16AbelianReturnTransfer.lean) | Exact value/derivative epsilon thresholds, normalized first-order displacement identity, and preservation of a supplied positive margin | Analytic expansion, root cover, remainder bounds, DRR graphic membership, and singular endpoint capture remain external |
| [Hilbert16FoldLeading.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16FoldLeading.lean) | Fold I-map, lifted inner map `B=(x-r)^2+μ`, relative Z remainder `O(ε)`, zeta-rho split, inner/outer interface | Physical C2 off the lifted map, first-hit completeness, chart O, G1 and G4 |
| [Hilbert16ShrinkingRoot.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16ShrinkingRoot.lean) | Two-root I-map, `r1 -> 0` remainder, `C -> 1`, rescaled `B`, limiting `ξ' = r2(1-ξ)` | Outgoing first-hit of the large first-root section, uniform `a_min`, G1 and G4 |
| [Hilbert16CanonicalZeta.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16CanonicalZeta.lean) | Slow-line `V` factorization, `zeta(0)=-1`, limiting cubic, implicit `k` at `lambda1 ≠ 0`, linear jet | Physical C2, first-hit, G1 and G4; fold compact is Hilbert16FoldZeta |
| [Hilbert16FoldZeta.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16FoldZeta.lean) | Fold-wall disc identities `L=r^2`, `lambda1=-2 r`, `lambda1^2=4L`, `B_-=(x-r)^2` | Picard/Cauchy enclosure of `Z`, physical C2, first-hit, G1 and G4 |
| [Hilbert16PhysicalC2.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16PhysicalC2.lean) | Frozen-Z first-log-derivative and C2 remainder identities versus the lifted fold | `Z_x` bound, `sep>0`, uniform-in-`eps` majorant, first-hit, G1 and G4 |
| [Hilbert16ZXGap.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16ZXGap.lean) | Unfrozen first-log-derivative gap `eps^2 (2 x^3 Z + x^4 Z_x)`; `Z_x=0` recovers frozen | `Z_x` bound, `sep>0`, uniform-in-`eps` majorant, first-hit, G1 and G4 |
| [Hilbert16ZVBound.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16ZVBound.lean) | Termwise `q1_v`; product-rule wall numerator; quotient-rule remainder for `Z0` | fold `Z_x`, `sep>0`, first-hit, G1 and G4 |
| [Hilbert16ZSlowV.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16ZSlowV.lean) | Slow-line `V_v+ell=0`; cleared `ell Z_V + Z_v`; declared `ell` floor | fold I-map `Z_x`, `sep>0`, first-hit, G1 and G4 |
| [Hilbert16FoldZX.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16FoldZX.lean) | Matching `ell Z_x - Z_v nu`; declared rate `1/196`; `x=3/2` interior | `sep>0`, first-hit, G1 and G4 |
| [Hilbert16OutgoingCorridor.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16OutgoingCorridor.lean) | Cleared two-root I-map numerator; first-root wall `r1-d<0` once `r1<d` | Height-section first-hit, uniform `a_min`, G1 and G4 |
| [Hilbert16PostCorridor.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16PostCorridor.lean) | Restored `V=-eps x` margin; leading `q/eps^3=(x-r1)(x-r2)`; wall can fail while `x_*-r1>0` | Height-section first-hit, sealed `T-h=O(eps)`, G1 and G4 |
| [Hilbert16HeightEnvelope.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16HeightEnvelope.lean) | AM-GM `|q|` identity; `C=0` `T-h` conservation; exit envelope; `T_e>h_e` once `2 eps y0 < x^2` | Actual-field `T-h=O(eps)`, height-section first-hit, G1 and G4 |
| [Hilbert16QRatioC2.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16QRatioC2.lean) | Inner/outer gap and complete square at `lambda1=-2`; disc `-12(1+L)<0` | `Z` bound, height-section first-hit, G1 and G4 |
| [Hilbert16KZetaRemainder.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16KZetaRemainder.lean) | `C=0` normal `k` jet; cubic versus two-root plus `eps^4 x^3/3`; relative prefactor | `Z` bound small enough for `C=2+delta`, `C!=0` `|g_h|`, `T-h` along the orbit, first-hit, G1 and G4 |
| [Hilbert16KillZeta.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16KillZeta.lean) | `lambda1=-2` product `L=r1(2-r1)`; disc `4(1-L)`; two-root cubic | Picard/Cauchy enclosure of `Z`, usable `C=2+delta`, first-hit, G1 and G4 |
| [Hilbert16CancelledN.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16CancelledN.lean) | Cubic slow-line factor; lambda source `O(nu^2)`; `L` source `O(nu^4)` | `T-h` along the orbit, first-hit, G1 and G4 |
| [Hilbert16HeightMix.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16HeightMix.lean) | `C!=0` `ell`/`V` mixing; `V_v+ell=0`; first-order `g` jet independent of `h` | `T-h` along the orbit, first-hit, G1 and G4 |
| [Hilbert16OrbitTh.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16OrbitTh.lean) | Actual-versus-comparison `T_h` gap; equals `k-1` at flux touching | Integrated `T-h` orbit, first-hit, G1 and G4 |
| [Hilbert16ThIntegral.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16ThIntegral.lean) | Comparison-bootstrap `T-h` slope split and linear FTC; sqrt-prefactor identity | Lohner orbit, first-hit, G1 and G4 |
| [Hilbert16VhOrbit.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16VhOrbit.lean) | Cubic `q+f=0`, `hdot=-V h`, `g` jet, `Vdot=f+hg` | Matching-chart `E_out` first-hit, G1 and G4 |
| [Hilbert16EOutSection.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16EOutSection.lean) | Matching-chart `E_out` at `nu=0` is `V+rho`; embedding `x=rho/eps` maps to `-rho`; `E_sigma` at `nu=0` is `V-1+sigma rho h` | GRAZING `E_sigma` first-hit, uniform `eps->0`, G1 and G4 |
| [Hilbert16EOutEps.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16EOutEps.lean) | Matching `V(0)=-eps`, `h(0)=4 eps^3`; majorant `T=n^2/8`; kill-line matching slope | Uniform-in-`eps` theorem, GRAZING `E_sigma`, G1 and G4 |
| [Hilbert16EOutSpeed.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16EOutSpeed.lean) | Comparison `F=f+4 eps^3 g` expansion; `F_V=eps phi`; `phi(-eps)=eps^2+4 eps^3`; sample `T=(rho-eps)/(3 eps^3)` | Lohner for every `eps`, GRAZING `E_sigma`, G1 and G4 |
| [Hilbert16ESigmaSpeed.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16ESigmaSpeed.lean) | Incoming `F(0)=-4 eps^3(1+eps)`; `phi(0)=-2 eps(1-2 eps^2)`; sample `T=1/(4 eps^3(1+eps))` | Certified `E_sigma` first-hit, G1 and G4 |
| [Hilbert16ESigmaIn.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16ESigmaIn.lean) | Incoming start `V=0`; reverse `hdot=V h`; matching reverse slope; wall `V=1/4` | Certified `E_sigma` from `V=0`, G1 and G4 |
| [Hilbert16ESigmaHit.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16ESigmaHit.lean) | Restart `V=3/4`, `h=1/4`; `E_sigma` at `nu=0`; grouped HEIGHT-COMPARISON polynomial | GRAZING band from `V=0`, G1 and G4 |
| [Hilbert16ESigmaFrom0.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16ESigmaFrom0.lean) | Kill-line `f` factor; reverse `Vdot` split; `dh/dV` ratio; cubic Taylor numerator; `dE/dV>=55/79` | Lohner `E_sigma` from `V=0`, G1 and G4 |
| [Hilbert16ESigmaUnif.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16ESigmaUnif.lean) | Cancelled `I` at `eps=0` is `V^2/2`; `E_sigma` at `nu=0`; `1-rho V*=7/10` | Lohner for every `eps`, G1 and G4 |
| [Hilbert16ESigmaWall.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16ESigmaWall.lean) | Restart `V=1/4`, `h=1/40`; `E_sigma` at `nu=0` is `-121/160`; sample `E_sigma=-619519/819200` | Lohner `E_sigma` from `V=0`, G1 and G4 |
| [Hilbert16ESigmaBox.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16ESigmaBox.lean) | Cover `[1/50, 4/125]` is twelve slabs of `1/1000`; contains `h=1/40` | Whole wall `h`-interval, Lohner `E_sigma` from `V=0`, G1 and G4 |
| [Hilbert16ESigmaSpan.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16ESigmaSpan.lean) | Cover `[19/1000, 1/25]` is twenty-one slabs of `1/1000`; contains `h=1/40` | `L in {9/25, 1/16}` walls, Lohner `E_sigma` from `V=0`, G1 and G4 |
| [Hilbert16ESigmaPack.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16ESigmaPack.lean) | Cover `[17/1000, 7/200]` is eighteen slabs of `1/1000`; contains `h=1/40` | Lohner `E_sigma` from `V=0`, G1 and G4 |
| [Hilbert16ESigmaEps.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16ESigmaEps.lean) | Pack `{1/16, 1/20, 1/25}` sums to `61/400`; `h(0)=4 eps^3`; aligned compact `T=10` | Lohner `E_sigma` from `V=0`, uniform `eps`, G1 and G4 |
| [Hilbert16ESigmaOneshot.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16ESigmaOneshot.lean) | Compact `T=280*(1/4)=70`; short `T=50`; L-pack `9/25+1/16=169/400` | uniform `eps`, `Z_x`, G1 and G4 |
| [Hilbert16ESigmaOneshotEps.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16ESigmaOneshotEps.lean) | Pack `{1/16, 1/20, 1/25}` sums to `61/400`; `T=100` and `T=250`; `h(0)=4 eps^3` at `n=25` | uniform `eps`, `Z_x`, G1 and G4 |
| [Hilbert16ESigmaEpsSpan.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16ESigmaEpsSpan.lean) | Three slabs of `3/400` fill `[1/25, 1/16]`; contains `1/20`; L-pack `9/25+1/16=169/400` | every `eps`, Lohner from `V=0` on that compact, `Z_x`, G1 and G4 |
| [Hilbert16ESigmaEpsLo.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16ESigmaEpsLo.lean) | Six slabs of `1/128` fill `[1/64, 1/16]`; contains `1/32`; L-pack `9/25+1/16=169/400` | every `eps`, Lohner from `V=0` on that compact, `Z_x`, G1 and G4 |
| [hilbert16_formal_replay.py](../../benchmarks/hilbert16_formal_replay.py) | Six sealed rational interval-sign obligations from the declared first-root saddle rectangle | The interval evaluator, membership of the actual canonical coefficients in that envelope, and the physical passage proof |

The listed Mathlib Hilbert modules are imported
by the full analytic project. They
contain no `sorry`, custom axioms, or `native_decide`.

Reproduce the analytic-module builds from `formal/omnibias-analytic`:

```bash
lake build OmnibiasAnalytic.Dynamics.Hilbert16Rolle
lake build OmnibiasAnalytic.Dynamics.Hilbert16Parabola
lake build OmnibiasAnalytic.Dynamics.Hilbert16Resonance
lake build OmnibiasAnalytic.Dynamics.Hilbert16ReturnMap
lake build OmnibiasAnalytic.Dynamics.Hilbert16Scale
lake build OmnibiasAnalytic.Dynamics.Hilbert16ChiScale
lake build OmnibiasAnalytic.Dynamics.Hilbert16SaddleNode
lake build OmnibiasAnalytic.Dynamics.Hilbert16TwoBlowup
lake build OmnibiasAnalytic.Dynamics.Hilbert16ScaleDichotomy
lake build OmnibiasAnalytic.Dynamics.Hilbert16LNCell
lake build OmnibiasAnalytic.Dynamics.Hilbert16EntryExit
lake build OmnibiasAnalytic.Dynamics.Hilbert16StageB
lake build OmnibiasAnalytic.Dynamics.Hilbert16StageA
lake build OmnibiasAnalytic.Dynamics.Hilbert16ChiB
lake build OmnibiasAnalytic.Dynamics.Hilbert16DxELeading
lake build OmnibiasAnalytic.Dynamics.Hilbert16DxEUnif
lake build OmnibiasAnalytic.Dynamics.Hilbert16StageC
lake build OmnibiasAnalytic.Dynamics.Hilbert16StageCExit
lake build OmnibiasAnalytic.Dynamics.Hilbert16StageCTh
lake build OmnibiasAnalytic.Dynamics.Hilbert16StageCGap
lake build OmnibiasAnalytic.Dynamics.Hilbert16StageCEnv
lake build OmnibiasAnalytic.Dynamics.Hilbert16StageCIf
lake build OmnibiasAnalytic.Dynamics.Hilbert16StageCInt
lake build OmnibiasAnalytic.Dynamics.Hilbert16StageCLo
lake build OmnibiasAnalytic.Dynamics.Hilbert16StageCK
lake build OmnibiasAnalytic.Dynamics.Hilbert16StageCBoot
lake build OmnibiasAnalytic.Dynamics.Hilbert16StageCRect
lake build OmnibiasAnalytic.Dynamics.Hilbert16StageCHit
lake build OmnibiasAnalytic.Dynamics.Hilbert16StageCSec
lake build OmnibiasAnalytic.Dynamics.Hilbert16StageCOneshot
lake build OmnibiasAnalytic.Dynamics.Hilbert16StageCOneshotEps
lake build OmnibiasAnalytic.Dynamics.Hilbert16StageCEpsSpan
lake build OmnibiasAnalytic.Dynamics.Hilbert16StageCOrigin
lake build OmnibiasAnalytic.Dynamics.Hilbert16StageCOriginSpan
lake build OmnibiasAnalytic.Dynamics.Hilbert16StageCOriginIface
lake build OmnibiasAnalytic.Dynamics.Hilbert16StageCOriginNear
lake build OmnibiasAnalytic.Dynamics.Hilbert16StageCOriginX32
lake build OmnibiasAnalytic.Dynamics.Hilbert16StageCCompare
lake build OmnibiasAnalytic.Dynamics.Hilbert16StageCUniform
lake build OmnibiasAnalytic.Dynamics.Hilbert16StageCInterface
lake build OmnibiasAnalytic.Dynamics.Hilbert16SepSpre
lake build OmnibiasAnalytic.Dynamics.Hilbert16DxEOff
lake build OmnibiasAnalytic.Dynamics.Hilbert16DxERay
lake build OmnibiasAnalytic.Dynamics.Hilbert16DxENear
lake build OmnibiasAnalytic.Dynamics.Hilbert16DxEOpen
lake build OmnibiasAnalytic.Dynamics.Hilbert16WeightedSection
lake build OmnibiasAnalytic.Dynamics.Hilbert16FoldLeading
lake build OmnibiasAnalytic.Dynamics.Hilbert16ShrinkingRoot
lake build OmnibiasAnalytic.Dynamics.Hilbert16CanonicalZeta
lake build OmnibiasAnalytic.Dynamics.Hilbert16FoldZeta
lake build OmnibiasAnalytic.Dynamics.Hilbert16PhysicalC2
lake build OmnibiasAnalytic.Dynamics.Hilbert16ZXGap
lake build OmnibiasAnalytic.Dynamics.Hilbert16ZVBound
lake build OmnibiasAnalytic.Dynamics.Hilbert16ZSlowV
lake build OmnibiasAnalytic.Dynamics.Hilbert16FoldZX
lake build OmnibiasAnalytic.Dynamics.Hilbert16OutgoingCorridor
lake build OmnibiasAnalytic.Dynamics.Hilbert16PostCorridor
lake build OmnibiasAnalytic.Dynamics.Hilbert16HeightEnvelope
lake build OmnibiasAnalytic.Dynamics.Hilbert16QRatioC2
lake build OmnibiasAnalytic.Dynamics.Hilbert16KZetaRemainder
lake build OmnibiasAnalytic.Dynamics.Hilbert16KillZeta
lake build OmnibiasAnalytic.Dynamics.Hilbert16CancelledN
lake build OmnibiasAnalytic.Dynamics.Hilbert16HeightMix
lake build OmnibiasAnalytic.Dynamics.Hilbert16OrbitTh
lake build OmnibiasAnalytic.Dynamics.Hilbert16ThIntegral
lake build OmnibiasAnalytic.Dynamics.Hilbert16VhOrbit
lake build OmnibiasAnalytic.Dynamics.Hilbert16EOutSection
lake build OmnibiasAnalytic.Dynamics.Hilbert16EOutEps
lake build OmnibiasAnalytic.Dynamics.Hilbert16EOutSpeed
lake build OmnibiasAnalytic.Dynamics.Hilbert16ESigmaSpeed
lake build OmnibiasAnalytic.Dynamics.Hilbert16ESigmaIn
lake build OmnibiasAnalytic.Dynamics.Hilbert16ESigmaHit
lake build OmnibiasAnalytic.Dynamics.Hilbert16ESigmaFrom0
lake build OmnibiasAnalytic.Dynamics.Hilbert16ESigmaUnif
lake build OmnibiasAnalytic.Dynamics.Hilbert16ESigmaWall
lake build OmnibiasAnalytic.Dynamics.Hilbert16ESigmaBox
lake build OmnibiasAnalytic.Dynamics.Hilbert16ESigmaSpan
lake build OmnibiasAnalytic.Dynamics.Hilbert16ESigmaPack
lake build OmnibiasAnalytic.Dynamics.Hilbert16ESigmaEps
lake build OmnibiasAnalytic.Dynamics.Hilbert16ESigmaOneshot
lake build OmnibiasAnalytic.Dynamics.Hilbert16ESigmaOneshotEps
lake build OmnibiasAnalytic.Dynamics.Hilbert16ESigmaEpsSpan
lake build OmnibiasAnalytic.Dynamics.Hilbert16ESigmaEpsLo
```

The implementation evidence command rebuilds all listed Hilbert modules, audits every
named theorem's axioms, and hashes the exact analytic and executable sources:

```bash
uv run --no-sync python -m benchmarks.hilbert16_program --lean
```

Its event, zero-count, curve, and surface certificates are checked by their
Python replay algorithms. These finite certificates do not acquire a Lean
verification tier merely because the conditional analytic modules build.

Reproduce the finite margin replay from the repository root:

```bash
uv run --no-sync python -m benchmarks.hilbert16_formal_replay --output artifacts/hilbert16/formal_margin_replay.json
```

The varying-detuning argument concerns the actual function
`G(kappa)=log D'(ti(kappa))-log Hreg'(ti(kappa))`, with every physical
coefficient held fixed. The
[actual singular estimate](HILBERT16-SINGULAR-KAPPA-JETS.md) controls
the two-scale remainder through two kappa derivatives and gives actual
input-label jets of order u squared. The
[fixed-field regular composition](HILBERT16-REGULAR-KAPPA-JETS.md)
then bounds the regular logarithmic curvature. These written estimates
prove strict convexity of G on the joint admitted interval. The
[synthesis](HILBERT16-VARYING-DETUNING.md) identifies its zeros with
the displacement's critical points there, and joins the outer comparisons
using a single convex sublevel interval on the connected admission domain.
These analytic premises are not inferred from any finite certificate.

The exact-resonance proof uses the separate negative-at-zeros implication.
Its regular value estimate is substituted only at actual displacement
zeros. Such a substitution alone does not prove convexity away from those
zeros. The fixed-field regular section jets supply the derivative
estimate without changing the splitting parameter.

## Joint jets and the fixed-field scale path

The two-scale path has `omega(s)=omega*exp(-s)`, `u(s)=u*exp(s)` and
fixed `epsilon=omega*u`. Thus its derivative operator is
`E=u*partial_u-omega*partial_omega`. Its second derivative includes
the acceleration of that path:

    E^2 R = omega^2 R_omega,omega - 2 omega u R_omega,u
              + u^2 R_u,u + omega R_omega + u R_u.

A straight directional Hessian omits the last two terms. For the actual
scale product `R=omega*u`, it would return `-2*epsilon`, although the
correct second derivative is zero.

The [scale-jet benchmark](../../benchmarks/hilbert16_scale_jets.py) uses
the new live input/parameter jets with both the velocity and acceleration
rows. PyTorch and JAX agree exactly on three dyadic scale pairs. Exact
realization algebra checks the weighted-derivative identities; an
operand-bound replay checks explicit rational coefficient evaluations in
Lean and rejects substitution of a different re-sealed source. The
benchmark deliberately detects the missing-acceleration error.

```bash
uv run --no-sync python benchmarks/hilbert16_scale_jets.py --lean --output artifacts/hilbert16/scale_jets.json
```

This checks the derivative mechanism used by the analytic remainder
estimate. It does not supply the actual ODE error bound. No neural
approximation is used in these proofs. Confluent representations can
stabilize a future fitted passage model, while validated continuation can
certify supported compact branches and joins. Either would still need
sound error enclosures and the singular tail argument before its output
could replace an actual passage estimate. Existing exact polynomial and
inverse-height formulas currently give a direct route without fitting.

## Exact coefficient, moving-event and composition identities

The [singular-kappa benchmark](../../benchmarks/hilbert16_singular_kappa.py)
checks 22 exact identities. These include the second normal-coefficient
jet derived from the physical off-line field, the fixed-epsilon scale
generator, radial height derivatives, the moving-cut second derivative,
and the actual regular-coordinate chain rules. It rejects both an omitted
second entrance-label jet and an unresolved symbolic expression. Normal
and optimized Python reports agree exactly.

```bash
uv run --no-sync python benchmarks/hilbert16_singular_kappa.py --output artifacts/hilbert16/singular_kappa.json
```

These finite symbolic checks establish their displayed identities. They
do not certify the uniform analytic remainder constants, the admissible
small-parameter cutoff, or the physical cycle bound.
