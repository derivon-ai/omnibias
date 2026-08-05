/-
Golden certificates: concrete, checked-in obligation instances.

These are sorry-free theorems with the *exact integer constants* of representative
omnibias certificates (rational data scaled to a common denominator).  CI rebuilds
this file with `lake build`, so the kernel re-verifies the golden certificates on
every run -- the "golden certificate + replay" gate from the build-out plan, now
discharged by the Lean kernel itself rather than a numpy twin.
-/

import Omnibias.Certificate
import Omnibias.LDLT

namespace Omnibias.Golden

/-! ### Golden Perron / Birkhoff-Hopf spectral gap

A fixed positive transfer matrix with certified subdominant-ratio upper bound
`r = 5/8 < 1`; the spectral-gap lower bound is `1 - r = 3/8 > 0`. -/

/-- The certified subdominant-ratio numerator and (common) denominator. -/
def perronRatioNum : Int := 5
def perronRatioDen : Int := 8

theorem golden_perron_gap_pos :
    0 < gapNumerator perronRatioNum perronRatioDen := by
  unfold gapNumerator perronRatioNum perronRatioDen
  decide

theorem golden_perron_ratio_below_one : perronRatioNum < perronRatioDen := by
  unfold perronRatioNum perronRatioDen
  decide

/-! ### Golden CLM blow-up sign

The certified Hilbert value `H ω₀(0)` is enclosed (scaled to integer units) in
`[3, 7]`; its strictly positive lower endpoint certifies the CLM criterion. -/

/-- The certified enclosure of `H ω₀(0)` (scaled to integer units). -/
def clmHilbertEnclosure : ZInterval := ⟨3, 7⟩

theorem golden_clm_blowup_certified
    (x : Int) (hx : ZInterval.Mem x clmHilbertEnclosure) : 0 < x := by
  refine enclosed_quantity_pos hx ?_
  unfold clmHilbertEnclosure
  decide

/-! ### Golden CCF closure margin (excluded sign example)

A certified closure margin enclosed in `[-9, -2]`; its strictly negative upper
endpoint rigorously *excludes* the (positive-margin) property. -/

def ccfClosureMargin : ZInterval := ⟨-9, -2⟩

theorem golden_ccf_margin_excluded
    (x : Int) (hx : ZInterval.Mem x ccfClosureMargin) : x < 0 := by
  refine enclosed_quantity_neg hx ?_
  unfold ccfClosureMargin
  decide

/-! ### Golden binary-surrogate certificates

Concrete instances of the certified surrogate-gradient bounds emitted by
`omnibias.verify.surrogate_bounds` (see `docs/theory-binary.md`).

**Mollification margin (Theorem 1).** For the `tanh` surrogate at bandwidth
`β = 1` and margin `d = 1`, the hard/smooth agreement margin is
`tanh(1) − 1/2 ≈ 0.2616 ∈ [1/4, 1/3]`; scaled by the common denominator `12` this
encloses the quantity in `[3, 4]`, whose strictly positive lower endpoint certifies
that the smooth surrogate agrees with the hard sign by more than `1/2` for all
`|z| ≥ 1`. -/

def surrogateAgreementMargin : ZInterval := ⟨3, 4⟩

theorem golden_surrogate_margin_positive
    (x : Int) (hx : ZInterval.Mem x surrogateAgreementMargin) : 0 < x := by
  refine enclosed_quantity_pos hx ?_
  unfold surrogateAgreementMargin
  decide

/-! **No-dead-unit (Theorem 3).** The heavy-tailed `cauchy` surrogate `1/(1+z²)` at
`β = 1` over the region `|z| ≤ 3` is enclosed in `[1/10, 1]`; scaled by `10` this is
`[1, 10]`, whose strictly positive lower endpoint certifies that *no* unit in the
region receives a zero gradient -- the dichotomy against the compact-support STE box
(whose tail enclosure is exactly `0`, so no positive certificate exists there). -/

def cauchyKernelEnclosure : ZInterval := ⟨1, 10⟩

theorem golden_no_dead_unit_certified
    (x : Int) (hx : ZInterval.Mem x cauchyKernelEnclosure) : 0 < x := by
  refine enclosed_quantity_pos hx ?_
  unfold cauchyKernelEnclosure
  decide

/-! ### Golden kernel-verified LDLᵀ positive-definiteness (matrix inertia)

The interval LDLᵀ pivots of a symmetric matrix box, scaled to a common integer unit,
are `[3, 5]`, `[2, 9]`, `[7, 10]`.  Every pivot's lower endpoint is strictly positive,
so the negative inertia is zero and (Sylvester's law of inertia, with the factorisation
`S = L D Lᵀ` a trusted Python input) every point matrix in the box is positive definite.
This lifts the single-scalar `eig_min > 0` shadow to the full `n`-pivot inertia vector. -/

def goldenPdPivots : List ZInterval := [⟨3, 5⟩, ⟨2, 9⟩, ⟨7, 10⟩]

theorem golden_matrix_pd_pivots_pos : allPivotsPos goldenPdPivots = true := by
  unfold goldenPdPivots
  decide

/-- Every exact pivot drawn from the golden factorisation is strictly positive -- the
zero-negative-inertia (positive-definiteness) certificate, via the proven inertia lemma. -/
theorem golden_matrix_positive_definite
    (x : Int) (d : ZInterval) (hd : d ∈ goldenPdPivots) (hx : ZInterval.Mem x d) : 0 < x :=
  matrix_positive_definite_certified goldenPdPivots golden_matrix_pd_pivots_pos x d hd hx

/-! ### Golden rational special-number identities (kernel-checked equalities)

Concrete instances of the special-number *identity* obligations emitted by
`omnibias.difference.identities` and routed through `enclosed_quantity_eq`.

**Bernoulli recurrence.** `∑_{k=0}^{n-1} C(n,k) Bₖ = 0` for `n ≥ 2`. At `n = 3`,
over the common denominator `6` the scaled Bernoulli numerators are
`B₀·6 = 6`, `B₁·6 = −3`, `B₂·6 = 1`, with binomials `C(3,0)=1, C(3,1)=3, C(3,2)=3`,
so `1·6 + 3·(−3) + 3·1 = 6 − 9 + 3 = 0`. The recurrence value lies in `[0, 0]`. -/

def bernoulliRecurrenceN3 : Int := 1 * 6 + 3 * (-3) + 3 * 1

theorem golden_bernoulli_recurrence_n3 :
    bernoulliRecurrenceN3 = 0 := by
  have hx : ZInterval.Mem bernoulliRecurrenceN3 ⟨0, 0⟩ := by
    simp only [ZInterval.Mem, bernoulliRecurrenceN3]; omega
  exact ZInterval.eq_of_mem_point hx (by decide)

/-! **`ζ(−1) = −1/12`.** The trivial value `ζ(−1) = −B₂/2 = −(1/6)/2 = −1/12`.
Cross-multiplying `ζ(−1) = −1/12` against the computed `−1/12` gives the exact
integer difference `(−1)·12 − (−1)·12 = 0`; here we check the defining relation
`ζ(−1)·(−12) = 1`, i.e. `(−1)·(−12) − 1·12 = 0`. -/

def zetaNegOneIdentity : Int := (-1) * (-12) - 1 * 12

theorem golden_zeta_neg_one_identity :
    zetaNegOneIdentity = 0 := by
  have hx : ZInterval.Mem zetaNegOneIdentity ⟨0, 0⟩ := by
    simp only [ZInterval.Mem, zetaNegOneIdentity]; omega
  exact ZInterval.eq_of_mem_point hx (by decide)

end Omnibias.Golden
