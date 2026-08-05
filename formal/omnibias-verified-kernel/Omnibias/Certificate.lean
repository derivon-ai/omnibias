/-
Certificate obligations discharged by the verified kernel.

The omnibias certificate format v1 (`omnibias.core.proof.certificate`) emits, among
others, two *finite, rational* proof obligations that are genuinely decidable and
hence kernel-checkable here, sorry-free:

* **spectral-gap positivity** -- a Birkhoff-Hopf / Perron certificate carries a
  subdominant-ratio upper bound `r = rn/rd < 1`; the spectral gap lower bound is
  `1 - r = (rd - rn)/rd`, whose positivity reduces to the `Int` fact `rn < rd`;
* **sign of an enclosed quantity** -- a CLM / CCF certificate encloses a scalar
  (e.g. `H ω₀(0)`, or a closure margin) in a rational interval `[lo, hi]`; the
  blow-up / closure obligation is `0 < lo` (certified) or `hi < 0` (excluded);
* **a rational equality** -- a special-number identity `p/q = r/s` (a Bernoulli
  recurrence, `ζ(1−2m) = −B₂ₘ/(2m)`, ...), scaled to a common positive `Int`
  denominator, reduces to `p·s − r·q = 0`, i.e. that difference lies in the point
  interval `[0, 0]`, discharged by `enclosed_quantity_eq`.

Rational data is scaled to a common positive `Int` denominator by the Python
bridge before instantiating these lemmas, so each obligation becomes an exact
`Int` statement closed by `omega`.

Scope (important). The kernel checks ONLY these finite arithmetic facts. That
`r = rn/rd` really is a valid subdominant-ratio upper bound for some transfer
operator, and that `[lo, hi]` really encloses the intended analytic quantity
(`H ω₀(0)`, a closure margin, ...), are computed by the *unverified* Python/numpy
bridge and are **trusted inputs** here, not kernel-checked. A kernel pass thus
certifies the implication "if the certificate's rational inputs are correct, then
the stated sign holds" -- never the underlying spectral or PDE statement itself.
-/

import Omnibias.Interval

namespace Omnibias

/-- The numerator of the spectral-gap lower bound `1 - r` for a subdominant-ratio
upper bound `r = rn/rd` (common denominator `rd > 0`). -/
def gapNumerator (rn rd : Int) : Int := rd - rn

/-- **Spectral-gap positivity (integer inequality only).** This certifies the
`Int` implication `rn < rd ⟹ 0 < rd - rn` and nothing more. It does NOT certify
that `rn/rd` is a genuine subdominant-ratio upper bound of any transfer operator
-- that ratio is produced by the unverified Python/numpy bridge and trusted as an
input. The denominator-positivity hypothesis `_hrd` only records certificate
well-formedness; the inequality itself follows from `rn < rd` alone. -/
theorem spectral_gap_pos {rn rd : Int} (_hrd : 0 < rd) (hlt : rn < rd) :
    0 < gapNumerator rn rd := by
  unfold gapNumerator
  omega

/-- The gap, as a `ZInterval` lower-bounded by its certified numerator. -/
def gapInterval (rn rd : Int) : ZInterval := ⟨gapNumerator rn rd, rd⟩

/-- **CLM / CCF sign obligation (certified).** A quantity enclosed in `I` with a
strictly positive lower endpoint is strictly positive -- the blow-up / closure
criterion holds for *every* value in the enclosure. -/
theorem enclosed_quantity_pos {x : Int} {I : ZInterval}
    (hx : ZInterval.Mem x I) (hlo : 0 < I.lo) : 0 < x :=
  ZInterval.pos_of_mem_of_lo_pos hx hlo

/-- **CLM sign obligation (excluded).** A quantity enclosed in `I` with a strictly
negative upper endpoint is strictly negative -- the criterion fails for *every*
value in the enclosure, so the property is rigorously ruled out. -/
theorem enclosed_quantity_neg {x : Int} {I : ZInterval}
    (hx : ZInterval.Mem x I) (hhi : I.hi < 0) : x < 0 :=
  ZInterval.neg_of_mem_of_hi_neg hx hhi

/-- **Rational-equality obligation (certified).** A quantity enclosed in a point
interval `I` (`I.hi ≤ I.lo`) equals its endpoint `I.lo`. Scaling a special-number
identity to a common positive `Int` denominator turns `p/q = r/s` into the exact
`Int` fact that `p·s − r·q` lies in `[0, 0]`; this lemma then certifies the equality
itself, not merely a sign. The mapping from the analytic quantity to the integer
data is a trusted Python input, exactly as for the sign obligations. -/
theorem enclosed_quantity_eq {x : Int} {I : ZInterval}
    (hx : ZInterval.Mem x I) (hpt : I.hi ≤ I.lo) : x = I.lo :=
  ZInterval.eq_of_mem_point hx hpt

end Omnibias
