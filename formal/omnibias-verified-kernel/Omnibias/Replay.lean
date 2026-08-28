/-
Generic finite-trace replay of a straight-line `ZInterval` derivation
(phase1-lean-replay).

Scope (read before extending). This module checks exactly the vocabulary
the interval `LDLᵀ` diagonal-pivot recurrence
(`omnibias.core.verified.eig_operator._ldlt_pivots`) needs to derive one
pivot: `literal` (a trusted input -- a matrix entry, or an off-diagonal
`L`-factor entry that needed `Interval.reciprocal`, which stays outside
this Mathlib-free kernel exactly as `Omnibias/LDLT.lean` already documents
for the division-bearing factorisation), `sub`, and `mul`. It is
deliberately **not** a speculative general-purpose interval interpreter:
`add` / `div` / `sqrt` / `reciprocal` replay is out of scope until a
concrete call site needs it and a matching kernel lemma exists.

Every `Interval` endpoint the Python bridge hands this checker is an exact
IEEE-754 double, i.e. an exact `numerator / 2 ^ exponent` (a *dyadic*
rational) -- so representing each recorded value as `(num, exp)` loses no
information. `sub` requires its two operands at a common exponent (the
larger of the two, reached by an exact multiply-by-a-power-of-two lift of
the coarser side); `mul` needs no lift at all, since
`(a / 2^ea) * (b / 2^eb) = (a * b) / 2 ^ (ea + eb)` is exact.

What this checks: that each recorded step's envelope genuinely *contains*
the exact composition of the *previously recorded* envelopes via the same
kernel `ZInterval.sub` / `ZInterval.mul` the rest of this project already
trusts (`mem_sub`, `mem_mul` in `Omnibias/Interval.lean`). It does **not**
re-derive the trusted `literal` leaves themselves (the matrix entries, or
the division-bearing `L`-factor values) -- those stay a trusted Python
input, the same trust boundary `Omnibias/LDLT.lean` already documents.
-/

import Omnibias.Interval

namespace Omnibias.Replay

/-- One step of a straight-line replay trace.

`kind`: `0` = literal (trusted input, no operands), `1` = `sub`,
`2` = `mul`. `arg0` / `arg1` index earlier steps of the *same* trace
(ignored for `literal`). `recLo` / `recHi` at `recExp` is the interval
Python's `Interval` class *actually* computed for this step
(`recLo / 2 ^ recExp`, `recHi / 2 ^ recExp`); the replay checks this
envelope, it never assumes it. -/
structure Step where
  kind : Nat
  arg0 : Nat
  arg1 : Nat
  recLo : Int
  recHi : Int
  recExp : Nat
deriving Repr, DecidableEq

/-- `x` at exponent `e` lifted to exponent `e' ≥ e`: an exact multiply by a
power of two (widening the denominator loses no precision). When
`e' < e` this returns `x` unchanged -- callers only ever lift to
`max _ _`, so that branch is unreachable in practice but keeps the
function total. -/
def liftTo (x : Int) (e e' : Nat) : Int :=
  if e ≤ e' then x * 2 ^ (e' - e) else x

/-- The exact `ZInterval` a step's operation produces from *prior recorded*
values, plus the exponent it is expressed at. `sub`: the max of the two
operand exponents (lifting the coarser operand first). `mul`: the sum of
the operand exponents (no lift needed). Any other `kind`, or an
out-of-range operand index, falls back to the step's own recorded value
(harmless: `stepOk` never calls `evalStep` for a `literal` step, and a
malformed index makes the containment check below compare the recorded
value to itself, which trivially holds -- the well-formedness of indices
is checked separately, structurally, by the Python exporter). -/
def evalStep (prior : List Step) (s : Step) : ZInterval × Nat :=
  match prior[s.arg0]?, prior[s.arg1]? with
  | some a, some b =>
      if s.kind = 1 then
        let e := max a.recExp b.recExp
        let al : ZInterval := ⟨liftTo a.recLo a.recExp e, liftTo a.recHi a.recExp e⟩
        let bl : ZInterval := ⟨liftTo b.recLo b.recExp e, liftTo b.recHi b.recExp e⟩
        (ZInterval.sub al bl, e)
      else if s.kind = 2 then
        (ZInterval.mul ⟨a.recLo, a.recHi⟩ ⟨b.recLo, b.recHi⟩, a.recExp + b.recExp)
      else (⟨s.recLo, s.recHi⟩, s.recExp)
  | _, _ => (⟨s.recLo, s.recHi⟩, s.recExp)

/-- Does the *recorded* interval (at `recExp`) contain the exactly
recomputed one (at exponent `ce`)? Both are lifted to their common
(larger) exponent first, so the comparison is an exact `Int` inequality --
no floating point, no Mathlib. -/
def containsComputed (rec : ZInterval) (recExp : Nat) (computed : ZInterval) (ce : Nat) : Bool :=
  let e := max recExp ce
  let rLo := liftTo rec.lo recExp e
  let rHi := liftTo rec.hi recExp e
  let cLo := liftTo computed.lo ce e
  let cHi := liftTo computed.hi ce e
  decide (rLo ≤ cLo) && decide (cHi ≤ rHi)

/-- Replay one step against the (already-processed) prior steps. A
`literal` step is trusted data and always passes; a `sub` / `mul` step
must have its recorded envelope contain the exact recomputation. -/
def stepOk (prior : List Step) (s : Step) : Bool :=
  if s.kind = 0 then true
  else
    let (computed, ce) := evalStep prior s
    containsComputed ⟨s.recLo, s.recHi⟩ s.recExp computed ce

/-- Replay every step of the trace in order, each against the steps
recorded strictly before it. -/
def replayOkFrom (seen : List Step) : List Step → Bool
  | [] => true
  | s :: rest => stepOk seen s && replayOkFrom (seen ++ [s]) rest

def replayOk (steps : List Step) : Bool := replayOkFrom [] steps

/-- **Soundness of one step's replay.** If `containsComputed` holds, any
value drawn from the exact recomputation (lifted to the common exponent)
lies in the recorded envelope (lifted the same way) -- the recorded
interval really is a sound consequence of the prior recorded values under
the trusted kernel `ZInterval` operations, not merely an unrelated
Python-reported number. -/
theorem containsComputed_sound {rec computed : ZInterval} {recExp ce : Nat}
    (h : containsComputed rec recExp computed ce = true) :
    ∀ x : Int,
      ZInterval.Mem x
          ⟨liftTo computed.lo ce (max recExp ce), liftTo computed.hi ce (max recExp ce)⟩ →
      ZInterval.Mem x ⟨liftTo rec.lo recExp (max recExp ce), liftTo rec.hi recExp (max recExp ce)⟩ := by
  simp only [containsComputed, Bool.and_eq_true, decide_eq_true_eq] at h
  intro x hx
  simp only [ZInterval.Mem] at hx ⊢
  omega

end Omnibias.Replay
