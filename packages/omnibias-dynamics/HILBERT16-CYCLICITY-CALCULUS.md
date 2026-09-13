# Executable displacement calculus and its physical boundary

The new `omnibias.dynamics.cyclicity` module supports exact polynomial
families, exact real confluent exponential polynomials, and actual finite-time
planar return maps produced by `return_maps`. These three sources have different
proof obligations. No polynomial approximation or finite Taylor prefix is
silently substituted for an actual return displacement.

## 1. Exact polynomial families and identity fibers

Write the supplied polynomial as `D(h,theta)=sum c_j(theta) h^j` over Q.
`identity_coefficients` computes every coefficient polynomial exactly. For a
fixed rational parameter, their simultaneous vanishing is equivalent to the
identity polynomial in h. Its zeros are not isolated on a nondegenerate height
interval, so the isolated-zero count is zero.

For nonidentity fixed fibers, the shared exact Sturm engine counts distinct
roots. Endpoint factors are divided out with exact polynomial arithmetic before
Sturm is invoked, and each endpoint is counted once regardless of multiplicity.
For parameter boxes, a strictly nonzero k-th height derivative on the entire
box gives at most k distinct zeros by repeated Rolle. Otherwise the exact
height degree supplies a bound valid on every fiber, including identity fibers.
This degree argument would be invalid for a truncated approximation with an
uncontrolled remainder; the API accepts exact polynomials only.

The binary cover checker recomputes all leaf certificates against the original
polynomial. The two closed children meet at their exact rational split. Height
splits sum their bounds; parameter splits take their maximum because a fixed
fiber lies in at least one child. A root on a shared height endpoint may be
counted twice, harmless for an upper bound. The independent root-box bound can
sharpen that sum. A missing strip, substituted polynomial, or altered result
fails replay. This proves finite polynomial-box coverage, not physical capture.

## 2. A terminating confluent exponential class

Let

    F(kappa) = sum_a p_a(kappa) exp(a*kappa),

with distinct real rational exponents a and nonzero rational polynomials p_a.
Define `dimension=sum_a(deg(p_a)+1)`. Equal exponents are merged exactly before
computing this dimension; cancellation and the identity sum are decided over Q.

**Theorem.** A nonzero supplied sum has at most `dimension-1` distinct real
zeros. The same bound holds counting multiplicities. The identity sum has no
isolated real zeros.

**Proof.** Choose the least exponent a. Multiplication by `exp(-a*kappa)`
preserves the zeros and their multiplicities. Its derivative is
`exp(-a*kappa)*(D-a)F`. The coefficient of each exponent b becomes
`p_b'+(b-a)*p_b`. For b=a its degree drops by one, with a constant annihilated;
for b!=a its leading coefficient is multiplied by b-a and its degree is
unchanged. Thus the total dimension decreases exactly one. At dimension one
the function is a nonzero constant times a real exponential and has no zeros.
Induction using Rolle on any finite interval gives the bound. Any alleged
larger set of real zeros lies in some finite interval, so the bound is global.
Repeated Rolle with root multiplicities gives the multiplicity version.

`certify_exponential_cyclicity` records this exact annihilation chain and
`verify_exponential_cyclicity` recomputes it from the source. The Mathlib
`Hilbert16ReturnMap` module checks the exponential-weight derivative and
zero-equivalence rules, alongside first/second derivative zero-count implications.
The full variable-dimension induction is the written proof above, not a claimed
formalized theorem. This is a classical finite-function-class argument made
executable here; no mathematical novelty is asserted.

The class contains expressions such as `Gamma+a*exp(-kappa)+b*epsilon*exp(kappa)`
at exact fixed coefficients and confluent terms `kappa^j exp(a*kappa)`.
It is closed under differentiation and the specific positive-exponential
division used above. It does not include arbitrary reciprocals, nonlinear
composition, Stokes cochains, or the unknown remainder of a physical passage.
Membership of a general Dulac/return map remains an open program obligation.

## 3. Actual regular return maps

For the supported physical consumer, the field is autonomous and planar,
the section is exactly `x=c`, and the initial embedding is exactly `(c,h)`.
The h parameter is checked to be absent from the field coefficients. A
polynomial guard, for example `y>0`, chooses an eligible section branch.
The source-bound event producer proves a unique first eligible positive hit
and its parameter derivatives, including event-time terms. On a connected
height interval the displacement is the actual `returned_y-h`.

A nonzero enclosure of that displacement excludes fixed points. A nonzero
actual first derivative gives at most one; a nonzero actual second derivative
gives at most two. Failure of these tests is unresolved, including for a center.
A fixed point returns to the same nonstationary state at positive time and
therefore determines a periodic orbit. Every counted orbit must close at this
first eligible event. Other itineraries, other section branches, cycles whose
period exceeds the checked horizon, and global graphic capture are separate.

## 4. Why the strict saddle-tail exponent cannot be kept at coalescence

The missing uniformity is visible already in an exact linear saddle model:

    Xdot=-epsilon*s*X, Ydot=r*Y,
    X(0)=1, Y(0)=exp(-kappa/epsilon), exit Y=1,
    X_exit=exp(-(s/r)*kappa).

At fixed positive r, the exit sensitivity is
`|dX_exit/dkappa|=(s/r)*exp(-(s/r)*kappa)`.
There cannot be constants C,gamma>0, independent of all sufficiently small
s>0, bounding this by `C*exp(-gamma*kappa)` for every sufficiently large
kappa. Choose s/r<gamma and let kappa tend to infinity: the ratio is
`s/(r*C)*exp((gamma-s/r)*kappa)` and diverges.

Consequently the strict first-root proof's decay exponent cannot simply be
held fixed when its eigenvalue ratio tends to zero. A natural new variable is
`chi=(s/r)*kappa`; its simultaneous zero/infinite limits require matching to
the central and endpoint charts. This is a precise obstruction to that proof
extension, not a counterexample to finite cyclicity or a realization of every
linear-saddle limit by the full canonical quadratic family.

## Remaining analytic gates

The [coalescing-capture companion](HILBERT16-COALESCING-CAPTURE.md) now
supplies the χ-atlas, the Lean obstruction, and a restricted
shrinking-rectangle first-derivative bound. The
[saddle-node](HILBERT16-SADDLE-NODE.md),
[shrinking-root](HILBERT16-SHRINKING-ROOT.md), and
[two-blow-up](HILBERT16-TWO-BLOWUP.md) follow-ups fail on those
same sequences. It **fails G1 and G4**:
super-small separation and `L -> 0` remain named falsifiers. The
[chart-cell ledger](HILBERT16-CHART-CELLS.md) records the labels; it
does not close G1.

- A uniform actual passage theorem through coalescing roots and the origin.
- Weighted remainder closure under the operations used by the zero-count proof.
- Correct identity strata for actual return maps rather than polynomial models.
- A phase/parameter atlas capturing every nearby cycle in the chosen graphic.
- Degree-controlled representation and coverage for arbitrary polynomial fields.

The linked [program ledger](HILBERT16-PROGRAM.md) records these as open,
separately from the executable finite calculations above.
