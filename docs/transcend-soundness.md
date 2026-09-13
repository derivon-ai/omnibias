# Directed transcendental enclosures: September 2026 correction

The certified backend now evaluates elementary functions with mpmath's
**directed interval arithmetic**. A high-precision point value padded by an
assumed ulp allowance is no longer treated as a proof of enclosure.

Two concrete failures motivated the change:

* The former standard-normal CDF implementation returned [0, 5e-324] at
  x = −20. The actual CDF is approximately 2.7536241186062337e-89:
  1 + erf(...) had cancelled to zero even at the selected point precision.
* The former atan cap was 1.5707963267948966, which lies **below**
  \(\pi/2\). It cut off the correct answer for sufficiently large positive
  inputs. The cap now comes from the outward PI_IV/2 enclosure.

**Certificates relying on the former backend must be recomputed and replayed.**
An old digest confirms unchanged bytes, not the soundness of the analytic
enclosure used to create them. The backend stamp remains “mpmath” for API
compatibility; record the implementation revision with a research artifact.

## What the corrected path computes

Exp, log, sin and cos use directed interval functions; atan(x) uses interval
atan2(x,1). Tanh and sigmoid use stable algebraic compositions of interval
exponentials. Every intermediate remains an interval. The private context is
cached separately per thread, fixed at the declared precision, and is
independent of caller changes to the global mp or iv contexts.

The [mpmath interval documentation](https://mpmath.org/doc/current/contexts.html#arbitrary-precision-interval-arithmetic-iv)
states the inclusion contract and warns that interval support is incomplete.
This implementation uses its elementary interval primitives. It does not
assume that every high-level special function supports intervals.

Each already enclosed binary endpoint is converted to binary64 and compared
with its exact dyadic source. An inward-rounded endpoint moves one
representable number outward. This also handles gradual underflow and finite
overflow; mpmath documents that its directed to_float option alone does not
cover those cases.

For error functions, the implementation supplies the missing remainder:

\[
\operatorname{erf}(x)=\frac{2e^{-x^2}}{\sqrt\pi}
 \sum_{k\ge0}\frac{x(2x^2)^k}{(2k+1)!!},\qquad x\ge0.
\]

Terms are positive and consecutive ratios \(2x^2/(2k+3)\) decrease. Once a
ratio is below one, a geometric majorant encloses every omitted term. For
arguments above eight, repeated integration by parts gives

\[
\operatorname{erfc}(x)=\frac{e^{-x^2}}{x\sqrt\pi}
\left[\sum_{k<n}\frac{(-1)^k(2k-1)!!}{(2x^2)^k}+R_n(x)\right],
\quad
0\le(-1)^nR_n(x)\le\frac{(2n-1)!!}{(2x^2)^n}.
\]

Every retained partial sum therefore has a proved signed remainder;
the code chooses a useful one before terms start increasing. Negative
Gaussian tails use erfc directly and avoid subtraction from one.
A finite budget that establishes no remainder bound raises an error.

Bessel \(I_n\) now always uses its positive ascending series and geometric
tail bound. Consequently **arguments with absolute value above 600 refuse,
including when mpmath is installed**. The former high-precision point fallback
is gone. Orders must be actual integers; nonintegral inputs are not rounded
into a different certified function.

The optional stdlib fallback remains explicitly conditional on its stated
libm error assumption. Strict certificate mode refuses that path. The
positive Bessel series itself does not need mpmath and remains rigorous
without it.

## Verification scope

The new regression pack covers the two counterexamples, exact dyadic
underflow/overflow conversion, elementary functions on fixed and random
points, error-function transitions and tails, full interval samples, rejection
of unsupported Bessel arguments and noninteger orders, and thread isolation.
A separate test prevents accidental use of point functions as enclosure
oracles. Grid and random checks detect implementation errors; the directed
arithmetic contract and displayed remainder bounds supply the mathematical
inclusion argument.

No Lean verification of the transcendental implementation is asserted.
