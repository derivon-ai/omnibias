# Hamiltonian rectangle amplitude enclosures

This API encloses a precisely defined charged Hamiltonian amplitude from a
replayed [static confinement certificate](gauge-charged-confinement.md).
It includes the entire link-spin Hilbert space and centers at the true
vacuum energy. The amplitude is not silently identified with a table of
isotropic Euclidean Wilson loops at an unspecified coupling.

```python
from fractions import Fraction
from omnibias.geometry.gauge.transfer.vacuum_fourier import su2_vacuum_fourier_bounds
from omnibias.geometry.gauge.transfer.charged_confinement import su2_static_confinement_bounds
from omnibias.geometry.gauge.transfer.charged_amplitude import (
    su2_hamiltonian_rectangle_enclosure,
    replay_su2_hamiltonian_rectangle_certificate,
)

vacuum = su2_vacuum_fourier_bounds(
    4, [(0, 1), (1, 2), (2, 3), (3, 0)],
    plaquettes=[(1, 2, 3, 4)], kappa=64,
)
charged = su2_static_confinement_bounds(vacuum['certificate'], 0, 2)
amplitude = su2_hamiltonian_rectangle_enclosure(
    charged['certificate'], time=Fraction(1, 4),
)
assert replay_su2_hamiltonian_rectangle_certificate(amplitude['certificate'])
assert amplitude['witness']['dimensionless_rectangle_area'] == '1/2'
assert not amplitude['isotropic_euclidean_wilson_identification_verified']
```

## Definition and proof

Let \(\gamma\) be the certified shortest path of length \(d\) between
the two source vertices, let \(\psi_0\) be the actual normalized neutral
vacuum, and put \(\Phi_\gamma=\psi_0U_\gamma/\sqrt2\). The charged
Hilbert norm uses the Hilbert--Schmidt matrix norm. Since \(U_\gamma\)
is unitary, \(\|\Phi_\gamma\|=1\). Define

\[
C_\gamma(T)=\langle\Phi_\gamma,
e^{-T(A_{\kappa,s,t}-E_0)}\Phi_\gamma\rangle,\qquad T\ge0.
\]

Here \(A_\kappa=aH^{\rm phys}\) has unit electric weights and
\(T\) is dimensionless, conjugate to \(aH^{\rm phys}\).
Its charged restriction and exact ground energy are those of the replayed
parent. The spectral measure of \(\Phi_\gamma\) is a positive probability
measure. Its support lies above the parent's lower energy bound
\(L=\kappa\rho d/2\). This gives \(C_\gamma(T)\le e^{-LT}\).

The exact vacuum-dressed path trial has energy mean
\(U=3\kappa d/8\): differentiation of the path inserts one traceless
SU(2) generator, so its cross term with the scalar vacuum derivative
vanishes. Its summed squared derivative is \(3/4\) on each path edge
and zero elsewhere. All vacuum terms cancel at the true \(E_0\).
Jensen's inequality for the convex function \(\lambda\mapsto e^{-T\lambda}\)
therefore gives \(C_\gamma(T)\ge e^{-UT}\). Thus

\[
e^{-3\kappa dT/8}\le C_\gamma(T)\le e^{-\kappa\rho dT/2}.
\]

For the standard Fourier radius, \(\rho=7/16\). The exponent grows with
the rectangle area \(dT\), uniformly in the finite surrounding volume.
These bounds do not require an overlap estimate with the charged groundstate.
They also do not assert existence of an asymptotic string-tension limit.

## Arithmetic and scope

The endpoint exponents are exact rational values. The rigorous interval
exponential supplies an outward lower endpoint for \(e^{-UT}\) and upper
endpoint for \(e^{-LT}\), with the conditional libm fallback refused.
At \(T=0\), the enclosure is exactly \([1,1]\). Underflow may produce a
zero lower endpoint; the report then does not assert numeric positivity.
A family certificate alone is refused, since it does not specify a graph,
path or particular amplitude.

Replay checks both nested parent certificates and every amplitude field.
`hamiltonian_rectangle_envelope_verified` concerns this written spectral
implication and its interval constants. Continuum, isotropic Euclidean
Wilson-measure identification, asymptotic string tension, Lean and Mathlib
flags remain false.
